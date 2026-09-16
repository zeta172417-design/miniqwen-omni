from __future__ import annotations

import sys
import time
from pathlib import Path

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import BenchmarkSample, GenerationResult


class NoValidSnacFrames(ValueError):
    pass


def sanitize_snac_codes(codes, codebook_size: int = 4096):
    """Truncate hierarchical SNAC codes before the first invalid audio frame.

    Mini-Omni2 can occasionally emit an audio special token (>= 4096) in a
    residual stream. Passing that value into SNAC's embedding causes a device
    assertion and poisons the accelerator context for every remaining sample.
    SNAC-24k uses 1/2/4 code rates, so all streams are cut at the same real
    audio-frame boundary.
    """
    if not isinstance(codes, (list, tuple)) or len(codes) != 3:
        raise ValueError(f"expected three hierarchical SNAC code tensors, got {type(codes).__name__}")
    rates = (1, 2, 4)
    frames = min(int(code.shape[-1]) // rate for code, rate in zip(codes, rates))
    if frames <= 0:
        raise NoValidSnacFrames("SNAC generation produced no complete audio frame")

    valid_frames = None
    for code, rate in zip(codes, rates):
        usable = code[..., : frames * rate]
        valid = ((usable >= 0) & (usable < codebook_size)).reshape(-1, frames, rate).all(dim=0).all(dim=-1)
        valid_frames = valid if valid_frames is None else valid_frames & valid
    invalid = (~valid_frames).nonzero(as_tuple=False)
    if invalid.numel():
        frames = int(invalid[0].item())
    if frames <= 0:
        raise NoValidSnacFrames("SNAC generation starts with an invalid codec token")
    return [code[..., : frames * rate].long() for code, rate in zip(codes, rates)]


class MiniOmni2Adapter(ModelAdapter):
    """Thin wrapper around the authors' inference code.

    Mini-Omni2's public vision path accepts an image plus a spoken question;
    manifests must therefore provide ``audio`` for image samples.
    """

    def __init__(self, model_id: str, output_dir: Path, source: str, checkpoint: str,
                 device: str = "cuda:0", max_new_tokens: int = 2048,
                 quiet: bool = True, **kwargs: object):
        super().__init__(model_id, output_dir, **kwargs)
        self.source, self.checkpoint = Path(source).resolve(), checkpoint
        self.device, self.max_new_tokens = device, max_new_tokens
        self.quiet = quiet

    def load(self) -> None:
        sys.path.insert(0, str(self.source))
        import inference
        import inference_vision
        if self.quiet:
            inference.tqdm = lambda iterable, *args, **kwargs: iterable
            inference_vision.tqdm = lambda iterable, *args, **kwargs: iterable
        from inference_vision import OmniVisionInference
        self.engine = OmniVisionInference(self.checkpoint, self.device)
        raw_decode = self.engine.snacmodel.decode
        codebook_size = int(getattr(self.engine.snacmodel, "codebook_size", 4096))

        def safe_decode(codes):
            import torch
            try:
                clean_codes = sanitize_snac_codes(codes, codebook_size)
            except NoValidSnacFrames:
                # Preserve the generated text while recording this sample as
                # having no valid speech. The placeholder only lets upstream
                # inference finish; it is removed before returning the result.
                self._audio_decode_invalid = True
                return torch.zeros((1, 1, 2400), dtype=torch.float32, device=codes[0].device)
            return raw_decode(clean_codes)

        self.engine.snacmodel.decode = safe_decode

    def generate(self, sample: BenchmarkSample, seed: int) -> GenerationResult:
        import torch
        from inference import (
            A1_A2_batch, A1_T1, T1_A2, _asr, _pad_a, get_input_ids_TA,
            get_input_ids_whisper, get_input_ids_whisper_ATBatch, load_audio,
        )

        torch.manual_seed(seed)
        self._audio_decode_invalid = False
        output = str(self.output_dir / f"{sample.sample_id}.wav")
        Path(output).unlink(missing_ok=True)
        started = time.perf_counter()
        try:
            if sample.image:
                if not sample.audio:
                    raise ValueError("Mini-Omni2 vision evaluation requires a spoken question")
                chunks = self.engine.run_vision_AA_batch_stream(
                    sample.audio, sample.image, max_returned_tokens=self.max_new_tokens, top_k=1, save_path=output,
                )
                text = "".join(text_chunk for _, text_chunk in chunks).strip()
            elif sample.audio:
                mel, length = load_audio(sample.audio)
                if sample.metric == "wer":
                    audio_feature, input_ids = get_input_ids_whisper(
                        mel, length, self.engine.whispermodel, self.device,
                        special_token_a=_pad_a, special_token_t=_asr,
                    )
                    text = A1_T1(
                        self.engine.fabric, audio_feature, input_ids, length,
                        self.engine.model, self.engine.text_tokenizer, 0,
                    ).strip()
                    output = None
                else:
                    audio_feature, input_ids = get_input_ids_whisper_ATBatch(
                        mel, length, self.engine.whispermodel, self.device,
                    )
                    text = A1_A2_batch(
                        self.engine.fabric, audio_feature, input_ids, length, self.engine.model,
                        self.engine.text_tokenizer, 0, self.engine.snacmodel, out_dir=str(self.output_dir),
                    ).strip()
                    upstream = self.output_dir / "A1-A2-batch" / "00.wav"
                    target = self.output_dir / f"{sample.sample_id}.wav"
                    if upstream.exists():
                        upstream.replace(target)
                        output = str(target)
                    else:
                        output = None
            else:
                input_ids = get_input_ids_TA(sample.prompt, self.engine.text_tokenizer)
                text = T1_A2(
                    self.engine.fabric, input_ids, self.engine.model, self.engine.text_tokenizer,
                    0, self.engine.snacmodel, out_dir=str(self.output_dir),
                ).strip()
                upstream = self.output_dir / "T1-A2" / "00.wav"
                target = self.output_dir / f"{sample.sample_id}.wav"
                if upstream.exists():
                    upstream.replace(target)
                    output = str(target)
                else:
                    output = None
        finally:
            # Upstream helpers clear this only on their success path. A failed
            # sample must not leave a stale cache for the next benchmark item.
            self.engine.model.clear_kv_cache()
        elapsed = time.perf_counter() - started
        if self._audio_decode_invalid:
            if output:
                Path(output).unlink(missing_ok=True)
            output = None
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
        return GenerationResult(
            sample.sample_id, self.model_id, text, output, elapsed,
            peak_memory_gb=peak, status="ok" if text else "empty_text",
        )
