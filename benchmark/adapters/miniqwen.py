from __future__ import annotations

import time
from pathlib import Path

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import BenchmarkSample, GenerationResult


class MiniQwenAdapter(ModelAdapter):
    def __init__(self, model_id: str, output_dir: Path, checkpoint: str, audio_encoder: str,
                 vision_encoder: str, mimi: str, device: str = "cuda", max_new_tokens: int = 256,
                 return_audio: bool = True, **kwargs: object):
        super().__init__(model_id, output_dir, **kwargs)
        self.checkpoint = checkpoint
        self.audio_encoder_path = audio_encoder
        self.vision_encoder_path = vision_encoder
        self.mimi_path = mimi
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.return_audio = return_audio

    def load(self) -> None:
        import torch
        from transformers import MimiModel
        from trainer.trainer_utils import (
            configure_token_ids,
            get_omni_model_class,
            infer_omni_model_arch,
            load_external_encoder_sidecars,
            load_omni_tokenizer,
        )

        model_class = get_omni_model_class(infer_omni_model_arch(self.checkpoint))
        self.tokenizer = load_omni_tokenizer(self.checkpoint)
        self.model = model_class.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16, audio_encoder_path=None, vision_model_path=None,
        )
        audio_encoder, audio_processor = model_class.load_sensevoice(self.audio_encoder_path)
        vision_encoder, vision_processor = model_class.load_vision(self.vision_encoder_path)
        object.__setattr__(self.model, "audio_encoder", audio_encoder)
        object.__setattr__(self.model, "audio_processor", audio_processor)
        object.__setattr__(self.model, "vision_encoder", vision_encoder)
        object.__setattr__(self.model, "vision_processor", vision_processor)
        load_external_encoder_sidecars(self.model, self.checkpoint)
        configure_token_ids(self.model.config, self.tokenizer)
        self.model.mimi_model = MimiModel.from_pretrained(self.mimi_path).eval()
        self.model.eval().to(self.device)
        if self.model.audio_encoder is not None:
            self.model.audio_encoder.to(self.device)
        if self.model.vision_encoder is not None:
            self.model.vision_encoder.to(self.device)

    def _inputs(self, sample: BenchmarkSample):
        import torch
        from PIL import Image
        from dataset.omni_dataset import OmniDataset
        from trainer.trainer_utils import format_audio_prompt, format_image_prompt

        prompt = sample.prompt
        audio_inputs = audio_lens = pixel_values = None
        image_paths = sample.image_paths
        if sample.audio:
            mel, valid_len = OmniDataset.process_audio(sample.audio, self.model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(self.device)
            audio_lens = torch.tensor([valid_len], device=self.device)
            audio_prompt = format_audio_prompt(self.model.config, max(valid_len, 1))
            prompt = (sample.prompt.strip() + "\n\n" + audio_prompt) if sample.prompt.strip() else audio_prompt
            if image_paths:
                prompt += "\n\n" + format_image_prompt(self.model.config, len(image_paths))
        elif image_paths:
            prompt = prompt + "\n\n" + format_image_prompt(self.model.config, len(image_paths))
        if image_paths:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            processed = self.model.vision_processor(images=images, return_tensors="pt")
            if len(images) == 1:
                pixel_values = {key: value.to(self.device) for key, value in processed.items()}
            else:
                # MiniQwen's multi-image path expects (B,N,C,H,W), whereas the
                # processor returns (N,C,H,W).
                pixel_values = processed["pixel_values"].unsqueeze(0).to(self.device)
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = torch.tensor(
            self.tokenizer(rendered, add_special_tokens=False)["input_ids"],
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        return input_ids, audio_inputs, audio_lens, pixel_values

    def generate(self, sample: BenchmarkSample, seed: int) -> GenerationResult:
        import soundfile as sf
        import torch
        from trainer.trainer_utils import setup_seed

        if len(sample.image_paths) > 1 and not bool(getattr(self.model.config, "use_mrope", False)):
            return GenerationResult(
                sample.sample_id, self.model_id, status="unsupported",
                error="this MiniQwen checkpoint predates multi-image/MRoPE support",
            )
        setup_seed(seed)
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        input_ids, audio_inputs, audio_lens, pixel_values = self._inputs(sample)
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        audio_frames, final_ids, first_text, first_audio = [], None, None, None
        with torch.inference_mode():
            stream = self.model.generate(
                input_ids,
                max_new_tokens=min(self.max_new_tokens, int(sample.metadata.get("max_new_tokens", self.max_new_tokens))),
                temperature=1.0, top_k=1, top_p=1.0,
                stream=True,
                return_audio_codes=self.return_audio and sample.metadata.get("require_audio", True),
                open_thinking=False,
                newline_token_id=newline_ids[0],
                pad_token_id=self.tokenizer.pad_token_id,
                audio_inputs=audio_inputs,
                audio_lens=audio_lens,
                pixel_values=pixel_values,
            )
            for ids, frame in stream:
                now = time.perf_counter()
                if ids is not None:
                    final_ids = ids
                    first_text = first_text or now - started
                if frame:
                    audio_frames.append(frame)
                    first_audio = first_audio or now - started
        text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True).strip() if final_ids is not None else ""
        audio_path = None
        if audio_frames:
            codes = torch.tensor(audio_frames, dtype=torch.long, device=self.device).T.unsqueeze(0)
            codes = torch.where(codes >= 2048, torch.zeros_like(codes), codes)
            with torch.inference_mode():
                waveform = self.model.mimi_model.decode(codes).audio_values.squeeze().float().cpu().numpy()
            audio_path = str(self.output_dir / f"{sample.sample_id}.wav")
            sf.write(audio_path, waveform, 24000)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
        status = "ok" if text or audio_path else "empty_output"
        return GenerationResult(
            sample.sample_id, self.model_id, text, audio_path, elapsed, first_text, first_audio,
            peak, input_tokens=input_ids.size(1), output_tokens=final_ids.size(1) if final_ids is not None else 0,
            status=status,
        )

    def generate_batch(self, samples: list[BenchmarkSample], seeds: list[int]) -> list[GenerationResult]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        if len(samples) <= 1 or self.model.config.audio_head_type != "main_codec_predictor":
            return super().generate_batch(samples, seeds)

        # Keep incompatible modalities out of the same encoder call. The
        # benchmark runner may pass a mixed window around task boundaries.
        groups = {}
        for index, sample in enumerate(samples):
            signature = (
                bool(sample.audio), len(sample.image_paths),
                bool(self.return_audio and sample.metadata.get("require_audio", True)),
                min(self.max_new_tokens, int(sample.metadata.get("max_new_tokens", self.max_new_tokens))),
            )
            groups.setdefault(signature, []).append(index)

        results = [None] * len(samples)
        for signature, indices in groups.items():
            if len(indices) == 1:
                index = indices[0]
                results[index] = self.generate(samples[index], seeds[index])
                continue

            group_started = time.perf_counter()
            prepared = [self._inputs(samples[index]) for index in indices]
            preprocessing_seconds = time.perf_counter() - group_started
            token_rows = [item[0].squeeze(0) for item in prepared]
            reversed_rows = [row.flip(0) for row in token_rows]
            input_ids = pad_sequence(
                reversed_rows, batch_first=True, padding_value=self.tokenizer.pad_token_id,
            ).flip(1)
            attention_mask = pad_sequence(
                [torch.ones_like(row).flip(0) for row in token_rows],
                batch_first=True,
                padding_value=0,
            ).flip(1)

            audio_inputs = audio_lens = None
            if signature[0]:
                feature_rows = [item[1].squeeze(0) for item in prepared]
                audio_inputs = pad_sequence(feature_rows, batch_first=True, padding_value=0).to(self.device)
                audio_lens = torch.cat([item[2].reshape(1) for item in prepared]).to(self.device)

            pixel_values = None
            if signature[1]:
                if isinstance(prepared[0][3], dict):
                    keys = prepared[0][3].keys()
                    pixel_values = {
                        key: torch.cat([item[3][key] for item in prepared], dim=0).to(self.device)
                        for key in keys
                    }
                else:
                    pixel_values = torch.cat([item[3] for item in prepared], dim=0).to(self.device)

            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if len(newline_ids) != 1:
                raise ValueError(f"newline must encode to one token, got {newline_ids}")
            outputs = self.model.batch_generate(
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=signature[3],
                temperature=1.0, top_k=1, top_p=1.0,
                return_audio_codes=signature[2],
                seeds=[seeds[index] for index in indices],
                open_thinking=False,
                newline_token_id=newline_ids[0],
                pad_token_id=self.tokenizer.pad_token_id,
                audio_inputs=audio_inputs,
                audio_lens=audio_lens,
                pixel_values=pixel_values,
            )
            peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
            decoded = []
            for local_index, (index, output) in enumerate(zip(indices, outputs)):
                token_ids = output["token_ids"]
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                audio_path = None
                frames = output["audio_frames"]
                if frames:
                    import soundfile as sf
                    codes = torch.tensor(frames, dtype=torch.long, device=self.device).T.unsqueeze(0)
                    codes = torch.where(codes >= 2048, torch.zeros_like(codes), codes)
                    with torch.inference_mode():
                        waveform = self.model.mimi_model.decode(codes).audio_values.squeeze().float().cpu().numpy()
                    audio_path = str(self.output_dir / f"{samples[index].sample_id}.wav")
                    sf.write(audio_path, waveform, 24000)
                decoded.append((local_index, index, output, text, audio_path))
            batch_wall_seconds = time.perf_counter() - group_started
            for local_index, index, output, text, audio_path in decoded:
                results[index] = GenerationResult(
                    samples[index].sample_id,
                    self.model_id,
                    text,
                    audio_path,
                    batch_wall_seconds,
                    (
                        preprocessing_seconds + output["first_token_seconds"]
                        if output["first_token_seconds"] is not None else None
                    ),
                    (
                        preprocessing_seconds + output["first_audio_seconds"]
                        if output["first_audio_seconds"] is not None else None
                    ),
                    peak,
                    input_tokens=int(attention_mask[local_index].sum().item()),
                    output_tokens=len(output["token_ids"]),
                    status="ok" if text or audio_path else "empty_output",
                    metadata={
                        "native_batch_size": len(indices),
                        "batch_wall_seconds": batch_wall_seconds,
                    },
                )
        return results
