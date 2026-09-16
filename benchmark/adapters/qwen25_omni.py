from __future__ import annotations

import time
from pathlib import Path

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import BenchmarkSample, GenerationResult


def generated_tokens_only(text_ids, input_ids):
    """Remove the chat prompt when ``generate`` returns prompt + continuation."""
    prompt_length = int(input_ids.shape[-1])
    if text_ids.shape[-1] < prompt_length:
        return text_ids
    prefix = text_ids[..., :prompt_length]
    if prefix.shape == input_ids.shape and bool((prefix == input_ids).all().item()):
        return text_ids[..., prompt_length:]
    return text_ids


class Qwen25OmniAdapter(ModelAdapter):
    def __init__(self, model_id: str, output_dir: Path, checkpoint: str, device: str = "cuda",
                 speaker: str = "Chelsie", max_new_tokens: int = 256, return_audio: bool = True,
                 **kwargs: object):
        super().__init__(model_id, output_dir, **kwargs)
        self.checkpoint, self.device = checkpoint, device
        self.speaker, self.max_new_tokens, self.return_audio = speaker, max_new_tokens, return_audio

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.checkpoint, local_files_only=True, use_fast=False,
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16, local_files_only=True,
        ).eval().to(self.device)

    def generate(self, sample: BenchmarkSample, seed: int) -> GenerationResult:
        import soundfile as sf
        import torch
        from qwen_omni_utils import process_mm_info

        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        content = []
        if sample.image:
            content.append({"type": "image", "image": sample.image})
        if sample.audio:
            content.append({"type": "audio", "audio": sample.audio})
        if sample.prompt:
            content.append({"type": "text", "text": sample.prompt})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                "perceiving auditory and visual inputs, as well as generating text and speech."
            )}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        ).to(self.device)
        inputs = inputs.to(self.model.dtype)
        started = time.perf_counter()
        with torch.inference_mode():
            return_audio = self.return_audio and sample.metadata.get("require_audio", True)
            generated = self.model.generate(
                **inputs, return_audio=return_audio, speaker=self.speaker,
                max_new_tokens=self.max_new_tokens, use_audio_in_video=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        if return_audio:
            text_ids, waveform = generated
        else:
            text_ids, waveform = generated, None
        text_ids = generated_tokens_only(text_ids, inputs.input_ids)
        decoded = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        audio_path = None
        if waveform is not None:
            audio_path = str(self.output_dir / f"{sample.sample_id}.wav")
            sf.write(audio_path, waveform.reshape(-1).float().cpu().numpy(), 24000)
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
        return GenerationResult(
            sample.sample_id, self.model_id, decoded, audio_path, elapsed,
            peak_memory_gb=peak, output_tokens=int(text_ids.numel()), status="ok" if decoded else "empty_text",
        )
