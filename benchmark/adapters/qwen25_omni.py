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


QWEN_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech."
)


def talker_token_budget(task: str, ceiling: int = 1536) -> int:
    """Return a label-independent safety cap for Qwen's 50 Hz audio branch."""
    requested = 1536 if task == "tts" else 600
    return min(int(ceiling), requested)


def qwen_conversation(content: list[dict[str, str]], return_audio: bool) -> list[dict[str, object]]:
    """Build the official Qwen conversation protocol for optional speech output."""
    conversation: list[dict[str, object]] = []
    if return_audio:
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": QWEN_AUDIO_SYSTEM_PROMPT}],
        })
    conversation.append({"role": "user", "content": content})
    return conversation


class Qwen25OmniAdapter(ModelAdapter):
    def __init__(self, model_id: str, output_dir: Path, checkpoint: str, device: str = "cuda",
                 speaker: str = "Chelsie", max_new_tokens: int = 256, return_audio: bool = True,
                 talker_max_new_tokens: int = 1536,
                 **kwargs: object):
        super().__init__(model_id, output_dir, **kwargs)
        self.checkpoint, self.device = checkpoint, device
        self.speaker, self.max_new_tokens, self.return_audio = speaker, max_new_tokens, return_audio
        self.talker_max_new_tokens = talker_max_new_tokens

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
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        content = []
        for image_path in sample.image_paths:
            content.append({"type": "image", "image": image_path})
        if sample.audio:
            content.append({"type": "audio", "audio": sample.audio})
        if sample.prompt:
            content.append({"type": "text", "text": sample.prompt})
        return_audio = self.return_audio and sample.metadata.get("require_audio", True)
        # The official model card requires this exact system prompt whenever
        # Talker is enabled. It is model protocol, not task-specific guidance.
        conversation = qwen_conversation(content, return_audio)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        ).to(self.device)
        inputs = inputs.to(self.model.dtype)
        with torch.inference_mode():
            thinker_limit = min(
                self.max_new_tokens,
                int(sample.metadata.get("max_new_tokens", self.max_new_tokens)),
            )
            # Natural-label duration must not influence model inference.
            talker_limit = talker_token_budget(sample.task, self.talker_max_new_tokens)
            generated = self.model.generate(
                **inputs, return_audio=return_audio, speaker=self.speaker,
                thinker_max_new_tokens=thinker_limit,
                talker_max_new_tokens=talker_limit,
                do_sample=False,
                # Keep Qwen's official stochastic Talker settings. The seed is
                # fixed above; forcing codec greedy decoding causes repetition.
                talker_do_sample=True,
                use_audio_in_video=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
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
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
        return GenerationResult(
            sample.sample_id, self.model_id, decoded, audio_path, elapsed,
            peak_memory_gb=peak, output_tokens=int(text_ids.numel()),
            status="ok" if decoded or audio_path else "empty_output",
        )
