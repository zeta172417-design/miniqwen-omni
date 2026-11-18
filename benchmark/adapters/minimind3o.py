from __future__ import annotations

import time
from pathlib import Path

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import BenchmarkSample, GenerationResult


class MiniMind3OAdapter(ModelAdapter):
    """Adapter for the official MiniMind-3O-MoE Transformers checkpoint."""

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
        from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True, local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            audio_encoder_path=None,
            vision_model_path=None,
        )
        audio_encoder, audio_processor = self.model.__class__.load_sensevoice(self.audio_encoder_path)
        vision_encoder, vision_processor = self.model.__class__.load_vision(self.vision_encoder_path)
        object.__setattr__(self.model, "audio_encoder", audio_encoder)
        object.__setattr__(self.model, "audio_processor", audio_processor)
        object.__setattr__(self.model, "vision_encoder", vision_encoder)
        object.__setattr__(self.model, "vision_processor", vision_processor)
        self.mimi = MimiModel.from_pretrained(self.mimi_path, local_files_only=True).eval().to(self.device)
        self.model.eval().to(self.device)
        if self.model.audio_encoder is not None:
            self.model.audio_encoder.to(self.device)
        if self.model.vision_encoder is not None:
            self.model.vision_encoder.to(self.device)

    def _inputs(self, sample: BenchmarkSample):
        import torch
        from PIL import Image
        from dataset.omni_dataset import OmniDataset

        prompt = sample.prompt
        audio_inputs = audio_lens = pixel_values = None
        if sample.audio:
            mel, valid_len = OmniDataset.process_audio(sample.audio, self.model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(self.device)
            audio_lens = torch.tensor([valid_len], device=self.device)
            audio_prompt = self.model.config.audio_special_token * max(valid_len, 1)
            prompt = (sample.prompt.strip() + "\n\n" + audio_prompt) if sample.prompt.strip() else audio_prompt
            if sample.image:
                prompt += "\n\n" + self.model.config.image_special_token * self.model.config.image_token_len
        elif sample.image:
            prompt += "\n\n" + self.model.config.image_special_token * self.model.config.image_token_len
        if sample.image:
            image = Image.open(sample.image).convert("RGB")
            pixel_values = {
                key: value.to(self.device)
                for key, value in self.model.vision_processor(images=image, return_tensors="pt").items()
            }
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        input_ids = torch.tensor(
            self.tokenizer(rendered, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        return input_ids, audio_inputs, audio_lens, pixel_values

    def generate(self, sample: BenchmarkSample, seed: int) -> GenerationResult:
        import soundfile as sf
        import torch

        if len(sample.image_paths) > 1:
            return GenerationResult(
                sample.sample_id, self.model_id, status="unsupported",
                error="MiniMind-3O public inference supports one image per request",
            )
        torch.manual_seed(seed)
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        input_ids, audio_inputs, audio_lens, pixel_values = self._inputs(sample)
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"MiniMind newline must encode to one token, got {newline_ids}")

        audio_frames, final_ids, first_text, first_audio = [], None, None, None
        with torch.inference_mode():
            stream = self.model.generate(
                input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=min(self.max_new_tokens, int(sample.metadata.get("max_new_tokens", self.max_new_tokens))),
                temperature=0.85, top_k=50, top_p=0.85, do_sample=True,
                stream=True,
                return_audio_codes=self.return_audio and sample.metadata.get("require_audio", True),
                open_thinking=False,
                enter_token_id=newline_ids[0],
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
                if frame and len(frame) == 8:
                    audio_frames.append(frame)
                    first_audio = first_audio or now - started
        text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True).strip() if final_ids is not None else ""
        audio_path = None
        if audio_frames:
            codes = torch.tensor(audio_frames, dtype=torch.long, device=self.device).T.unsqueeze(0)
            codes = torch.where(codes < 2048, codes, torch.zeros_like(codes))
            with torch.inference_mode():
                waveform = self.mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
            audio_path = str(self.output_dir / f"{sample.sample_id}.wav")
            sf.write(audio_path, waveform, 24000)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
        return GenerationResult(
            sample.sample_id,
            self.model_id,
            text,
            audio_path,
            elapsed,
            first_text,
            first_audio,
            peak,
            input_tokens=input_ids.size(1),
            output_tokens=final_ids.size(1) if final_ids is not None else 0,
            status="ok" if text or audio_path else "empty_output",
        )
