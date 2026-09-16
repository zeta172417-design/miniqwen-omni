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
        if sample.audio:
            mel, valid_len = OmniDataset.process_audio(sample.audio, self.model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(self.device)
            audio_lens = torch.tensor([valid_len], device=self.device)
            audio_prompt = format_audio_prompt(self.model.config, max(valid_len, 1))
            prompt = (sample.prompt.strip() + "\n\n" + audio_prompt) if sample.prompt.strip() else audio_prompt
            if sample.image:
                prompt += "\n\n" + format_image_prompt(self.model.config)
        elif sample.image:
            prompt = prompt + "\n\n" + format_image_prompt(self.model.config)
        if sample.image:
            image = Image.open(sample.image).convert("RGB")
            pixel_values = {
                key: value.to(self.device)
                for key, value in self.model.vision_processor(images=image, return_tensors="pt").items()
            }
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

        setup_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        input_ids, audio_inputs, audio_lens, pixel_values = self._inputs(sample)
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        audio_frames, final_ids, first_text, first_audio = [], None, None, None
        started = time.perf_counter()
        with torch.inference_mode():
            stream = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
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
        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True).strip() if final_ids is not None else ""
        audio_path = None
        if audio_frames:
            codes = torch.tensor(audio_frames, dtype=torch.long, device=self.device).T.unsqueeze(0)
            codes = torch.where(codes >= 2048, torch.zeros_like(codes), codes)
            with torch.inference_mode():
                waveform = self.model.mimi_model.decode(codes).audio_values.squeeze().float().cpu().numpy()
            audio_path = str(self.output_dir / f"{sample.sample_id}.wav")
            sf.write(audio_path, waveform, 24000)
        peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.startswith("cuda") else None
        status = "ok" if text else "empty_text"
        return GenerationResult(
            sample.sample_id, self.model_id, text, audio_path, elapsed, first_text, first_audio,
            peak, input_tokens=input_ids.size(1), output_tokens=final_ids.size(1) if final_ids is not None else 0,
            status=status,
        )
