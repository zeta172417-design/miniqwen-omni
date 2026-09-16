"""Password-protected MiniQwen-Omni Gradio inference server."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import secrets
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import gradio as gr
import librosa
import numpy as np
import torch
from PIL import Image
from transformers import MimiModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.trainer_utils import (  # noqa: E402
    configure_token_ids,
    format_audio_prompt,
    format_image_prompt,
    get_omni_model_class,
    infer_omni_model_arch,
    load_omni_tokenizer,
)

LOGGER = logging.getLogger("miniqwen.web")
MODEL_LOCK = Lock()
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ModelRuntime:
    label: str
    model: Any
    tokenizer: Any


@dataclass
class Runtime:
    model: Any
    tokenizer: Any
    mimi_model: Any
    asr_model: Any
    voices: dict[str, dict[str, torch.Tensor]]
    device: str
    dtype: torch.dtype
    max_audio_seconds: float
    models: dict[str, ModelRuntime] = field(default_factory=dict)
    default_model: str = "v01"

    def select_model(self, model_key: str | None = None) -> ModelRuntime:
        if self.models:
            key = model_key if model_key in self.models else self.default_model
            return self.models[key]
        # Backward-compatible path for small unit-test runtimes.
        return ModelRuntime(label="MiniQwen-Omni", model=self.model, tokenizer=self.tokenizer)


def normalize_audio(samples: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("音频为空或格式无效")
    if np.issubdtype(samples.dtype, np.integer):
        limit = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / limit
    else:
        samples = samples.astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("音频包含 NaN 或 Inf")
    peak = float(np.abs(samples).max())
    if peak > 1.0:
        samples /= peak
    if sample_rate != target_rate:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_rate)
    return np.ascontiguousarray(samples, dtype=np.float32)


def file_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "path", None) or getattr(value, "name", None)


def first_image(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    for value in message.get("files") or []:
        path = file_path(value)
        if path and Path(path).suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def frames_to_mimi(frames: list[list[int]]) -> torch.Tensor | None:
    complete = [frame for frame in frames if frame and len(frame) == 8]
    if not complete:
        return None
    return torch.tensor(complete, dtype=torch.long).T.unsqueeze(0)


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def require_dir(path: str, label: str) -> str:
    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).is_dir():
        raise FileNotFoundError(f"{label}目录不存在: {resolved}")
    return resolved


def load_voices() -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    voice_dir = PROJECT_ROOT / "model" / "speaker"
    for filename in ("voices.pt", "voices_unseen.pt"):
        path = voice_dir / filename
        if not path.is_file():
            continue
        values = torch.load(path, map_location="cpu")
        for name, voice in values.items():
            result.setdefault(name, voice)
    return result


def load_runtime(args: argparse.Namespace) -> Runtime:
    model_path = require_dir(args.model_path, "V0.1 checkpoint")
    baseline_model_path = require_dir(args.baseline_model_path, "V0 checkpoint")
    audio_path = require_dir(args.audio_encoder, "SenseVoice")
    vision_path = require_dir(args.vision_model, "SigLIP2")
    mimi_path = require_dir(args.mimi_path, "Mimi")
    dtype = resolve_dtype(args.dtype, args.device)

    def load_core(key: str, label: str, checkpoint: str) -> ModelRuntime:
        LOGGER.info("Loading %s (%s) from %s", label, key, checkpoint)
        model_class = get_omni_model_class(infer_omni_model_arch(str(checkpoint)))
        tokenizer = load_omni_tokenizer(checkpoint)
        # Use the current local implementation so inference fixes do not depend
        # on an older remote-code snapshot copied into a training checkpoint.
        model = model_class.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            audio_encoder_path=None,
            vision_model_path=None,
        )
        configure_token_ids(model.config, tokenizer)
        model = model.eval().to(args.device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        LOGGER.info("Loaded %s: %.2fM parameters", label, parameters / 1e6)
        return ModelRuntime(label=label, model=model, tokenizer=tokenizer)

    models = {
        "v01": load_core("v01", args.model_label, model_path),
        "v0": load_core("v0", args.baseline_model_label, baseline_model_path),
    }
    model_class = get_omni_model_class("production")
    audio_encoder, audio_processor = model_class.load_sensevoice(audio_path)
    vision_encoder, vision_processor = model_class.load_vision(vision_path)
    # SenseVoice's loader lowers the root logger level globally; restore web logs.
    logging.getLogger().setLevel(logging.INFO)
    if audio_encoder is None or audio_processor is None:
        raise RuntimeError("SenseVoice加载失败，无法处理语音输入")
    if vision_encoder is None or vision_processor is None:
        raise RuntimeError("SigLIP2加载失败，无法处理图片输入")
    for bundle in models.values():
        object.__setattr__(bundle.model, "audio_encoder", audio_encoder)
        object.__setattr__(bundle.model, "audio_processor", audio_processor)
        object.__setattr__(bundle.model, "vision_encoder", vision_encoder)
        object.__setattr__(bundle.model, "vision_processor", vision_processor)
    audio_encoder.to(args.device)
    vision_encoder.to(args.device)

    LOGGER.info("Loading Mimi from %s", mimi_path)
    mimi_model = MimiModel.from_pretrained(mimi_path).eval().to(args.device)
    if torch.device(args.device).type != "cpu":
        mimi_model = mimi_model.to(dtype=dtype)

    asr_model = None
    if not args.disable_asr:
        LOGGER.info("Loading ASR display model on %s", args.asr_device)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from funasr import AutoModel

                asr_model = AutoModel(
                    model=audio_path,
                    trust_remote_code=True,
                    disable_update=True,
                    disable_pbar=True,
                    disable_log=True,
                    device=args.asr_device,
                )
        finally:
            root_logger.setLevel(previous_level)

    voices = load_voices()
    LOGGER.info(
        "Ready: models=%s | core=%s | device=%s | voices=%d",
        ", ".join(bundle.label for bundle in models.values()),
        dtype,
        args.device,
        len(voices),
    )
    default_bundle = models["v01"]
    return Runtime(
        model=default_bundle.model,
        tokenizer=default_bundle.tokenizer,
        mimi_model=mimi_model,
        asr_model=asr_model,
        voices=voices,
        device=args.device,
        dtype=dtype,
        max_audio_seconds=args.max_audio_seconds,
        models=models,
        default_model="v01",
    )


def prepare_audio(
    runtime: Runtime,
    audio: tuple[int, np.ndarray] | None,
    selected: ModelRuntime | None = None,
):
    if audio is None:
        return None, None, None, None
    sample_rate, raw_samples = audio
    samples = normalize_audio(raw_samples, int(sample_rate))
    duration = samples.size / 16000
    if duration > runtime.max_audio_seconds:
        raise ValueError(f"语音最长支持 {runtime.max_audio_seconds:g} 秒，当前约 {duration:.1f} 秒")

    selected = selected or runtime.select_model()
    inputs = selected.model.audio_processor(
        samples,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    mel = inputs.input_features.squeeze(0).unsqueeze(0).to(runtime.device)
    valid_len = max(1, int(inputs.attention_mask.sum().item()))
    audio_lens = torch.tensor([valid_len], device=runtime.device)

    asr_state: dict[str, str | None] = {"text": None, "error": None}
    asr_thread = None
    if runtime.asr_model is not None:
        def transcribe():
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess

                result = runtime.asr_model.generate(
                    input=samples.copy(),
                    cache={},
                    language="auto",
                    use_itn=True,
                    disable_pbar=True,
                    disable_log=True,
                )
                if result:
                    asr_state["text"] = rich_transcription_postprocess(result[0]["text"]).strip()
            except Exception as exc:  # ASR is informational and must not break inference.
                LOGGER.exception("ASR display failed")
                asr_state["error"] = str(exc)

        asr_thread = Thread(target=transcribe, daemon=True)
        asr_thread.start()
    asr_task = (asr_thread, asr_state) if asr_thread is not None else None
    return mel, audio_lens, valid_len, asr_task


def prepare_image(runtime: Runtime, path: str | None, selected: ModelRuntime | None = None):
    if path is None:
        return None
    with Image.open(path) as source:
        width, height = source.size
        if width * height > 40_000_000:
            raise ValueError("图片像素过大，请上传小于 4000 万像素的图片")
        image = source.convert("RGB").copy()
    selected = selected or runtime.select_model()
    return {
        name: value.to(runtime.device)
        for name, value in selected.model.vision_processor(images=image, return_tensors="pt").items()
    }


def voice_condition(runtime: Runtime, voice_name: str, selected: ModelRuntime | None = None):
    if voice_name == "default" or voice_name not in runtime.voices:
        return None, None
    voice = runtime.voices[voice_name]
    ref_codes = voice.get("ref_codes")
    if ref_codes is not None:
        ref_codes = ref_codes.unsqueeze(0).to(runtime.device)
    spk_emb = voice.get("spk_emb")
    if spk_emb is not None:
        selected = selected or runtime.select_model()
        spk_dtype = next(selected.model.talker.spk_proj.parameters()).dtype
        spk_emb = spk_emb.to(device=runtime.device, dtype=spk_dtype).unsqueeze(0)
    return ref_codes, spk_emb


def render_input(
    runtime: Runtime,
    history: list[dict],
    prompt: str,
    max_new_tokens: int,
    thinking: bool,
    selected: ModelRuntime | None = None,
):
    selected = selected or runtime.select_model()
    tokenizer = selected.tokenizer
    retained = list(history)
    while True:
        messages = retained + [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        limit = int(getattr(selected.model.config, "max_position_embeddings", 40960))
        if len(token_ids) + max_new_tokens <= limit:
            tensor = torch.tensor(token_ids, dtype=torch.long, device=runtime.device).unsqueeze(0)
            return tensor, retained
        if len(retained) >= 2:
            retained = retained[2:]
            continue
        raise ValueError(f"输入过长：{len(token_ids)} tokens，且至少需要预留 {max_new_tokens} tokens")


def decode_audio(runtime: Runtime, frames: list[list[int]]):
    codes = frames_to_mimi(frames)
    if codes is None:
        return None
    device = next(runtime.mimi_model.parameters()).device
    codes = torch.where(codes >= 2049, torch.zeros_like(codes), codes).to(device)
    with torch.inference_mode():
        audio = runtime.mimi_model.decode(codes).audio_values
    samples = audio.squeeze().float().cpu().numpy()
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 0.99:
        samples *= 0.95 / peak
    return 24000, np.ascontiguousarray(samples)


def stream_answer(
    runtime: Runtime,
    model_key: str,
    text: str,
    audio,
    image_path: str | None,
    history: list[dict],
    voice_name: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    thinking: bool,
    generate_audio: bool,
):
    started_at = time.perf_counter()
    selected = runtime.select_model(model_key)
    model = selected.model
    tokenizer = selected.tokenizer
    audio_inputs, audio_lens, audio_token_len, asr_task = prepare_audio(runtime, audio, selected)
    pixel_values = prepare_image(runtime, image_path, selected)
    prompt_parts = [text.strip()] if text.strip() else []
    if audio_token_len is not None:
        prompt_parts.append(format_audio_prompt(model.config, audio_token_len))
    if pixel_values is not None:
        prompt_parts.append(format_image_prompt(model.config))
    prompt = "\n\n".join(prompt_parts)
    input_ids, retained_history = render_input(
        runtime, history, prompt, max_new_tokens=max_new_tokens, thinking=thinking,
        selected=selected,
    )
    ref_codes, spk_emb = voice_condition(runtime, voice_name, selected)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError(f"换行符应编码为一个 token，实际为 {newline_ids}")

    frames: list[list[int]] = []
    decoded_length = 0
    first_text_at = None
    with MODEL_LOCK, torch.inference_mode():
        generator = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            rp=1.05,
            stream=True,
            use_cache=True,
            return_audio_codes=generate_audio,
            open_thinking=thinking,
            newline_token_id=newline_ids[0],
            pad_token_id=tokenizer.pad_token_id,
            audio_inputs=audio_inputs,
            audio_lens=audio_lens,
            pixel_values=pixel_values,
            ref_codes=ref_codes,
            spk_emb=spk_emb,
        )
        for generated, audio_frame in generator:
            chunk = None
            if generated is not None:
                answer = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                if answer and not answer.endswith("�") and len(answer) > decoded_length:
                    chunk = answer[decoded_length:]
                    decoded_length = len(answer)
                    first_text_at = first_text_at or time.perf_counter()
            if audio_frame:
                frames.append(audio_frame)
            yield chunk, None, None, retained_history

    asr_text = None
    if asr_task is not None:
        thread, state = asr_task
        thread.join(timeout=30)
        asr_text = state["text"]
    generation_elapsed = time.perf_counter() - started_at
    if generate_audio and frames:
        yield None, "decoding", asr_text, retained_history
        audio_result = decode_audio(runtime, frames)
        LOGGER.info(
            "Inference timing [%s]: first_text=%.2fs generation=%.2fs total=%.2fs audio_frames=%d",
            selected.label,
            (first_text_at - started_at) if first_text_at else -1,
            generation_elapsed,
            time.perf_counter() - started_at,
            len(frames),
        )
        yield None, audio_result, asr_text, retained_history
    else:
        LOGGER.info(
            "Inference timing [%s]: first_text=%.2fs total=%.2fs audio=off",
            selected.label,
            (first_text_at - started_at) if first_text_at else -1,
            time.perf_counter() - started_at,
        )
        yield None, None, asr_text, retained_history


def build_demo(runtime: Runtime, args: argparse.Namespace) -> gr.Blocks:
    voice_choices = [("默认音色", "default")] + [(name, name) for name in sorted(runtime.voices)]
    model_choices = [
        (bundle.label, key) for key, bundle in runtime.models.items()
    ] or [("MiniQwen-Omni", runtime.default_model)]

    def respond_core(text, audio, image, model_key, voice, chat_history, model_history, turns,
                     temperature, top_p, max_tokens, thinking, response_mode,
                     request: gr.Request):
        chat_history = list(chat_history or [])
        model_history = list(model_history or [])
        text = str(text or "").strip()
        image_path = file_path(image)
        if len(text) > args.max_text_chars:
            yield chat_history, None, model_history, f"❌ 文本超过 {args.max_text_chars} 字符"
            return
        if not text and audio is None and image_path is None:
            yield chat_history, None, model_history, "❌ 请先输入文字、选择图片或完成录音"
            return

        if image_path:
            chat_history.append({"role": "user", "content": {"path": image_path}})
        if audio is not None:
            display = "🎤 已发送一段语音"
        elif text:
            display = text
        else:
            display = "请描述这张图片。"
        chat_history.append({"role": "user", "content": display})
        user_message_index = len(chat_history) - 1
        chat_history.append({"role": "assistant", "content": "正在生成…"})
        yield chat_history, None, model_history, "⏳ 已收到请求，正在推理"

        response = ""
        asr_text = None
        final_audio = None
        last_ui_update = 0.0
        generate_audio = response_mode == "文本 + 语音"
        retained = model_history[-int(turns) * 2:] if int(turns) > 0 else []
        retained_history = retained
        selected = runtime.select_model(model_key)
        try:
            for chunk, audio_result, asr_result, retained_history in stream_answer(
                runtime=runtime,
                model_key=model_key,
                text=text or ("请描述这张图片。" if image_path and audio is None else ""),
                audio=audio,
                image_path=image_path,
                history=retained,
                voice_name=voice,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_tokens),
                thinking=bool(thinking),
                generate_audio=bool(generate_audio),
            ):
                if chunk:
                    response += chunk
                    chat_history[-1]["content"] = response
                    now = time.perf_counter()
                    if now - last_ui_update >= 0.08:
                        last_ui_update = now
                        yield chat_history, None, model_history, "✍️ 正在生成回复"
                if asr_result:
                    asr_text = asr_result
                if audio_result == "decoding":
                    yield chat_history, None, model_history, "🔊 正在组装语音回复"
                elif audio_result is not None:
                    final_audio = audio_result

            if not response:
                response = "（模型没有生成可显示的文本）"
                chat_history[-1]["content"] = response
            if audio is not None and asr_text:
                chat_history[user_message_index]["content"] = "🎤 " + asr_text
            memory_parts = []
            if text:
                memory_parts.append(text)
            if audio is not None:
                memory_parts.append("[语音转写] " + (asr_text or "未获得转写"))
            if image_path:
                memory_parts.append("[图片]")
            user_memory = "\n".join(memory_parts) or "多模态输入"
            model_history = retained_history + [
                {"role": "user", "content": user_memory},
                {"role": "assistant", "content": response},
            ]
            LOGGER.info(
                "Completed request for user=%s model=%s",
                getattr(request, "username", "unknown"), selected.label,
            )
            final_status = (
                f"✅ {selected.label} 完成；若浏览器未自动播放，请点击下方播放器的 ▶"
                if final_audio is not None else f"✅ {selected.label} 文本完成"
            )
            yield chat_history, final_audio, model_history, final_status
        except Exception as exc:
            LOGGER.exception("Inference failed for user=%s", getattr(request, "username", "unknown"))
            chat_history[-1]["content"] = f"推理失败：{type(exc).__name__}。请缩短输入后重试。"
            yield chat_history, None, model_history, "❌ 详细错误已写入服务器日志"

    def clear_chat():
        return [], None, [], "已清空对话"

    def respond_text_image(text, image, model_key, voice, chat_history, model_history, turns,
                           temperature, top_p, max_tokens, thinking, response_mode,
                           request: gr.Request):
        LOGGER.info(
            "Text/image submit for user=%s (text=%s image=%s)",
            getattr(request, "username", "unknown"),
            bool(str(text or "").strip()),
            image is not None,
        )
        yield from respond_core(
            text, None, image, model_key, voice, chat_history, model_history, turns,
            temperature, top_p, max_tokens, thinking, response_mode, request,
        )

    def respond_audio(audio, model_key, voice, chat_history, model_history, turns, temperature,
                      top_p, max_tokens, thinking, response_mode, request: gr.Request):
        LOGGER.info(
            "Audio submit for user=%s (has_audio=%s)",
            getattr(request, "username", "unknown"),
            audio is not None,
        )
        yield from respond_core(
            "", audio, None, model_key, voice, chat_history, model_history, turns,
            temperature, top_p, max_tokens, thinking, response_mode, request,
        )

    def switch_model(model_key):
        selected = runtime.select_model(model_key)
        return [], None, [], f"✅ 已切换到 {selected.label}；对话上下文已清空"

    css = """
    .gradio-container {max-width: 1080px !important; margin: auto !important;}
    #hero {text-align: center; margin: 0.4rem 0 0.8rem;}
    #hero h1 {font-size: 1.75rem; margin-bottom: 0.15rem;}
    #hero p {color: var(--body-text-color-subdued); margin: 0;}
    #chatbox {border-radius: 16px;}
    #status {min-height: 28px; text-align: center; font-weight: 600;}
    .input-card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 12px;}
    """
    with gr.Blocks(title="MiniQwen-Omni", css=css, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            '<div id="hero"><h1>MiniQwen-Omni</h1>'
            '<p>文本 · 麦克风 · 图片 → 流式文本 + 语音回复</p></div>'
        )
        chatbot = gr.Chatbot(
            type="messages",
            height=520,
            elem_id="chatbox",
            label="对话",
            placeholder="输入文本，或者上传语音/图片开始体验。",
            show_copy_button=True,
            show_share_button=False,
            allow_file_downloads=False,
        )
        model_history = gr.State([])
        default_label = runtime.select_model(runtime.default_model).label
        status = gr.Markdown(f"模型已就绪：{default_label}", elem_id="status")

        gr.Markdown("### 1. 选择模型与回复形式")
        with gr.Row(equal_height=True):
            model_choice = gr.Radio(
                model_choices,
                value=runtime.default_model,
                label="模型版本（切换会清空上下文）",
                scale=3,
            )
            response_mode = gr.Radio(
                [("仅文字（更快）", "快速文本"), ("文字 + 语音", "文本 + 语音")],
                value="文本 + 语音",
                label="模型如何回复",
                scale=2,
            )
            voice = gr.Dropdown(
                voice_choices,
                value="default",
                label="语音回复音色",
                scale=1,
            )

        gr.Markdown("### 2. 选择一种输入方式并发送")
        with gr.Tabs():
            with gr.Tab("文字 / 图片"):
                gr.Markdown("可以只发文字、只发图片，或者文字和图片一起发送。")
                with gr.Row(equal_height=True):
                    text_input = gr.Textbox(
                        label="文字",
                        placeholder="在这里输入问题；按 Enter 也可以发送",
                        lines=4,
                        max_lines=8,
                        scale=3,
                    )
                    image_in = gr.Image(
                        sources=["upload", "clipboard"],
                        type="filepath",
                        format="webp",
                        label="图片（可选，建议小于 5 MB）",
                        height=220,
                        scale=2,
                    )
                with gr.Row():
                    send_text = gr.Button("发送文字 / 图片", variant="primary")
                    gr.ClearButton([text_input, image_in], value="清空文字和图片")

            with gr.Tab("语音"):
                gr.Markdown("① 开始录音　② 停止录音并等待波形出现　③ 点击“发送这段语音”。无需填写文字。")
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    format="wav",
                    label=f"录音或上传音频（最长 {runtime.max_audio_seconds:g} 秒）",
                    max_length=runtime.max_audio_seconds,
                )
                with gr.Row():
                    send_audio = gr.Button("发送这段语音", variant="primary")
                    gr.ClearButton([audio_in], value="清除录音")

        gr.Markdown("### 3. 模型回复")
        audio_out = gr.Audio(
            label="模型语音回复（选择“文字 + 语音”时生成）",
            autoplay=True,
            format="wav",
            show_download_button=True,
        )
        with gr.Accordion("高级生成设置", open=False):
            with gr.Row():
                turns = gr.Dropdown([0, 2, 4, 6, 8], value=4, label="保留对话轮数")
                max_tokens = gr.Slider(32, 512, value=128, step=32, label="最大生成 tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Top-p")
                thinking = gr.Checkbox(value=bool(args.open_thinking), label="启用 thinking")
        with gr.Row():
            stop = gr.Button("停止生成", variant="stop")
            clear = gr.Button("清空对话")
            gr.Button("退出登录", link="/logout?all_session=false")

        common_inputs = [model_choice, voice, chatbot, model_history, turns, temperature,
                         top_p, max_tokens, thinking, response_mode]
        outputs = [chatbot, audio_out, model_history, status]
        text_event = send_text.click(
            respond_text_image,
            [text_input, image_in, *common_inputs],
            outputs,
            api_name="chat",
            concurrency_id="miniqwen-model",
            concurrency_limit=1,
            show_progress="hidden",
        )
        enter_event = text_input.submit(
            respond_text_image,
            [text_input, image_in, *common_inputs],
            outputs,
            api_name=False,
            concurrency_id="miniqwen-model",
            concurrency_limit=1,
            show_progress="hidden",
        )
        audio_event = send_audio.click(
            respond_audio,
            [audio_in, *common_inputs],
            outputs,
            api_name="chat_audio",
            concurrency_id="miniqwen-model",
            concurrency_limit=1,
            show_progress="hidden",
        )
        model_choice.change(
            switch_model,
            model_choice,
            outputs,
            queue=False,
            api_name=False,
        )
        stop.click(
            lambda: "⏹️ 已请求停止",
            None,
            status,
            cancels=[text_event, enter_event, audio_event],
            queue=False,
            api_name=False,
        )
        clear.click(clear_chat, None, outputs, queue=False, api_name=False)
    return demo


def auth_from_environment(args: argparse.Namespace):
    if args.no_auth:
        if args.share or args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("--no-auth 仅允许本机监听，不能与 --share 或公网监听一起使用")
        return None
    username = os.environ.get("MINIQWEN_WEB_USERNAME", "miniqwen")
    password = os.environ.get("MINIQWEN_WEB_PASSWORD", "")
    if len(password) < 8:
        raise ValueError("请设置至少 8 位的 MINIQWEN_WEB_PASSWORD；密码不会写入仓库")

    def authenticate(candidate_user: str, candidate_password: str) -> bool:
        return secrets.compare_digest(candidate_user, username) and secrets.compare_digest(
            candidate_password, password
        )

    return authenticate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniQwen-Omni password-protected Gradio server")
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "out/miniqwen_omni_full_main_codec_cp_v5/checkpoint"),
        help="默认的新模型（V0.1 Main codec + Code Predictor）",
    )
    parser.add_argument(
        "--baseline-model-path",
        default=str(PROJECT_ROOT / "out/miniqwen_omni_full/checkpoint"),
        help="用于网页A/B切换的V0基线模型",
    )
    parser.add_argument("--model-label", default="V0.1 · Main codec + Code Predictor")
    parser.add_argument("--baseline-model-label", default="V0 · Parallel 8-code Talker")
    parser.add_argument("--audio-encoder", default=str(PROJECT_ROOT / "model/SenseVoiceSmall"))
    parser.add_argument("--vision-model", default=str(PROJECT_ROOT / "model/siglip2-base-p32-256-ve"))
    parser.add_argument("--mimi-path", default=str(PROJECT_ROOT / "model/mimi"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", default=None, help="反向代理子路径，例如 /miniqwen")
    parser.add_argument("--share", action="store_true", help="创建临时 gradio.live HTTPS 地址")
    parser.add_argument("--no-auth", action="store_true", help="仅允许 127.0.0.1 本机调试")
    parser.add_argument("--disable-asr", action="store_true", help="不显示语音输入的 ASR 转写")
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--max-text-chars", type=int, default=4000)
    parser.add_argument("--max-upload-mb", type=int, default=20)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--ssl-certfile", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    warnings.filterwarnings("ignore", message=".*path not found: None.*")
    if args.max_audio_seconds <= 0 or args.max_text_chars <= 0:
        raise ValueError("输入长度限制必须大于 0")
    auth = auth_from_environment(args)
    runtime = load_runtime(args)
    demo = build_demo(runtime, args)
    LOGGER.info("Starting web server on %s:%d (share=%s)", args.host, args.port, args.share)
    demo.queue(api_open=False, max_size=args.queue_size, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        auth_message="MiniQwen-Omni 私有体验服务",
        show_error=False,
        show_api=False,
        quiet=False,
        max_threads=8,
        max_file_size=f"{args.max_upload_mb}mb",
        blocked_paths=[
            str(PROJECT_ROOT / name)
            for name in (".git", "dataset", "envs", "model", "out", "releases", "trainer/swanlog")
        ],
        root_path=args.root_path,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        enable_monitoring=False,
        strict_cors=True,
        state_session_capacity=128,
    )


if __name__ == "__main__":
    main()
