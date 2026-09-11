#!/usr/bin/env python3
"""Deterministic teacher-forcing metrics for architecture experiments."""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

__package__ = "trainer"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".runtime/numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import logging as hf_logging

from dataset.omni_dataset import OmniDataset
from model.model_omni import MiniQwenOmni
from trainer.train_sft_omni import build_talker_attention_mask, omni_collate_fn
from trainer.trainer_utils import configure_token_ids, load_omni_tokenizer, setup_seed


def move_pixels(pixel_values, device):
    if pixel_values is None:
        return None
    if hasattr(pixel_values, "keys"):
        return {key: value.to(device) for key, value in pixel_values.items()}
    return pixel_values.to(device)


def roll_pixels(pixel_values):
    if pixel_values is None:
        return None
    if hasattr(pixel_values, "keys"):
        return {key: torch.roll(value, 1, 0) for key, value in pixel_values.items()}
    return torch.roll(pixel_values, 1, 0)


def empty_metrics():
    return {
        "text_nll_sum": 0.0,
        "text_correct": 0,
        "text_tokens": 0,
        "audio_nll_sum": [0.0] * 8,
        "audio_weighted_sum": [0.0] * 8,
        "audio_correct": [0] * 8,
        "audio_tokens": [0] * 8,
        "audio_stop_correct": [0] * 8,
        "audio_stop_tokens": [0] * 8,
        "audio_code_nll_sum": [0.0] * 8,
        "audio_code_tokens": [0] * 8,
        "main_nll_sum": 0.0,
        "main_tokens": 0,
        "main_correct": 0,
        "main_eos_correct": 0,
        "main_eos_tokens": 0,
        "main_code_nll_sum": 0.0,
        "main_code_tokens": 0,
        "main_code_correct": 0,
    }


def update_metrics(metrics, result, labels, audio_labels, audio_stop_token):
    text_mask = labels.ne(-100)
    text_targets = labels[text_mask]
    if text_targets.numel():
        logits = result.logits.reshape(-1, result.logits.size(-1)).float()
        metrics["text_nll_sum"] += float(F.cross_entropy(logits, text_targets, reduction="sum"))
        metrics["text_correct"] += int(logits.argmax(-1).eq(text_targets).sum())
        metrics["text_tokens"] += int(text_targets.numel())

    if result.audio_logits is None:
        return
    if getattr(result, "main_audio_logits", None) is not None:
        targets = audio_labels[:, 0].reshape(-1)
        valid = targets.ne(-100)
        logits = result.main_audio_logits.reshape(-1, result.main_audio_logits.size(-1)).float()
        if valid.any():
            selected_logits = logits[valid]
            selected_targets = targets[valid]
            losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
            predictions = selected_logits.argmax(-1)
            eos = selected_targets.eq(audio_stop_token)
            codes = selected_targets.lt(2048)
            metrics["main_nll_sum"] += float(losses.sum())
            metrics["main_tokens"] += int(selected_targets.numel())
            metrics["main_correct"] += int(predictions.eq(selected_targets).sum())
            metrics["main_eos_correct"] += int(predictions[eos].eq(selected_targets[eos]).sum())
            metrics["main_eos_tokens"] += int(eos.sum())
            metrics["main_code_nll_sum"] += float(losses[codes].sum())
            metrics["main_code_tokens"] += int(codes.sum())
            metrics["main_code_correct"] += int(predictions[codes].eq(selected_targets[codes]).sum())
        if result.residual_audio_logits is not None:
            frames = audio_labels.permute(0, 2, 1)[result.residual_audio_mask]
            for index, residual_logits in enumerate(result.residual_audio_logits, start=1):
                residual_logits = residual_logits.float()
                residual_targets = frames[:, index]
                losses = F.cross_entropy(residual_logits, residual_targets, reduction="none")
                metrics["audio_nll_sum"][index] += float(losses.sum())
                metrics["audio_correct"][index] += int(residual_logits.argmax(-1).eq(residual_targets).sum())
                metrics["audio_tokens"][index] += int(residual_targets.numel())
        return
    for layer, logits in enumerate(result.audio_logits):
        targets = audio_labels[:, layer].reshape(-1)
        valid = targets.ne(-100)
        if not valid.any():
            continue
        selected_logits = logits.reshape(-1, logits.size(-1))[valid].float()
        selected_targets = targets[valid]
        losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
        stop = selected_targets.eq(audio_stop_token)
        metrics["audio_nll_sum"][layer] += float(losses.sum())
        metrics["audio_weighted_sum"][layer] += float((losses * (1 + stop.float() * 9)).sum())
        metrics["audio_correct"][layer] += int(selected_logits.argmax(-1).eq(selected_targets).sum())
        metrics["audio_tokens"][layer] += int(selected_targets.numel())
        real_codes = selected_targets.lt(2048)
        metrics["audio_code_nll_sum"][layer] += float(losses[real_codes].sum())
        metrics["audio_code_tokens"][layer] += int(real_codes.sum())
        if stop.any():
            metrics["audio_stop_correct"][layer] += int(selected_logits[stop].argmax(-1).eq(selected_targets[stop]).sum())
            metrics["audio_stop_tokens"][layer] += int(stop.sum())


def finalise(metrics, residual_weight=0.3):
    text_count = metrics["text_tokens"]
    result = {
        "text_loss": metrics["text_nll_sum"] / max(text_count, 1),
        "text_accuracy": metrics["text_correct"] / max(text_count, 1),
        "text_tokens": text_count,
    }
    if metrics["main_tokens"]:
        main_loss = metrics["main_nll_sum"] / metrics["main_tokens"]
        main_accuracy = metrics["main_correct"] / metrics["main_tokens"]
        main_eos_accuracy = metrics["main_eos_correct"] / max(metrics["main_eos_tokens"], 1)
        residual_losses = [
            metrics["audio_nll_sum"][index] / max(metrics["audio_tokens"][index], 1)
            for index in range(1, 8)
        ]
        residual_accuracies = [
            metrics["audio_correct"][index] / max(metrics["audio_tokens"][index], 1)
            for index in range(1, 8)
        ]
        residual_active = [index for index in range(7) if metrics["audio_tokens"][index + 1]]
        residual_loss = sum(residual_losses[index] for index in residual_active) / max(len(residual_active), 1)
        residual_accuracy = sum(residual_accuracies[index] for index in residual_active) / max(len(residual_active), 1)
        main_code_loss = metrics["main_code_nll_sum"] / max(metrics["main_code_tokens"], 1)
        main_code_accuracy = metrics["main_code_correct"] / max(metrics["main_code_tokens"], 1)
        objective = main_loss + residual_weight * residual_loss
        by_codebook = [main_code_loss] + residual_losses
        accuracy_by_codebook = [main_code_accuracy] + residual_accuracies
        result.update({
            "audio_loss": objective,
            "audio_weighted_loss": objective,
            "audio_accuracy": sum(accuracy_by_codebook) / 8,
            "main_codec_loss": main_loss,
            "main_codec_accuracy": main_accuracy,
            "main_eos_accuracy": main_eos_accuracy,
            "residual_codec_loss": residual_loss,
            "residual_codec_accuracy": residual_accuracy,
            "audio_nll_excluding_eos": sum(by_codebook) / 8,
            "audio_loss_by_codebook": by_codebook,
            "audio_accuracy_by_codebook": accuracy_by_codebook,
            "audio_stop_accuracy_by_codebook": [main_eos_accuracy] + [0.0] * 7,
            "audio_tokens_by_codebook": [metrics["main_code_tokens"]] + metrics["audio_tokens"][1:],
        })
        return result

    layer_loss, layer_weighted, layer_accuracy, stop_accuracy = [], [], [], []
    for layer in range(8):
        count = metrics["audio_tokens"][layer]
        stop_count = metrics["audio_stop_tokens"][layer]
        layer_loss.append(metrics["audio_nll_sum"][layer] / max(count, 1))
        layer_weighted.append(metrics["audio_weighted_sum"][layer] / max(count, 1))
        layer_accuracy.append(metrics["audio_correct"][layer] / max(count, 1))
        stop_accuracy.append(metrics["audio_stop_correct"][layer] / max(stop_count, 1))
    active = [i for i, count in enumerate(metrics["audio_tokens"]) if count]
    result.update({
        "audio_loss": sum(layer_loss[i] for i in active) / max(len(active), 1),
        "audio_weighted_loss": sum(layer_weighted[i] for i in active) / max(len(active), 1),
        "audio_accuracy": sum(layer_accuracy[i] for i in active) / max(len(active), 1),
        "audio_loss_by_codebook": layer_loss,
        "audio_accuracy_by_codebook": layer_accuracy,
        "audio_stop_accuracy_by_codebook": stop_accuracy,
        "audio_tokens_by_codebook": metrics["audio_tokens"],
        "audio_nll_excluding_eos": sum(
            metrics["audio_code_nll_sum"][index] / max(metrics["audio_code_tokens"][index], 1)
            for index in active
        ) / max(len(active), 1),
    })
    return result


def model_pass(model, batch, device, kind, condition="correct"):
    input_ids, attention_mask, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    audio_labels = audio_labels.to(device)
    audio_lens = audio_lens.to(device)
    audio_inputs = audio_inputs.to(device) if audio_inputs is not None else None
    pixel_values = move_pixels(pixel_values, device)
    spk_dtype = next(model.talker.spk_proj.parameters()).dtype
    spk_emb = spk_emb.to(device=device, dtype=spk_dtype)

    if condition.startswith("speaker_only"):
        input_ids = input_ids.clone()
        audio_ids = input_ids[:, :8]
        supervised_positions = audio_labels.ne(-100).any(dim=1)
        speaker_positions = audio_ids[:, 0].eq(model.audio_spk_token)
        populated_audio = audio_ids.ne(model.audio_pad_token).any(dim=1)
        reference_positions = populated_audio & ~supervised_positions & ~speaker_positions
        audio_ids.masked_fill_(reference_positions.unsqueeze(1), model.audio_pad_token)

    if input_ids.size(0) > 1:
        if condition == "missing_audio" and audio_inputs is not None:
            # Rolling variable-length waveforms would no longer match the
            # number of audio markers in each text row. Zero ablation preserves
            # alignment while directly measuring whether the model uses audio.
            audio_inputs = torch.zeros_like(audio_inputs)
            audio_lens = torch.zeros_like(audio_lens)
        elif condition == "shuffled_image":
            pixel_values = roll_pixels(pixel_values)
        elif condition in {"shuffled_speaker", "speaker_only_shuffled"}:
            spk_emb = torch.roll(spk_emb, 1, 0)

    output_audio = bool(audio_labels.ne(-100).any())
    result = model(
        input_ids,
        attention_mask=attention_mask,
        talker_attention_mask=build_talker_attention_mask(attention_mask, audio_labels),
        use_cache=False,
        audio_inputs=audio_inputs,
        audio_lens=audio_lens,
        pixel_values=pixel_values,
        spk_emb=spk_emb,
        output_audio_logits=output_audio,
        audio_targets=audio_labels,
        text_logits_mask=labels.ne(-100),
    )
    return result, labels, audio_labels


def evaluate_dataset(model, tokenizer, args, kind, path):
    max_length = 512 if kind == "t2a" else 768
    dataset = OmniDataset(
        str(path), tokenizer,
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=max_length,
        image_token_len=model.config.image_token_len,
        max_images=model.config.max_images,
        use_modality_boundaries=model.config.use_modality_boundaries,
        use_talker_ref_boundaries=model.config.use_talker_ref_boundaries,
        audio_head_type=model.config.audio_head_type,
        audio_bos_token=model.config.audio_bos_token,
        audio_start_special_token=model.config.audio_start_special_token,
        audio_end_special_token=model.config.audio_end_special_token,
        vision_start_special_token=model.config.vision_start_special_token,
        vision_end_special_token=model.config.vision_end_special_token,
        audio_ref_start_token=model.config.audio_ref_start_token,
        audio_ref_end_token=model.config.audio_ref_end_token,
        scheduled_sampling=0,
        training=False,
    )
    sample_count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    dataset = Subset(dataset, range(sample_count))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(omni_collate_fn, dynamic_padding=True, pad_to_multiple=8),
    )
    conditions = ["correct"]
    if kind == "a2a":
        conditions += ["missing_audio", "speaker_only_correct", "speaker_only_shuffled"]
    elif kind == "i2t":
        conditions += ["shuffled_image"]
    accumulators = {condition: empty_metrics() for condition in conditions}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            for condition in conditions:
                result, labels, audio_labels = model_pass(model, batch, args.device, kind, condition)
                update_metrics(accumulators[condition], result, labels, audio_labels, model.config.audio_stop_token)
            if batch_index % 8 == 0 or batch_index == len(loader):
                print(f"  {kind}: {min(batch_index * args.batch_size, sample_count)}/{sample_count}", flush=True)
    results = {
        condition: finalise(metrics, model.config.residual_codec_loss_weight)
        for condition, metrics in accumulators.items()
    }
    correct = results["correct"]
    if "missing_audio" in results:
        results["audio_input_gain"] = {
            "text_loss": results["missing_audio"]["text_loss"] - correct["text_loss"],
            "audio_weighted_loss": results["missing_audio"]["audio_weighted_loss"] - correct["audio_weighted_loss"],
            "audio_nll_excluding_eos": results["missing_audio"]["audio_nll_excluding_eos"] - correct["audio_nll_excluding_eos"],
        }
    if "speaker_only_shuffled" in results:
        results["speaker_only_gain"] = {
            "audio_weighted_loss": (
                results["speaker_only_shuffled"]["audio_weighted_loss"]
                - results["speaker_only_correct"]["audio_weighted_loss"]
            ),
            "audio_nll_excluding_eos": (
                results["speaker_only_shuffled"]["audio_nll_excluding_eos"]
                - results["speaker_only_correct"]["audio_nll_excluding_eos"]
            ),
        }
    if "shuffled_image" in results:
        results["vision_gain"] = {
            "text_loss": results["shuffled_image"]["text_loss"] - correct["text_loss"],
        }
    results["samples"] = sample_count
    results["seconds"] = time.perf_counter() - started
    return results


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=str(project_root / "dataset/arch_eval"))
    parser.add_argument("--audio-encoder-dir", default=str(project_root / "model/SenseVoiceSmall"))
    parser.add_argument("--vision-dir", default=str(project_root / "model/siglip2-base-p32-256-ve"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    setup_seed(42)
    hf_logging.set_verbosity_error()
    checkpoint = str(Path(args.checkpoint).resolve())
    tokenizer = load_omni_tokenizer(checkpoint)
    model = MiniQwenOmni.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        audio_encoder_path=None,
        vision_model_path=None,
    )
    configure_token_ids(model.config, tokenizer)
    audio_encoder, audio_processor = MiniQwenOmni.load_sensevoice(args.audio_encoder_dir)
    vision_encoder, vision_processor = MiniQwenOmni.load_vision(args.vision_dir)
    object.__setattr__(model, "audio_encoder", audio_encoder)
    object.__setattr__(model, "audio_processor", audio_processor)
    object.__setattr__(model, "vision_encoder", vision_encoder)
    object.__setattr__(model, "vision_processor", vision_processor)
    model = model.eval().to(args.device)
    if model.audio_encoder is not None:
        model.audio_encoder.to(args.device)
    if model.vision_encoder is not None:
        model.vision_encoder.to(args.device)
    if "cuda" in args.device:
        torch.cuda.reset_peak_memory_stats()

    data_dir = Path(args.data_dir)
    report = {
        "format_version": 1,
        "stage": args.stage,
        "checkpoint": checkpoint,
        "num_talker_hidden_layers": model.config.num_talker_hidden_layers,
        "talker_hidden_size": model.config.talker_hidden_size,
        "accept_hidden_layer": model.config.accept_hidden_layer,
        "use_mrope": model.config.use_mrope,
        "use_modality_boundaries": model.config.use_modality_boundaries,
        "use_talker_ref_boundaries": model.config.use_talker_ref_boundaries,
        "max_images": model.config.max_images,
        "audio_head_type": model.config.audio_head_type,
        "code_predictor_num_layers": model.config.code_predictor_num_layers,
        "code_predictor_hidden_size": model.config.code_predictor_hidden_size,
        "residual_codec_loss_weight": model.config.residual_codec_loss_weight,
        "datasets": {},
    }
    for kind in ("t2a", "a2a", "i2t"):
        print(f"Evaluating {kind} dev", flush=True)
        report["datasets"][kind] = evaluate_dataset(model, tokenizer, args, kind, data_dir / f"{kind}_dev.parquet")
    report["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3 if "cuda" in args.device else 0.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    compact = {
        kind: {
            "text": round(values["correct"]["text_loss"], 4),
            "audio": round(values["correct"]["audio_weighted_loss"], 4),
            "condition_gain": round(
                values.get("vision_gain", values.get("audio_input_gain", {})).get("text_loss", 0.0), 4
            ),
        }
        for kind, values in report["datasets"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Metrics saved: {output}")


if __name__ == "__main__":
    main()
