"""Shared training metrics for MiniQwen-Omni."""
import torch
import torch.nn.functional as F

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
