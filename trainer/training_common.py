"""Shared training components for MiniQwen Omni engines and evals."""
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from trainer.trainer_utils import Logger

warnings.filterwarnings("ignore")

TRAIN_MODULE_CHOICES = {
    'all', 'thinker', 'text_head', 'talker', 'talker_core',
    'speaker', 'audio_proj', 'vision_proj', 'audio_encoder', 'vision_encoder',
}

def configure_trainable_modules(model, specification):
    """Apply an explicit module freeze policy for reproducible ablations.

    An empty value preserves the production ``mode``/``freeze_backbone``
    behaviour. ``talker_core`` excludes ``spk_proj`` so a T2A-only stage cannot
    move the unused speaker projection through optimizer weight decay.
    """
    if not specification:
        return []
    requested = [part.strip() for part in specification.split(',') if part.strip()]
    unknown = sorted(set(requested) - TRAIN_MODULE_CHOICES)
    if unknown:
        raise ValueError(
            f'unknown --train_modules values: {unknown}; '
            f'choices={sorted(TRAIN_MODULE_CHOICES)}'
        )
    if 'all' in requested and len(requested) != 1:
        raise ValueError('--train_modules all cannot be combined with other module names')

    for parameter in model.parameters():
        parameter.requires_grad = False
    if requested == ['all']:
        for parameter in model.parameters():
            parameter.requires_grad = True
        vision_encoder = getattr(model, 'vision_encoder', None)
        if getattr(model, '_train_vision_encoder', False) and vision_encoder is not None:
            for name, parameter in vision_encoder.named_parameters():
                if name.startswith('vision_model.head.'):
                    parameter.requires_grad = False
        return requested

    modules = {
        'thinker': model.model,
        'text_head': model.lm_head,
        'talker': model.talker,
        'speaker': model.talker.spk_proj,
        'audio_proj': model.audio_proj,
        'vision_proj': model.vision_proj,
        'audio_encoder': getattr(model, 'audio_encoder', None),
        'vision_encoder': getattr(model, 'vision_encoder', None),
    }
    for name in requested:
        if name == 'talker_core':
            for parameter_name, parameter in model.talker.named_parameters():
                if not parameter_name.startswith('spk_proj.'):
                    parameter.requires_grad = True
        else:
            if modules[name] is None:
                raise ValueError(f'{name} is unavailable; check its model path')
            for parameter_name, parameter in modules[name].named_parameters():
                # MiniQwen consumes SigLIP's token-level last_hidden_state.
                # The pooling head is not on that graph and must remain frozen
                # or DDP correctly reports it as an unused trainable branch.
                unused_vision_pooler = (
                    name == 'vision_encoder' and
                    parameter_name.startswith('vision_model.head.')
                )
                parameter.requires_grad = not unused_vision_pooler
    return requested


def init_swanlab_tracking(args, ckp_data):
    """Initialize SwanLab without letting a stale run id block checkpoint resume."""
    import swanlab

    # `wandb_id` is read only for checkpoints produced before the SwanLab
    # naming cleanup; all new checkpoints use `swanlab_id`.
    previous_run_id = (ckp_data.get('swanlab_id') or ckp_data.get('wandb_id')) if ckp_data else None
    base_name = args.swanlab_run_name or (
        f"MiniQwen-Omni-E{args.epochs}-B{args.batch_size}-"
        f"QwenLR{args.qwen_learning_rate}-OmniLR{args.omni_learning_rate}"
    )
    common = {
        'project': args.swanlab_project,
        'name': base_name,
        'group': base_name,
        'config': {
            'stage_id': args.stage_id,
            'model_arch': getattr(args, 'model_arch', 'production'),
            'train_modules': getattr(args, 'train_modules', '') or 'legacy',
            'use_mrope': bool(getattr(args, 'use_mrope', 0)),
            'use_modality_boundaries': bool(getattr(args, 'use_modality_boundaries', 0)),
            'use_talker_ref_boundaries': bool(getattr(args, 'use_talker_ref_boundaries', 0)),
            'max_images': getattr(args, 'max_images', 1),
            'audio_head_type': getattr(args, 'audio_head_type', 'main_codec_predictor'),
            'code_predictor_num_layers': getattr(args, 'code_predictor_num_layers', 2),
            'code_predictor_hidden_size': getattr(args, 'code_predictor_hidden_size', 768),
            'residual_codec_loss_weight': getattr(args, 'residual_codec_loss_weight', 0.3),
            'train_audio_encoder': bool(getattr(args, 'train_audio_encoder', 0)),
            'train_vision_encoder': bool(getattr(args, 'train_vision_encoder', 0)),
            'audio_encoder_learning_rate': getattr(args, 'audio_encoder_learning_rate', 0.0),
            'vision_encoder_learning_rate': getattr(args, 'vision_encoder_learning_rate', 0.0),
            'checkpoint_epoch': ckp_data.get('epoch', 0) if ckp_data else 0,
            'checkpoint_step': ckp_data.get('step', 0) if ckp_data else 0,
            'resumed_from_run_id': previous_run_id,
        },
    }

    if previous_run_id:
        try:
            # `allow` resumes when the cloud run is still resumable and creates
            # a new run when it is not. Older code used `must`, so a completed
            # or stale SwanLab run could prevent valid model-state resumption.
            swanlab.init(**common, id=previous_run_id, resume='allow')
            Logger(f'SwanLab续接run: {previous_run_id}')
            return swanlab
        except Exception as exc:
            Logger(f'SwanLab云端run续接失败，改用新的分段run: {type(exc).__name__}: {exc}')
            # A failed init can leave partially-created SDK state behind.
            try:
                swanlab.finish()
            except Exception:
                pass
            segment_name = f"{base_name}-resume-e{ckp_data.get('epoch', 0) + 1}"
            try:
                swanlab.init(**{**common, 'name': segment_name}, resume='never', reinit=True)
                return swanlab
            except Exception as retry_exc:
                Logger(f'SwanLab云端不可用，训练指标暂存本地offline run: {type(retry_exc).__name__}: {retry_exc}')
                try:
                    swanlab.finish()
                except Exception:
                    pass
                swanlab.init(**{**common, 'name': segment_name}, mode='offline', resume='never', reinit=True)
                return swanlab

    swanlab.init(**common)
    return swanlab


def build_talker_attention_mask(thinker_attention_mask, audio_labels, input_ids=None, audio_pad_token=2049):
    """Keep text context plus every position participating in audio training."""
    if thinker_attention_mask.ndim != 2:
        raise ValueError(f'thinker_attention_mask must be (B,T), got {tuple(thinker_attention_mask.shape)}')
    if audio_labels.ndim != 3 or audio_labels.shape[0] != thinker_attention_mask.shape[0] or audio_labels.shape[2] != thinker_attention_mask.shape[1]:
        raise ValueError(f'audio_labels must be (B,8,T) aligned with thinker mask, got {tuple(audio_labels.shape)}')
    audio_positions = (audio_labels != -100).any(dim=1)
    if input_ids is not None:
        audio_positions |= input_ids[:, :8].ne(audio_pad_token).any(dim=1)
    return (thinker_attention_mask.bool() | audio_positions).to(dtype=thinker_attention_mask.dtype)


def distributed_audio_branch_enabled(audio_labels):
    """Make the conditional Talker graph identical on every DDP rank.

    Dynamic/truncated batches can leave one rank with no valid audio targets
    while another rank still has them.  Letting those ranks independently skip
    the Talker gives DDP a barrier on one rank and Talker gradient all-reduces
    on the others.  A one-scalar MAX keeps the optimization unchanged while
    making parameter participation consistent.
    """
    enabled = audio_labels.ne(-100).any().to(dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(enabled, op=dist.ReduceOp.MAX)
    return bool(enabled.item())


def compute_audio_losses(result, audio_labels, config):
    """Return objective, Main loss, residual mean and per-codebook losses."""
    zero = result.logits.sum() * 0
    if result.audio_logits is None:
        return zero, zero, zero, []
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    if config.audio_head_type == 'legacy_parallel_delay':
        layer_losses = []
        for layer, logits in enumerate(result.audio_logits):
            targets = audio_labels[:, layer].reshape(-1)
            losses = loss_fct(logits.reshape(-1, logits.size(-1)), targets)
            valid = targets.ne(-100)
            stop = targets.eq(config.audio_stop_token)
            if valid.any():
                layer_losses.append((losses * valid * (1 + stop * 9)).sum() / valid.sum())
        objective = torch.stack(layer_losses).mean() if layer_losses else zero
        return objective, objective, zero, layer_losses

    main_targets = audio_labels[:, 0].reshape(-1)
    main_logits = result.main_audio_logits.reshape(-1, result.main_audio_logits.size(-1))
    main_valid = main_targets.ne(-100)
    if main_valid.any():
        main_loss = loss_fct(main_logits, main_targets)[main_valid].mean()
    else:
        main_loss = main_logits.sum() * 0

    residual_losses = []
    if result.residual_audio_logits is not None:
        frames = audio_labels.permute(0, 2, 1)[result.residual_audio_mask]
        for index, logits in enumerate(result.residual_audio_logits):
            residual_losses.append(F.cross_entropy(logits, frames[:, index + 1]))
    residual_loss = torch.stack(residual_losses).mean() if residual_losses else main_loss.new_zeros(())
    objective = main_loss + config.residual_codec_loss_weight * residual_loss
    return objective, main_loss, residual_loss, residual_losses


def trim_batch_to_active_length(input_ids, attention_mask, labels, audio_labels, pad_to_multiple=8):
    """Drop only batch-wide trailing positions that cannot affect any loss.

    Every sample is stored at ``max_seq_len`` for simple dataset caching.  A
    random batch is usually much shorter, so running the model over the common
    all-padding suffix wastes most of the I2T compute.  The union below also
    includes audio targets because Talker supervision can extend beyond the
    Thinker attention mask.
    """
    active = attention_mask.bool() | labels.ne(-100) | audio_labels.ne(-100).any(dim=1)
    if not active.any():
        raise ValueError('batch has no active text/audio positions')
    active_columns = active.any(dim=0).nonzero(as_tuple=False)
    target_length = int(active_columns[-1].item()) + 1
    if pad_to_multiple > 1:
        target_length = min(
            input_ids.size(-1),
            ((target_length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple,
        )
    return (
        input_ids[..., :target_length],
        attention_mask[..., :target_length],
        labels[..., :target_length],
        audio_labels[..., :target_length],
    )


def omni_collate_fn(batch, dynamic_padding=True, pad_to_multiple=8):
    """自定义collate函数，处理变长audio_inputs和pixel_values"""
    input_ids, attention_mask, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    audio_labels = torch.stack(audio_labels)
    if dynamic_padding:
        input_ids, attention_mask, labels, audio_labels = trim_batch_to_active_length(
            input_ids,
            attention_mask,
            labels,
            audio_labels,
            pad_to_multiple=pad_to_multiple,
        )
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [a if a.size(1) == max_t else torch.nn.functional.pad(a, (0, 0, 0, max_t - a.size(1))) for a in valid_audios]
        audio_inputs = torch.cat(padded, dim=0)
    else:
        audio_inputs = None
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        max_images = max(p.size(0) for p in valid_images)
        padded_images = []
        for images in valid_images:
            padded = images.new_zeros((max_images, *images.shape[1:]))
            padded[:images.size(0)] = images
            padded_images.append(padded)
        pixel_values = torch.stack(padded_images)
    else:
        pixel_values = None
    spk_emb = torch.stack(spk_emb)
    return input_ids, attention_mask, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb
