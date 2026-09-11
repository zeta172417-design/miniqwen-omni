import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets
import argparse
import time
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from functools import partial
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_omni import OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_omni_model, omni_checkpoint, SkipBatchSampler, log_model_params

warnings.filterwarnings('ignore')

TRAINING_FORMAT_VERSION = 5

TRAIN_MODULE_CHOICES = {
    'all', 'thinker', 'text_head', 'talker', 'talker_core',
    'speaker', 'audio_proj', 'vision_proj',
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
        return requested

    modules = {
        'thinker': model.model,
        'text_head': model.lm_head,
        'talker': model.talker,
        'speaker': model.talker.spk_proj,
        'audio_proj': model.audio_proj,
        'vision_proj': model.vision_proj,
    }
    for name in requested:
        if name == 'talker_core':
            for parameter_name, parameter in model.talker.named_parameters():
                if not parameter_name.startswith('spk_proj.'):
                    parameter.requires_grad = True
        else:
            for parameter in modules[name].parameters():
                parameter.requires_grad = True
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
            'train_modules': getattr(args, 'train_modules', '') or 'legacy',
            'use_mrope': bool(getattr(args, 'use_mrope', 0)),
            'use_modality_boundaries': bool(getattr(args, 'use_modality_boundaries', 0)),
            'use_talker_ref_boundaries': bool(getattr(args, 'use_talker_ref_boundaries', 0)),
            'max_images': getattr(args, 'max_images', 1),
            'audio_head_type': getattr(args, 'audio_head_type', 'main_codec_predictor'),
            'code_predictor_num_layers': getattr(args, 'code_predictor_num_layers', 2),
            'code_predictor_hidden_size': getattr(args, 'code_predictor_hidden_size', 768),
            'residual_codec_loss_weight': getattr(args, 'residual_codec_loss_weight', 0.3),
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


def build_talker_attention_mask(thinker_attention_mask, audio_labels):
    """Keep text context plus every position participating in audio training."""
    if thinker_attention_mask.ndim != 2:
        raise ValueError(f'thinker_attention_mask must be (B,T), got {tuple(thinker_attention_mask.shape)}')
    if audio_labels.ndim != 3 or audio_labels.shape[0] != thinker_attention_mask.shape[0] or audio_labels.shape[2] != thinker_attention_mask.shape[1]:
        raise ValueError(f'audio_labels must be (B,8,T) aligned with thinker mask, got {tuple(audio_labels.shape)}')
    audio_positions = (audio_labels != -100).any(dim=1)
    return (thinker_attention_mask.bool() | audio_positions).to(dtype=thinker_attention_mask.dtype)


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


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None):
    start_time = time.time()
    last_step = start_step
    last_grad_norm = 0.0
    if 'cuda' in args.device:
        torch.cuda.reset_peak_memory_stats()
    for step, (input_ids, attention_mask, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        talker_attention_mask = build_talker_attention_mask(attention_mask, audio_labels)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if hasattr(pixel_values, 'keys'):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        last_step = step
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch * iters + step, args.epochs * iters, param_group['base_lr'])

        with autocast_ctx:
            text_logits_mask = labels.ne(-100)
            res = model(
                input_ids,
                attention_mask=attention_mask,
                talker_attention_mask=talker_attention_mask,
                use_cache=False,
                audio_inputs=audio_inputs,
                audio_lens=audio_lens,
                pixel_values=pixel_values,
                spk_emb=spk_emb,
                # Text/image-only batches have no audio targets. Skipping the
                # Talker here makes I2T joint tuning faster without changing
                # its loss or gradients. A2A projector tuning still runs the
                # frozen Talker because audio loss trains audio_proj through it.
                output_audio_logits=bool(audio_labels.ne(-100).any().item()),
                audio_targets=audio_labels,
                text_logits_mask=text_logits_mask,
            )
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            
            # Text loss
            text_targets = labels[text_logits_mask]
            text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), text_targets.reshape(-1))
            text_loss = text_loss_raw.sum() / (text_targets.numel() + 1e-9)
            
            # Audio loss
            audio_loss, main_codec_loss, residual_codec_loss, residual_losses = compute_audio_losses(
                res, audio_labels, omni_config
            )
            
            loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            last_grad_norm = float(grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            text_loss_val = text_loss.item() if isinstance(text_loss, torch.Tensor) else 0
            audio_loss_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0
            main_codec_loss_val = main_codec_loss.item() if isinstance(main_codec_loss, torch.Tensor) else 0
            residual_codec_loss_val = residual_codec_loss.item() if isinstance(residual_codec_loss, torch.Tensor) else 0
            qwen_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'qwen'), 0.0)
            omni_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'omni'), 0.0)
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if 'cuda' in args.device else 0.0
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            text_scale = float(raw_model.talker.text_scale.detach())
            audio_scale = float(raw_model.talker.audio_scale.detach())
            seconds_per_step = spend_time / max(step - start_step, 1)
            eta_min = seconds_per_step * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss:{current_loss:.4f} text:{text_loss_val:.4f} audio:{audio_loss_val:.4f} main:{main_codec_loss_val:.4f} residual:{residual_codec_loss_val:.4f} qlr:{qwen_lr:.2e} olr:{omni_lr:.2e} grad:{last_grad_norm:.2f} scale:{text_scale:.3f}/{audio_scale:.3f} seq:{input_ids.size(-1)} step:{seconds_per_step:.3f}s peak:{peak_mem_gb:.1f}GB eta:{eta_min:.0f}min')
            if swanlab:
                metrics = {"loss": current_loss, "text_loss": text_loss_val,
                          "audio_loss": audio_loss_val, "qwen_lr": qwen_lr, "omni_lr": omni_lr,
                          "main_codec_loss": main_codec_loss_val,
                          "residual_codec_loss": residual_codec_loss_val,
                          "grad_norm": last_grad_norm, "text_scale": text_scale,
                          "audio_scale": audio_scale, "peak_memory_gb": peak_mem_gb,
                          "sequence_length": input_ids.size(-1), "seconds_per_step": seconds_per_step,
                          "epoch_time": eta_min}
                if omni_config.audio_head_type == 'main_codec_predictor':
                    metrics.update({
                        f"residual_codec_loss_c{index + 1}": float(layer_loss.detach())
                        for index, layer_loss in enumerate(residual_losses)
                    })
                swanlab.log(metrics)

        interval_save = args.save_interval > 0 and step % args.save_interval == 0
        epoch_complete = step == iters and bool(args.save_at_epoch_end)
        if interval_save or epoch_complete:
            # All ranks wait while rank 0 atomically replaces the checkpoint;
            # otherwise the next epoch can enter DDP collectives during saving.
            if dist.is_initialized():
                dist.barrier()
            if is_main_process():
                model.eval()
                omni_checkpoint(
                    omni_config,
                    weight=args.save_weight,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1 if epoch_complete else epoch,
                    step=0 if epoch_complete else step,
                    swanlab=swanlab,
                    save_dir=args.save_dir,
                    scaler=scaler,
                    tokenizer=tokenizer,
                    save_optimizer_state=bool(args.save_optimizer_state),
                    stage_id=args.stage_id,
                    train_modules=args.train_modules or 'legacy',
                    training_format_version=TRAINING_FORMAT_VERSION,
                )
                model.train()
            if dist.is_initialized():
                dist.barrier()

        del input_ids, attention_mask, talker_attention_mask, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, text_logits_mask, text_targets, res, loss
        if args.max_steps > 0 and step >= args.max_steps:
            break

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniQwen-Omni SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_omni', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--qwen_learning_rate", type=float, default=1e-5, help="Qwen Thinker初始学习率")
    parser.add_argument("--omni_learning_rate", type=float, default=5e-4, help="Talker/projector初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=0, help="epoch内额外保存间隔；无论是否为0，每个epoch结束都会保存")
    parser.add_argument("--save_at_epoch_end", default=1, type=int, choices=[0, 1], help="epoch结束是否保存；仅性能测试时设0")
    parser.add_argument("--save_optimizer_state", type=int, default=0, choices=[0, 1], help="是否保存optimizer状态；断点续训必须设为1")
    parser.add_argument("--stage_id", type=int, default=0, help="训练流水线阶段编号，用于自动识别可续训checkpoint")
    parser.add_argument("--max_steps", type=int, default=0, help="每个epoch最多训练步数；0表示完整epoch")
    parser.add_argument('--num_talker_hidden_layers', default=6, type=int, choices=[4, 6, 8], help="Talker层数")
    parser.add_argument('--talker_hidden_size', default=768, type=int, help="Talker隐藏层维度")
    parser.add_argument('--accept_hidden_layer', default=14, type=int, help="接入的Qwen decoder block层号（1-based）")
    parser.add_argument('--audio_head_type', default='main_codec_predictor', type=str,
                        choices=['main_codec_predictor', 'legacy_parallel_delay'], help='Talker音频预测头')
    parser.add_argument('--code_predictor_num_layers', default=2, type=int, help='帧内Code Predictor层数')
    parser.add_argument('--code_predictor_hidden_size', default=768, type=int, help='帧内Code Predictor隐藏维度')
    parser.add_argument('--residual_codec_loss_weight', default=0.3, type=float, help='Residual codec平均loss权重')
    parser.add_argument('--use_mrope', default=1, type=int, choices=[0, 1], help="Thinker是否使用三轴MRoPE")
    parser.add_argument('--use_modality_boundaries', default=1, type=int, choices=[0, 1], help="是否为Thinker音频/图像加入边界token")
    parser.add_argument('--use_talker_ref_boundaries', default=1, type=int, choices=[0, 1], help="是否用REF_START/REF_END包围Talker参考音频")
    parser.add_argument('--max_images', default=4, type=int, help="每条训练样本最多使用的图片数")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/train_t2a_mini.parquet", help="训练数据路径（parquet格式）")
    parser.add_argument("--audio_encoder_dir", type=str, default="../model/SenseVoiceSmall", help="音频encoder路径(SenseVoice)")
    parser.add_argument("--vision_dir", type=str, default="../model/siglip2-base-p32-256-ve", help="CLIP视觉模型路径")
    parser.add_argument('--model_path', default='../model/Qwen3-0.6B', type=str, help="Qwen3或MiniQwen-Omni HF模型目录")
    parser.add_argument('--from_weight', default='qwen', type=str, help="qwen表示从model_path初始化，也可传HF checkpoint目录")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_backbone', default='none', type=str, choices=['none', 'all', 'last1'], help="冻结主干模型: none=全量训练, all=只训练audio层, last1=只训练最后1层+audio层")
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'audio_proj', 'vision_proj'], help="训练模式: all=全量训练, audio_proj=只训练audio_proj, vision_proj=只训练vision_proj")
    parser.add_argument(
        '--train_modules', default='', type=str,
        help='显式解冻模块（逗号分隔）：all/thinker/text_head/talker/talker_core/speaker/audio_proj/vision_proj；非空时覆盖mode/freeze的冻结结果',
    )
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="MiniQwen-Omni-SFT", help="SwanLab项目名")
    parser.add_argument("--swanlab_run_name", type=str, default="", help="SwanLab run名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--gradient_checkpointing", default=0, type=int, choices=[0, 1], help="Qwen梯度检查点")
    parser.add_argument("--dynamic_padding", default=1, type=int, choices=[0, 1], help="按batch裁掉全padding尾部（不删除有效token）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    omni_config = OmniConfig.from_qwen_pretrained(
        args.model_path,
        num_talker_hidden_layers=args.num_talker_hidden_layers,
        talker_hidden_size=args.talker_hidden_size,
        accept_hidden_layer=args.accept_hidden_layer,
        audio_head_type=args.audio_head_type,
        code_predictor_num_layers=args.code_predictor_num_layers,
        code_predictor_hidden_size=args.code_predictor_hidden_size,
        residual_codec_loss_weight=args.residual_codec_loss_weight,
        use_moe=bool(args.use_moe),
        use_mrope=bool(args.use_mrope),
        use_modality_boundaries=bool(args.use_modality_boundaries),
        use_talker_ref_boundaries=bool(args.use_talker_ref_boundaries),
        max_images=args.max_images,
    )
    ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume==1 else None
    if ckp_data and ckp_data.get('training_format_version') != TRAINING_FORMAT_VERSION:
        Logger('忽略旧版checkpoint：其token/MRoPE/Talker参考音频格式与当前训练格式不兼容，将从--from_weight重新开始')
        ckp_data = None
    if ckp_data and ckp_data.get('stage_id') != args.stage_id:
        Logger(f'checkpoint属于stage {ckp_data.get("stage_id", "unknown")}，当前为stage {args.stage_id}；仅加载模型权重并启动新阶段')
        ckp_data = None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配swanlab ==========
    swanlab = None
    if args.use_swanlab and is_main_process():
        swanlab = init_swanlab_tracking(args, ckp_data)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    load_weight = ckp_data['checkpoint_dir'] if ckp_data else args.from_weight
    model, tokenizer = init_omni_model(omni_config, from_weight=load_weight, tokenizer_path=args.model_path,
                                        audio_encoder_path=args.audio_encoder_dir,
                                        vision_model_path=args.vision_dir,
                                        save_dir=args.save_dir, device=args.device,
                                        freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)
    
    if args.gradient_checkpointing == 1:
        model.gradient_checkpointing_enable()
    if args.use_compile == 1:
        model = torch.compile(model)
    
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    
    if args.mode == 'audio_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.audio_proj.parameters(): p.requires_grad = True
    elif args.mode == 'vision_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.vision_proj.parameters(): p.requires_grad = True
    selected_modules = configure_trainable_modules(model, args.train_modules)
    trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    if trainable_dtypes != {torch.float32}:
        raise RuntimeError(f'训练参数必须为FP32并通过BF16 autocast计算，当前dtype={trainable_dtypes}')
    log_model_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    module_policy = ','.join(selected_modules) if selected_modules else f'legacy:{args.mode}/{args.freeze_backbone}'
    Logger(f'Trainable: {trainable:.2f}M | Params: FP32 | Autocast: {args.dtype} | Modules: {module_policy} | Compile: {"on" if args.use_compile else "off"}')
    
    # scheduled_sampling 现在会自动保护 image/audio token 的连续性
    train_ds = OmniDataset(
        args.data_path, 
        tokenizer, 
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
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
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    qwen_params, omni_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name.removeprefix('_orig_mod.')
        (qwen_params if clean_name.startswith('model.') or clean_name.startswith('lm_head.') else omni_params).append(param)
    param_groups = []
    if qwen_params: param_groups.append({'params': qwen_params, 'lr': args.qwen_learning_rate, 'base_lr': args.qwen_learning_rate, 'name': 'qwen'})
    if omni_params: param_groups.append({'params': omni_params, 'lr': args.omni_learning_rate, 'base_lr': args.omni_learning_rate, 'name': 'omni'})
    optimizer = optim.AdamW(param_groups)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        if 'optimizer' not in ckp_data:
            checkpoint_complete = ckp_data.get('step', 0) == 0 and ckp_data.get('epoch', 0) >= args.epochs
            if not checkpoint_complete:
                raise ValueError('checkpoint没有optimizer状态，不能续训未完成的stage；请通过--from_weight开始新训练阶段')
            Logger(f'stage {args.stage_id} 已训练完成；checkpoint不含optimizer，跳过训练并继续流水线')
        else:
            optimizer.load_state_dict(ckp_data['optimizer'])
            scaler.load_state_dict(ckp_data['scaler'])
            rng_state = ckp_data.get('rng_state')
            if rng_state:
                random.setstate(rng_state['python'])
                np.random.set_state(rng_state['numpy'])
                torch.set_rng_state(rng_state['torch'])
                if rng_state.get('cuda') is not None: torch.cuda.set_rng_state_all(rng_state['cuda'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'断点续训: stage {args.stage_id}, epoch {start_epoch + 1}/{args.epochs}, step {start_step}')
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        collate_fn = partial(omni_collate_fn, dynamic_padding=bool(args.dynamic_padding), pad_to_multiple=8)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
        epoch_iters = min(len(loader) + skip, args.max_steps) if args.max_steps > 0 else len(loader) + skip
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, epoch_iters, start_step, swanlab)
        else:
            train_epoch(epoch, loader, epoch_iters, 0, swanlab)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
