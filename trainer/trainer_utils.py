"""
训练工具函数集合
"""
import os
import json
import glob
import tempfile
import shutil
import random
import math
import numpy as np
import datasets
import torch
import torch.distributed as dist
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch.utils.data import Sampler
from transformers import AutoTokenizer, logging as hf_logging
from model.model_omni import MiniQwenOmni


def get_omni_model_class(model_arch='production'):
    """Resolve the supported production architecture implementation."""
    if model_arch == 'production':
        return MiniQwenOmni
    raise ValueError(f'unsupported model_arch: {model_arch}')


def infer_omni_model_arch(checkpoint_dir):
    """Select the implementation from a saved config without guessing."""
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if not os.path.isfile(config_path):
        return 'production'
    with open(config_path, encoding='utf-8') as handle:
        saved_config = json.load(handle)
    if saved_config.get('codec_feedback_type') == 'qwen_hidden_rmsnorm':
        raise ValueError(
            'This checkpoint uses the retired qwen_hidden_rmsnorm feedback experiment; '
            'use the archived experiment code only for historical analysis.'
        )
    return 'production'
    


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    # 与 mmv 保持一致：初始 lr=1.0*base_lr，最终 lr=0.1*base_lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # 明确绑定 rank 和 PPU，避免 ProcessGroup 猜测设备并输出警告。
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
    )
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_model_params(model, ignore_patterns=['audio_encoder', 'vision_encoder']):
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    cfg = model.config
    n_routed = getattr(cfg, 'n_routed_experts', getattr(cfg, 'num_experts', 0))
    n_active = getattr(cfg, 'num_experts_per_tok', 0)
    n_shared = getattr(cfg, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def load_omni_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, fix_mistral_regex=True)
    # Never replace Qwen's existing additional_special_tokens. Replacing the
    # list makes vision/chat controls stop participating in skip-special-token
    # handling after the tokenizer is saved.
    required_special_tokens = [
        '<|im_start|>', '<|im_end|>',
        '<|object_ref_start|>', '<|object_ref_end|>',
        '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>',
        '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>',
        '<|image_pad|>', '<|video_pad|>',
        # Keep audio_pad first so existing MiniQwen checkpoints retain ID
        # 151669; the two new boundaries consume subsequent reserved rows.
        '<|audio_pad|>', '<|audio_start|>', '<|audio_end|>',
    ]
    tokenizer.add_special_tokens(
        {'additional_special_tokens': required_special_tokens},
        replace_additional_special_tokens=False,
    )
    audio_ids = tokenizer.encode('<|audio_pad|>', add_special_tokens=False)
    image_ids = tokenizer.encode('<|image_pad|>', add_special_tokens=False)
    for token in required_special_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != tokenizer.convert_tokens_to_ids(token):
            raise ValueError(f'Qwen special token must encode to one token: {token} -> {ids}')
    return tokenizer


def configure_token_ids(omni_config, tokenizer):
    if len(tokenizer) > omni_config.vocab_size:
        raise ValueError(f'tokenizer size {len(tokenizer)} exceeds model vocab {omni_config.vocab_size}')
    omni_config.audio_ids = tokenizer.encode(omni_config.audio_special_token, add_special_tokens=False)
    omni_config.audio_start_ids = tokenizer.encode(omni_config.audio_start_special_token, add_special_tokens=False)
    omni_config.audio_end_ids = tokenizer.encode(omni_config.audio_end_special_token, add_special_tokens=False)
    omni_config.image_ids = tokenizer.encode(omni_config.image_special_token, add_special_tokens=False)
    omni_config.vision_start_ids = tokenizer.encode(omni_config.vision_start_special_token, add_special_tokens=False)
    omni_config.vision_end_ids = tokenizer.encode(omni_config.vision_end_special_token, add_special_tokens=False)
    omni_config.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
    omni_config.eos_token_id = tokenizer.eos_token_id
    omni_config.pad_token_id = tokenizer.pad_token_id

    configured = {
        'audio': omni_config.audio_ids,
        'audio_start': omni_config.audio_start_ids,
        'audio_end': omni_config.audio_end_ids,
        'image': omni_config.image_ids,
        'vision_start': omni_config.vision_start_ids,
        'vision_end': omni_config.vision_end_ids,
    }
    invalid = {name: ids for name, ids in configured.items() if len(ids) != 1}
    if invalid:
        raise ValueError(f'modality control tokens must each encode to one token: {invalid}')


def format_audio_prompt(config, length):
    payload = config.audio_special_token * int(length)
    if getattr(config, 'use_modality_boundaries', False):
        return config.audio_start_special_token + payload + config.audio_end_special_token
    return payload


def format_image_prompt(config, count=1):
    payload = config.image_special_token * config.image_token_len
    if getattr(config, 'use_modality_boundaries', False):
        payload = config.vision_start_special_token + payload + config.vision_end_special_token
    return payload * int(count)


def _attach_external_encoder(model, name, encoder, trainable=False):
    """Attach an auxiliary encoder with optional nn.Module registration.

    Frozen encoders retain the historical unregistered attachment, keeping
    production checkpoints unchanged. A trainable encoder must be registered
    so DDP, the optimizer and gradient clipping can see its parameters.
    """
    if name in model.__dict__:
        object.__delattr__(model, name)
    elif name in model._modules:
        delattr(model, name)
    if encoder is None:
        object.__setattr__(model, name, None)
        object.__setattr__(model, f'_train_{name}', False)
        return
    if trainable:
        model.add_module(name, encoder)
    else:
        object.__setattr__(model, name, encoder)
    for parameter in encoder.parameters():
        parameter.requires_grad = bool(trainable)
    encoder.train(bool(trainable))
    object.__setattr__(model, f'_train_{name}', bool(trainable))


def load_external_encoder_sidecars(model, checkpoint_dir):
    """Restore optional trained auxiliary encoders from a MiniQwen checkpoint."""
    restored = []
    for name, label in (('audio_encoder', '音频'), ('vision_encoder', '视觉')):
        path = os.path.join(checkpoint_dir, f'{name}.safetensors')
        encoder = getattr(model, name, None)
        if encoder is not None and os.path.isfile(path):
            load_safetensors_model(encoder, path, strict=True)
            object.__setattr__(model, f'_persist_{name}', True)
            Logger(f'已加载{label}encoder参数: {path}')
            restored.append(name)
    return restored


def init_omni_model(omni_config, from_weight='qwen', tokenizer_path='../model/Qwen3-0.6B', audio_encoder_path='../model/SenseVoiceSmall', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_backbone='none', from_resume=0, train_audio_encoder=False, train_vision_encoder=False, model_arch='production'):
    hf_logging.set_verbosity_error()
    model_class = get_omni_model_class(model_arch)
    tokenizer = load_omni_tokenizer(tokenizer_path)
    configure_token_ids(omni_config, tokenizer)
    load_path = from_weight if os.path.isdir(from_weight) else tokenizer_path
    saved_config_path = os.path.join(load_path, 'config.json')
    if os.path.isfile(saved_config_path):
        with open(saved_config_path, encoding='utf-8') as handle:
            saved_config = json.load(handle)
        if saved_config.get('model_type') == 'miniqwen-omni':
            saved_head = saved_config.get('audio_head_type', 'legacy_parallel_delay')
            if saved_head != omni_config.audio_head_type:
                raise ValueError(
                    f'checkpoint audio_head_type={saved_head} is incompatible with '
                    f'configured {omni_config.audio_head_type}; start the new head from Qwen3 '
                    f'or select the matching legacy branch explicitly'
                )
            saved_feedback = saved_config.get('codec_feedback_type', 'mean_embedding')
            configured_feedback = getattr(omni_config, 'codec_feedback_type', 'mean_embedding')
            if saved_feedback != configured_feedback:
                raise ValueError(
                    f'checkpoint codec_feedback_type={saved_feedback} is incompatible with '
                    f'configured {configured_feedback}; use a separate checkpoint '
                    f'for the Qwen feedback experiment'
                )
    model = model_class.from_pretrained(
        load_path,
        config=omni_config,
        # Keep trainable parameters and AdamW moments in FP32.  BF16 is used
        # only by autocast during forward/backward; otherwise a 1e-5 Qwen LR is
        # commonly rounded away when written directly into BF16 parameters.
        dtype=torch.float32,
        audio_encoder_path=None,
        vision_model_path=None,
    )
    audio_encoder, audio_processor = model_class.load_sensevoice(audio_encoder_path)
    vision_encoder, vision_processor = model_class.load_vision(vision_model_path)
    _attach_external_encoder(model, 'audio_encoder', audio_encoder, train_audio_encoder)
    object.__setattr__(model, 'audio_processor', audio_processor)
    _attach_external_encoder(model, 'vision_encoder', vision_encoder, train_vision_encoder)
    object.__setattr__(model, 'vision_processor', vision_processor)
    object.__setattr__(model, '_persist_audio_encoder', bool(train_audio_encoder))
    object.__setattr__(model, '_persist_vision_encoder', bool(train_vision_encoder))
    load_external_encoder_sidecars(model, load_path)
    Logger(f'已加载模型: {load_path}')
    
    # 冻结策略
    if freeze_backbone == 'all':
        # 冻结整个主干模型
        for param in model.model.parameters():
            param.requires_grad = False
    elif freeze_backbone == 'last1':
        # 冻结除了最后1层之外的所有层
        for param in model.model.parameters():
            param.requires_grad = False
        # 打开最后1层
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True
    return model.to(device), tokenizer


def omni_checkpoint(omni_config, weight='miniqwen_omni', model=None, optimizer=None, epoch=0, step=0, swanlab=None, save_dir='../checkpoints', tokenizer=None, save_optimizer_state=True, **kwargs):
    root = os.path.join(save_dir, weight)
    os.makedirs(root, exist_ok=True)
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        checkpoint_dir = os.path.join(root, 'checkpoint')
        tmp_dir = tempfile.mkdtemp(prefix='.checkpoint-', dir=root)
        # Save the FP32 master parameters.  Casting them back to BF16 here
        # would discard low-LR Qwen/Norm updates and make resume non-equivalent
        # even though forward/backward correctly used BF16 autocast.
        # Registered trainable encoders are external dependencies and use
        # dedicated sidecars. Keep the core HF state compatible with ordinary
        # MiniQwen checkpoints by excluding those registered prefixes.
        core_state = {
            name: value for name, value in raw_model.state_dict().items()
            if not name.startswith(('audio_encoder.', 'vision_encoder.'))
        }
        raw_model.save_pretrained(tmp_dir, state_dict=core_state, safe_serialization=True)
        encoder_sidecars = []
        if getattr(raw_model, '_persist_audio_encoder', False):
            save_safetensors_model(
                raw_model.audio_encoder,
                os.path.join(tmp_dir, 'audio_encoder.safetensors'),
            )
            encoder_sidecars.append('audio_encoder.safetensors')
        if getattr(raw_model, '_persist_vision_encoder', False):
            save_safetensors_model(
                raw_model.vision_encoder,
                os.path.join(tmp_dir, 'vision_encoder.safetensors'),
            )
            encoder_sidecars.append('vision_encoder.safetensors')
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_dir)
        swanlab_id = None
        if swanlab:
            if hasattr(swanlab, 'get_run'):
                run = swanlab.get_run()
                swanlab_id = getattr(run, 'id', None) if run else None
            else:
                swanlab_id = getattr(swanlab, 'id', None)
        
        resume_data = {
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'swanlab_id': swanlab_id,
            'external_encoder_sidecars': encoder_sidecars,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        if save_optimizer_state and optimizer is not None:
            resume_data['optimizer'] = optimizer.state_dict()
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        torch.save(resume_data, os.path.join(tmp_dir, 'trainer_state.pt'))
        previous_dir = os.path.join(root, '.previous-checkpoint')
        if os.path.exists(previous_dir):
            shutil.rmtree(previous_dir)
        if os.path.exists(checkpoint_dir):
            os.replace(checkpoint_dir, previous_dir)
        os.replace(tmp_dir, checkpoint_dir)
        if os.path.exists(previous_dir):
            shutil.rmtree(previous_dir)
        Logger(f'已保存 HF checkpoint: {checkpoint_dir}')
        return checkpoint_dir
    else:  # 加载模式
        stable_checkpoint = os.path.join(root, 'checkpoint')
        checkpoints = [stable_checkpoint] if os.path.isdir(stable_checkpoint) else sorted(glob.glob(os.path.join(root, 'checkpoint-*')))
        if checkpoints:
            checkpoint_dir = checkpoints[-1]
            ckp_data = torch.load(
                os.path.join(checkpoint_dir, 'trainer_state.pt'),
                map_location='cpu',
                weights_only=False,
            )
            ckp_data['checkpoint_dir'] = checkpoint_dir
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
