#!/usr/bin/env python3
"""Train prepared mixed SFT pools with token-weighted updates and resumable state."""
import argparse
from collections import deque
from contextlib import nullcontext
from functools import partial
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset.sft_dataset import SFTDataset, load_manifest_paths
from model.model_omni import OmniConfig
from trainer.metrics import empty_metrics, finalise, update_metrics
from trainer.pretrain_checkpoint import recover_checkpoint
from trainer.pretrain_sampler import EpochBatchSampler
from trainer.train_pretrain import (
    atomic_json, forward, move, objective, reduce_validation_sums,
    restore_rng, rng_state, training_metrics,
)
from trainer.training_common import configure_trainable_modules, omni_collate_fn
from trainer.trainer_utils import (
    init_distributed_mode, init_omni_model, log_model_params, omni_checkpoint, setup_seed,
)


def parameter_groups(model, plan):
    """Partition tied weights once and give the speaker projection its own rate."""
    grouped = {'thinker': [], 'omni': [], 'speaker': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(('model.', 'lm_head.')):
            group = 'thinker'
        elif name.startswith('talker.spk_proj.'):
            group = 'speaker'
        else:
            group = 'omni'
        grouped[group].append(parameter)
    rates = plan['learning_rates']
    return [{'name': name, 'params': values, 'lr': rates[name], 'base_lr': rates[name]}
            for name, values in grouped.items() if values]


def window_token_counts(batches, device):
    counts = [0, 0, 0]
    for batch in batches:
        counts[0] += int(batch[2].ne(-100).sum())
        counts[1] += int(batch[3][:, 0].ne(-100).sum())
        counts[2] += int(batch[3][:, 1].ne(-100).sum())
    tensor = torch.tensor(counts, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def lr_factor(step, total_steps, warmup_ratio):
    warmup = max(1, math.ceil(total_steps * warmup_ratio))
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1)))


def make_dataset(paths, tokenizer, model, capacity):
    return SFTDataset(
        ','.join(str(ROOT / path) for path in paths), tokenizer,
        audio_processor=model.audio_processor, vision_processor=model.vision_processor,
        max_length=capacity, image_token_len=model.config.image_token_len,
        max_images=model.config.max_images, training=False,
    )


def dev_datasets(manifest, tokenizer, model, capacity, samples, seed):
    sources = {name: json.loads((ROOT / value['manifest']).read_text())
               for name, value in manifest['sources'].items()}
    result = {}
    for name, source in sources.items():
        for key, paths in source['data_files'].items():
            tier, split = key.split('/')
            if split != 'dev':
                continue
            label = name + '/' + tier
            dataset = make_dataset(paths, tokenizer, model, capacity)
            derived_seed = seed + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
            indices = random.Random(derived_seed).sample(range(len(dataset)), min(samples, len(dataset)))
            result[label] = (dataset, indices)
    return result


def evaluate(model, datasets, device, rank, world, batch_size, collate):
    model.eval()
    metrics = {}
    totals = empty_metrics()
    with torch.inference_mode():
        for name, (dataset, indices) in datasets.items():
            sums = empty_metrics()
            selected = indices[rank::world]
            loader = DataLoader(Subset(dataset, selected), batch_size=batch_size,
                                collate_fn=collate, num_workers=0)
            for batch in loader:
                batch = move(batch, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = forward(model, batch, 'tts')
                update_metrics(sums, result, batch[2], batch[3], model.config.audio_stop_token)
            sums = reduce_validation_sums(sums, device)
            metrics[name] = finalise(sums, model.config.residual_codec_loss_weight)
            for key, value in sums.items():
                if isinstance(value, list):
                    totals[key] = [a + b for a, b in zip(totals[key], value)]
                else:
                    totals[key] += value
    metrics['all'] = finalise(totals, model.config.residual_codec_loss_weight)
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--checkpoint', default='',
                        help='Optional downloaded V1-Base checkpoint under model/')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--no-tracking', action='store_true')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--skip-initial-eval', action='store_true')
    parser.add_argument('--benchmark-steps', type=int, default=0)
    parser.add_argument('--benchmark-warmup', type=int, default=2)
    parser.add_argument('--benchmark-from-qwen', action='store_true')
    args = parser.parse_args()
    benchmarking = args.benchmark_steps > 0
    if args.benchmark_from_qwen and not benchmarking:
        parser.error('--benchmark-from-qwen requires a bounded benchmark')
    plan = json.loads(Path(args.plan).read_text())
    if benchmarking:
        target = (ROOT / (args.output or plan['output'])).resolve()
        if not target.is_relative_to(ROOT / '.runtime') or args.resume:
            parser.error('Benchmarks require fresh .runtime outputs and no resume')
        if (target / 'benchmark.json').exists():
            parser.error('Benchmark output already exists')
        if not 0 <= args.benchmark_warmup < args.benchmark_steps <= 100:
            parser.error('Require 0 <= warmup < benchmark steps <= 100')
        args.max_steps = args.benchmark_steps
        args.no_tracking = args.no_checkpoint = args.skip_initial_eval = True
    manifest = json.loads((ROOT / plan['manifest']).read_text())
    if manifest['status'] != 'READY':
        raise ValueError('SFT manifest must reference accepted prepared data')
    local = init_distributed_mode()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world != plan['world_size']:
        raise ValueError('World size differs from the training plan')
    device = torch.device('cuda', local)
    setup_seed(plan['seed'] + rank)
    os.environ['MINIQWEN_PARQUET_MEMORY_CACHE_MIB'] = str(plan['loader']['cache_mib'])
    output = ROOT / (args.output or plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    signature_data = {'plan': plan, 'manifest': manifest}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    recovered = recover_checkpoint(output / 'last')
    if recovered and not args.resume:
        raise ValueError('Existing SFT checkpoint requires --resume or a fresh output')
    saved = None
    if args.resume:
        if recovered is None:
            raise ValueError('No committed SFT checkpoint to resume')
        saved = torch.load(recovered / 'trainer_state.pt', map_location='cpu', weights_only=False)
        optimizer_file = saved.get('optimizer_file')
        if optimizer_file and 'optimizer' not in saved:
            optimizer_path = recovered / optimizer_file
            if not optimizer_path.is_file():
                raise ValueError(f'Missing optimizer checkpoint: {optimizer_path}')
            saved['optimizer'] = torch.load(optimizer_path, map_location='cpu', weights_only=False)
        if saved['sft_signature'] != signature:
            raise ValueError('Training plan differs from the checkpoint plan')
        if 'optimizer' not in saved and not saved.get('phase_complete'):
            raise ValueError('Optimizer state is required to resume an unfinished run')
        if saved.get('phase_complete'):
            if rank == 0:
                print('SFT_ALREADY_COMPLETE', flush=True)
            return
    checkpoint = recovered if saved else Path(args.checkpoint or plan['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    config = (OmniConfig.from_qwen_pretrained(
        str(ROOT / 'model/Qwen3-0.6B'), talker_condition_prefix_length=160,
        use_mrope=True, use_modality_boundaries=True, use_talker_ref_boundaries=True,
    ) if args.benchmark_from_qwen else OmniConfig.from_pretrained(checkpoint, local_files_only=True))
    model, tokenizer = init_omni_model(
        config, from_weight='qwen' if args.benchmark_from_qwen else str(checkpoint), tokenizer_path=str(ROOT / 'model/Qwen3-0.6B'),
        audio_encoder_path=str(ROOT / 'model/SenseVoiceSmall'),
        vision_model_path=str(ROOT / 'model/siglip2-base-p32-256-ve'), device=str(device),
    )
    model.audio_encoder.to(device)
    model.vision_encoder.to(device)
    configure_trainable_modules(model, 'thinker,text_head,talker,audio_proj,vision_proj')
    if plan['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    assert all(p.requires_grad for p in model.model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in model.audio_encoder.parameters())
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    optimizer = torch.optim.AdamW(parameter_groups(model, plan), weight_decay=plan['weight_decay'])
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng_state_by_rank'][rank], device)
    if rank == 0:
        log_model_params(model)
        atomic_json(output / 'parameter-groups.json', {
            group['name']: {'parameters': sum(p.numel() for p in group['params']),
                            'learning_rate': group['base_lr']}
            for group in optimizer.param_groups
        })
    paths, capacity = load_manifest_paths(ROOT / plan['manifest'])
    train = SFTDataset(paths, tokenizer, model.audio_processor, model.vision_processor,
                       max_length=capacity, training=True, max_images=4)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                 seed=plan['seed'], drop_last=False)
    batch_size = plan['batch_size']
    accumulation = plan['accumulation_steps']
    micro_total = math.ceil(len(sampler) / batch_size) * plan['epochs']
    total_steps = math.ceil(micro_total / accumulation)
    micro = saved.get('micro_step', 0) if saved else 0
    step = saved.get('global_step', 0) if saved else 0
    loss_history = deque(saved.get('loss_history', []) if saved else [], maxlen=100)
    collate = partial(omni_collate_fn, dynamic_padding=True, pad_to_multiple=8)
    dev = {} if benchmarking else dev_datasets(manifest, tokenizer, model, capacity,
                                              plan['dev_samples_per_source_tier'], plan['seed'])
    batch_sampler = EpochBatchSampler(sampler, batch_size, plan['epochs'], skip_batches=micro,condition_epochs=True)
    loader_options = dict(num_workers=plan['loader']['workers'], pin_memory=True,
                          collate_fn=collate, batch_sampler=batch_sampler,
                          generator=torch.Generator().manual_seed(plan['seed'] + rank))
    if plan['loader']['workers']:
        # Keep workers non-persistent so their Parquet/media caches are
        # released before checkpoint serialization. Persistent workers kept a
        # second copy of cached rows alive and amplified host-memory peaks.
        loader_options.update(prefetch_factor=plan['loader']['prefetch'], persistent_workers=False)
    loader = DataLoader(train, **loader_options)
    raw = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)
    swanlab = None
    if rank == 0 and not args.no_tracking:
        import swanlab
        options = dict(project=plan['swanlab_project'], workspace=plan.get('swanlab_workspace'),
                       name=plan['swanlab_name'], config=plan,
                       mode='cloud', logdir=str(output / 'swanlog'))
        if saved and saved.get('swanlab_id'):
            options.update(id=saved['swanlab_id'], resume='allow')
        swanlab.init(**options)
        run = swanlab.get_run()
        run_info = {key: str(getattr(run, key, '')) for key in ('id', 'url', 'name')}
        atomic_json(output / 'swanlab.json', run_info)
    if rank == 0:
        print('SFT_START', 'rows', len(train), 'global_batch', batch_size * world * accumulation,
              'steps', total_steps, 'capacity', capacity, flush=True)
        atomic_json(output / 'run-plan.json', plan)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def validation():
        before = rng_state(device)
        metrics = evaluate(raw, dev, device, rank, world, plan['eval_batch_size'], collate)
        restore_rng(before, device)
        if rank == 0:
            values = {'dev/' + source + '/' + key: value
                      for source, results in metrics.items() for key, value in results.items()
                      if isinstance(value, (int, float))}
            atomic_json(output / 'validation-latest.json', {'step': step, 'metrics': metrics})
            with (output / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': step, 'metrics': metrics}) + '\n')
            if swanlab:
                swanlab.log(values, step=step)
            print('SFT_DEV', step, metrics['all']['text_loss'],
                  metrics['all'].get('audio_loss', 0), flush=True)

    def save_checkpoint(complete):
        states = [None] * world
        local_rng = rng_state(device)
        if world > 1:
            dist.all_gather_object(states, local_rng)
        else:
            states = [local_rng]
        if rank == 0:
            omni_checkpoint(
                config, weight='last', save_dir=str(output), model=raw, optimizer=optimizer,
                tokenizer=tokenizer, swanlab=swanlab, save_optimizer_state=True,
                epoch=micro // math.ceil(len(sampler) / batch_size), step=micro,
                micro_step=micro, global_step=step, phase_complete=complete,
                rng_state_by_rank=states, sft_signature=signature,
                loss_history=list(loss_history), training_format_version=5,
            )
        if world > 1:
            dist.barrier()

    if not saved and not args.skip_initial_eval:
        validation()
    iterator = iter(loader)
    last_saved = step if saved else None
    benchmark_records = []
    if benchmarking:
        torch.cuda.reset_peak_memory_stats(device)
    while micro < micro_total:
        if benchmarking:
            dist.barrier() if world > 1 else None
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        batches = list(islice(iterator, accumulation))
        if not batches:
            raise RuntimeError('Training loader ended before its recorded cursor')
        data_wait = time.perf_counter() - started
        counts = window_token_counts(batches, device)
        factor = lr_factor(step + 1, total_steps, plan['warmup_ratio'])
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        stats = torch.zeros(9, dtype=torch.float64, device=device)
        lengths = []
        for index, batch in enumerate(batches):
            synchronize = index == len(batches) - 1
            context = model.no_sync() if world > 1 and not synchronize else nullcontext()
            batch = move(batch, device)
            lengths.append(batch[0].shape[-1])
            with context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = forward(model, batch, 'tts')
                    loss, batch_stats = objective(
                        result, batch[2], batch[3], config, world, return_logging=True,
                        normalization_counts=counts,
                    )
                    loss = loss - result.aux_loss + result.aux_loss / len(batches)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite SFT loss')
                loss.backward()
            stats += batch_stats
            micro += 1
            if step == 0 and rank == 0:
                print('SFT_MICRO', index + 1, '/', len(batches), flush=True)
            del result, loss, batch, batch_stats
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad], plan['grad_clip'])
        if not torch.isfinite(grad_norm):
            raise RuntimeError('Nonfinite SFT gradients')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if world > 1:
            dist.all_reduce(stats)
        values = training_metrics(stats, config.residual_codec_loss_weight)
        loss_history.append(values['train/loss'])
        values.update({
            'train/loss_ma100': sum(loss_history) / len(loss_history),
            'train/grad_norm': float(grad_norm), 'train/progress': step / total_steps,
            'train/data_wait_seconds': data_wait,
            'train/seconds_per_update': time.perf_counter() - started,
            'train/max_sequence_length': max(lengths),
            'train/peak_memory_GiB': torch.cuda.max_memory_allocated(device) / 2**30,
        })
        for group in optimizer.param_groups:
            values['train/lr_' + group['name']] = group['lr']
        if rank == 0:
            atomic_json(output / 'train-latest.json', {'step': step, 'micro_step': micro, **values})
            with (output / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': step, 'micro_step': micro, **values}) + '\n')
            if swanlab:
                swanlab.log(values, step=step)
            print('SFT_STEP', step, '/', total_steps, 'loss', round(values['train/loss'], 5),
                  'grad', round(float(grad_norm), 3), 'seconds', round(values['train/seconds_per_update'], 2), flush=True)
        if benchmarking:
            torch.cuda.synchronize(device)
            measures = torch.tensor([
                time.perf_counter() - started, data_wait,
                torch.cuda.max_memory_allocated(device) / 2**30,
                torch.cuda.max_memory_reserved(device) / 2**30,
            ], dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(measures, op=dist.ReduceOp.MAX)
            seconds, wait, allocated, reserved = measures.cpu().tolist()
            record = {'step': step, 'warmup': step <= args.benchmark_warmup,
                      'seconds': seconds, 'data_wait_seconds': wait,
                      'peak_allocated_GiB': allocated, 'peak_reserved_GiB': reserved,
                      'samples': values['train/samples'],
                      'text_tokens': values['train/text_tokens'],
                      'audio_frames': values['train/main_codec_tokens']}
            benchmark_records.append(record)
            if rank == 0:
                atomic_json(output / 'benchmark-progress.json', record)
                print('SFT_BENCHMARK_STEP', json.dumps(record), flush=True)
        complete = micro == micro_total
        stopping = args.max_steps > 0 and step >= args.max_steps
        if not benchmarking and (step % plan['eval_interval'] == 0 or complete):
            validation()
        if not args.no_checkpoint and (step == 1 or step % plan['save_interval'] == 0 or complete or stopping):
            save_checkpoint(complete)
            last_saved = step
        if stopping:
            break
    if rank == 0:
        if benchmarking:
            measured = [row for row in benchmark_records if not row['warmup']]
            seconds = [row['seconds'] for row in measured]
            atomic_json(output / 'benchmark.json', {
                'status': 'PASS', 'world_size': world, 'batch_size': batch_size,
                'accumulation_steps': accumulation, 'global_batch': batch_size * world * accumulation,
                'optimizer_steps': step, 'warmup_steps': args.benchmark_warmup,
                'mean_seconds': statistics.mean(seconds), 'median_seconds': statistics.median(seconds),
                'samples_per_second': sum(row['samples'] for row in measured) / sum(seconds),
                'peak_allocated_GiB': max(row['peak_allocated_GiB'] for row in benchmark_records),
                'peak_reserved_GiB': max(row['peak_reserved_GiB'] for row in benchmark_records),
                'initialization': 'qwen backbone with untrained multimodal modules' if args.benchmark_from_qwen else str(checkpoint),
                'checkpoint_saved': False, 'tracking': False, 'records': benchmark_records,
            })
        atomic_json(output / 'status.json', {'status': 'COMPLETE' if micro == micro_total else 'STEP_LIMIT',
                                           'step': step, 'micro_step': micro, 'total_steps': total_steps,
                                           'last_saved_step': last_saved})
        if swanlab:
            swanlab.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
