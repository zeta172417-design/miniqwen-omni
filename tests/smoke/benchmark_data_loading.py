#!/usr/bin/env python3
"""Bounded concurrent reader benchmark; no accelerator model or optimizer."""
import argparse
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure_rank(dataset, rank, ranks, batch_size, batches, workers, prefetch, cache, results):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from trainer.training_common import omni_collate_fn
    try:
        torch.set_num_threads(1)
        os.environ['MINIQWEN_PARQUET_MEMORY_CACHE_MIB'] = str(cache)
        # The benchmark forks an already constructed reader; set its explicit budget too.
        from dataset.parquet_reader import ParquetRows
        for obj in [getattr(dataset, 'dataset', None),
                    getattr(getattr(dataset, 'dataset', None), 'rows', None)]:
            if isinstance(obj, ParquetRows):
                obj.cache_bytes = cache * 1024**2
        sampler = DistributedSampler(dataset, num_replicas=ranks, rank=rank, shuffle=True, seed=42)
        indices = list(iter(sampler))[:batch_size * batches]
        groups = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        options = dict(num_workers=workers, pin_memory=False)
        if workers:
            options.update(prefetch_factor=prefetch, multiprocessing_context='fork')
        loader = DataLoader(dataset, batch_sampler=groups,
                            collate_fn=partial(omni_collate_fn, dynamic_padding=True), **options)
        start = time.perf_counter()
        ticks = []
        for batch in loader:
            ticks.append(time.perf_counter())
        results.put({'rank': rank, 'batches': len(ticks), 'rows': len(indices),
                     'total_seconds': time.perf_counter() - start,
                     'first_batch_seconds': ticks[0] - start,
                     'steady_seconds': ticks[-1] - ticks[min(3, len(ticks) - 1)],
                     'steady_batches': max(0, len(ticks) - 4)})
    except Exception as error:
        import traceback
        results.put({'rank': rank, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ranks', type=int, default=16)
    parser.add_argument('--batches', type=int, default=24)
    parser.add_argument('--settings', nargs='+', default=['1:2:128', '2:2:128', '2:4:256', '4:2:128'],
                        help='Ordered workers:prefetch:cache_MiB configurations')
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    torch.set_num_threads(1)
    from dataset.sft_dataset import SFTDataset
    from model.model_omni import MiniQwenOmni
    from trainer.trainer_utils import load_omni_tokenizer
    from transformers import SiglipImageProcessor
    tokenizer = load_omni_tokenizer(ROOT / 'model/Qwen3-0.6B')
    encoder, audio = MiniQwenOmni.load_sensevoice(ROOT / 'model/SenseVoiceSmall')
    del encoder
    vision = SiglipImageProcessor.from_pretrained(ROOT / 'model/siglip2-base-p32-256-ve', local_files_only=True)
    manifest = json.loads((ROOT / 'dataset/sft/manifest.json').read_text())
    paths = manifest['tiers']['main_2048']['data_files']['train']
    # Use the complete pool and true global random indices, not a warmed small subset.
    dataset = SFTDataset(','.join(str(ROOT / p) for p in paths), tokenizer, audio, vision,
                         max_length=2048, training=False, augment=False,
                         scheduled_sampling=0, strict_length=True, condition_prefix_length=160,
                         use_modality_boundaries=True, use_talker_ref_boundaries=True, max_images=4)
    assert not torch.cuda.is_initialized(), 'CPU-only benchmark must not fork a GPU context'
    context = mp.get_context('fork')
    records = []
    for setting in args.settings:
        workers, prefetch, cache = map(int, setting.split(':'))
        print('LOADER_BENCHMARK_START', workers, prefetch, cache, flush=True)
        results = context.Queue()
        jobs = [context.Process(target=measure_rank, args=(dataset, rank, args.ranks, 4,
                args.batches, workers, prefetch, cache, results)) for rank in range(args.ranks)]
        start = time.perf_counter()
        for job in jobs:
            job.start()
        output = []
        try:
            for _ in jobs:
                output.append(results.get(timeout=240))
            for job in jobs:
                job.join(timeout=30)
            assert all(job.exitcode == 0 for job in jobs), 'Reader subprocess failed'
            assert not any('error' in row for row in output), output
        finally:
            for job in jobs:
                if job.is_alive():
                    job.terminate()
                job.join(timeout=5)
        elapsed = time.perf_counter() - start
        record = dict(workers_per_rank=workers, prefetch=prefetch, cache_mib_per_process=cache,
                      ranks=args.ranks, batches_per_rank=args.batches,
                      wall_seconds=elapsed, rows_per_second=sum(x['rows'] for x in output) / elapsed,
                      steady_rows_per_second=sum(x['steady_batches'] * 4 for x in output) /
                          max(max(x['steady_seconds'] for x in output), 1e-9),
                      max_first_batch_seconds=max(x['first_batch_seconds'] for x in output),
                      max_rank_seconds=max(x['total_seconds'] for x in output))
        records.append(record)
        Path(args.output).write_text(json.dumps({'status': 'IN_PROGRESS', 'records': records}, indent=2) + '\n')
        print('LOADER_BENCHMARK_RESULT', json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps({'status': 'PASS', 'records': records,
        'limitations': 'CPU reader-only throughput; no model/GPU overlap; configurations run in order with shared OS cache.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
