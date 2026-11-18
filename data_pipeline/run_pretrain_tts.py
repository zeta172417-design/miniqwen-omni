#!/usr/bin/env python3
"""Wait for raw TTS manifests, encode on four devices, and validate shard totals."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', choices=['libritts_r', 'aishell3'], default=['libritts_r', 'aishell3'])
    sources = parser.parse_args().sources
    log = ROOT / os.environ.get('MINIQWEN_PRETRAIN_LOG', '.runtime/pretrain')
    log.mkdir(parents=True, exist_ok=True)
    data = ROOT / os.environ.get('MINIQWEN_PRETRAIN_DATA', 'dataset/pretrain')
    devices = queue.Queue()
    for i in range(4):
        devices.put(i)

    def encode(source, shard):
        device = devices.get()
        try:
            path = log / f'encode-{source}-{shard}.log'
            with path.open('a') as f:
                command = [sys.executable, '-B', '-u', str(ROOT / 'data_pipeline/encode_pretrain_tts.py'),
                           '--source', source, '--device', f'cuda:{device}',
                           '--shard-index', str(shard), '--shard-count', '4']
                env = {**os.environ, 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{source} shard {shard} failed; see {path}')
        finally:
            devices.put(device)

    pending = set(sources)
    started = time.monotonic()
    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending or jobs:
            for source in list(pending):
                manifest = data / 'processed' / source / 'manifest.json'
                if manifest.exists():
                    print('ENCODE_SOURCE_START', source, flush=True)
                    jobs[source] = [pool.submit(encode, source, i) for i in range(4)]
                    pending.remove(source)
            for source, futures in list(jobs.items()):
                if not all(future.done() for future in futures):
                    continue
                for future in futures:
                    future.result()
                base = data / 'processed' / source
                original = json.loads((base / 'manifest.json').read_text())
                summaries = [json.loads((base / f'codec/{i:02d}-of-04/manifest.json').read_text()) for i in range(4)]
                counts = {split: sum(s['splits'].get(split, {}).get('rows', 0) for s in summaries) for split in original['splits']}
                if any(counts[split] != values['rows'] for split, values in original['splits'].items()):
                    raise RuntimeError(f'Encoded sample count mismatch: {source}')
                original['status'] = 'training_format_ready'
                original['codec'] = {'model': 'model/mimi', 'sample_rate': 24000, 'quantizers': 8,
                                 'precision': 'float32', 'frame_major': True, 'speaker_reference': False,
                                 'terminal_padding': 'encoding input only: right zero pad to 1920 samples, <80ms; original waveform unchanged',
                                 'shard_manifests': [f'codec/{i:02d}-of-04/manifest.json' for i in range(4)]}
                original['data_files'] = {split: [str((base / f'codec/{i:02d}-of-04/{split}.parquet').relative_to(ROOT))
                                             for i in range(4) if split in summaries[i]['splits']] for split in counts}
                temp = base / 'manifest.json.tmp'
                temp.write_text(json.dumps(original, ensure_ascii=False, indent=2) + '\n')
                os.replace(temp, base / 'manifest.json')
                print('ENCODE_SOURCE_COMPLETE', source, counts, flush=True)
                del jobs[source]
            if pending and time.monotonic() - started > 6 * 3600:
                raise RuntimeError('Raw TTS manifests not ready after six hours')
            if pending or jobs:
                time.sleep(5)


if __name__ == '__main__':
    main()
