#!/usr/bin/env python3
"""Encode approved basic TTS segments with local Mimi; no cutting or speaker references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
import torch
from transformers import MimiModel

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_waveform(row):
    wav, sr = sf.read(ROOT / row['audio_path'], dtype='float32', always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 24000:
        g = math.gcd(sr, 24000)
        wav = resample_poly(wav, 24000 // g, sr // g).astype(np.float32)
    if not len(wav) or not np.isfinite(wav).all():
        raise ValueError(f"Invalid waveform: {row['sample_id']}")
    # Mimi's internal convolution padding changes the terminal code when
    # unequal raw lengths are batched. Canonicalize only the encoding input
    # to full 1920-sample (80ms) frames; preserve the original file/duration.
    wav = np.pad(wav, (0, (-len(wav)) % 1920))
    return wav


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['libritts_r', 'aishell3'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--decode-workers', type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        p.error('shard-index must be in [0, shard-count)')
    base = ROOT / os.environ.get('MINIQWEN_PRETRAIN_DATA', 'dataset/pretrain') / 'processed' / args.source
    segments = base / 'segments.jsonl'
    metadata = json.loads((base / 'manifest.json').read_text())
    key = f'{args.shard_index:02d}-of-{args.shard_count:02d}'
    target = base / 'codec' / key
    target.mkdir(parents=True, exist_ok=True)
    fingerprint = {'segments_sha256': digest(segments),
                   'mimi_sha256': digest(ROOT / 'model/mimi/model.safetensors'),
                   'mimi_config_sha256': digest(ROOT / 'model/mimi/config.json'),
                   'sample_rate': 24000, 'num_quantizers': 8, 'precision': 'float32',
                   'flatten_order': 'frame_major_c0_to_c7', 'resampler': 'scipy.signal.resample_poly',
                   'terminal_padding': 'right_zero_to_multiple_of_1920_samples; less_than_80ms',
                   'batch_size': args.batch_size, 'encoder_format_version': 1,
                   'shard_index': args.shard_index, 'shard_count': args.shard_count}
    summary_path = target / 'manifest.json'
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise RuntimeError('Existing encoding fingerprint differs; use a new output directory')
        if all((target / name).exists() and digest(target / name) == sha for name, sha in old['sha256'].items()):
            print('ENCODE_ALREADY_COMPLETE', args.source, key, flush=True)
            return
        raise RuntimeError('Completed codec shard is missing or corrupt')
    rows = [json.loads(line) for line in segments.open()]
    rows = [r for r in rows if int(hashlib.sha256(r['sample_id'].encode()).hexdigest()[:8], 16) % args.shard_count == args.shard_index]
    rows.sort(key=lambda r: (r['duration'], r['sample_id']))
    model = MimiModel.from_pretrained(ROOT / 'model/mimi', local_files_only=True).eval().float().to(args.device)
    model.config.use_streaming = False
    schema = pa.schema([('sample_id', pa.string()), ('source', pa.string()), ('split', pa.string()),
                        ('task_type', pa.string()), ('language', pa.string()), ('speaker_id', pa.string()),
                        ('text_original', pa.string()), ('text_target', pa.string()), ('duration', pa.float64()),
                        ('codec_frames', pa.int32()), ('codec_padding_samples', pa.int32()), ('conversations', pa.string()),
                        ('answer_audios', pa.list_(pa.list_(pa.int64())))])
    writers, counts, pending = {}, {}, {}
    started = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool, torch.inference_mode():
        cursor = 0
        while cursor < len(rows):
            duration = rows[cursor]['duration']
            # Limit padded seconds per batch without dropping/cutting long rows.
            batch_size = max(1, min(args.batch_size, int(120 / max(duration + 1, 1))))
            selected = rows[cursor:cursor + batch_size]
            waves = list(pool.map(read_waveform, selected))
            lengths = torch.tensor([len(w) for w in waves], dtype=torch.long)
            frame_lengths = model.get_encoded_length(lengths.to(args.device)).cpu().tolist()
            inputs = torch.zeros(len(waves), 1, max(map(len, waves)), dtype=torch.float32)
            for i, wav in enumerate(waves):
                inputs[i, 0, :len(wav)] = torch.from_numpy(wav)
            codes = model.encode(inputs.to(args.device), num_quantizers=8, use_streaming=False).audio_codes.cpu()
            if codes.shape[1] != 8 or codes.min() < 0 or codes.max() >= 2048:
                raise RuntimeError('Invalid Mimi code shape or vocabulary')
            for i, row in enumerate(selected):
                frames = int(frame_lengths[i])
                if frames <= 0 or frames > codes.shape[-1]:
                    raise RuntimeError('Invalid encoded length')
                text = row['text_normalized']
                prompt = ('Read the following text aloud:\n' if row['language'] == 'en' else '请朗读以下文本：\n') + text
                record = dict(sample_id=row['sample_id'], source=args.source, split=row['split'], task_type='tts',
                              language=row['language'], speaker_id=row['speaker_id'], text_original=row['text_original'],
                              text_target=text, duration=row['duration'], codec_frames=frames,
                              codec_padding_samples=len(waves[i]) - math.ceil(row['samples'] * 24000 / row['sample_rate']),
                              conversations=json.dumps([{'role': 'user', 'content': prompt},
                                                        {'role': 'assistant', 'content': text}], ensure_ascii=False),
                              answer_audios=[codes[i, :, :frames].transpose(0, 1).reshape(-1).tolist()])
                split = row['split']
                if split not in writers:
                    writers[split] = pq.ParquetWriter(target / (split + '.parquet.partial'), schema, compression='zstd')
                    counts[split] = {'rows': 0, 'seconds': 0.0, 'codec_frames': 0}
                    pending[split] = []
                pending[split].append(record)
                if len(pending[split]) >= 128:
                    writers[split].write_table(pa.Table.from_pylist(pending[split], schema=schema))
                    pending[split].clear()
                st = counts[split]
                st['rows'] += 1
                st['seconds'] += row['duration']
                st['codec_frames'] += frames
            cursor += len(selected)
            if cursor - done >= 1000 or cursor == len(rows):
                done = cursor
                print('ENCODE_PROGRESS', args.source, key, cursor, '/', len(rows), 'elapsed', round(time.monotonic()-started, 1), flush=True)
    checksums = {}
    for split, writer in writers.items():
        if pending[split]:
            writer.write_table(pa.Table.from_pylist(pending[split], schema=schema))
        writer.close()
        path = target / (split + '.parquet')
        os.replace(target / (split + '.parquet.partial'), path)
        checksums[path.name] = digest(path)
    summary = {'source': args.source, 'fingerprint': fingerprint, 'splits': counts, 'sha256': checksums,
               'status': 'codec_shard_ready', 'speaker_reference': False}
    temp = summary_path.with_suffix('.tmp')
    temp.write_text(json.dumps(summary, indent=2))
    os.replace(temp, summary_path)
    print('ENCODE_COMPLETE', args.source, key, flush=True)


if __name__ == '__main__':
    main()
