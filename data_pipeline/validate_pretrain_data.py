#!/usr/bin/env python3
"""Validate representative real samples through OmniDataset without training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MINIQWEN_DATASET_CACHE', str(ROOT / '.runtime/pretrain/validation-cache'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/pretrain/numba'))

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import SiglipImageProcessor
from dataset.omni_dataset import OmniDataset
from model.model_omni import MiniQwenOmni
from trainer.trainer_utils import load_omni_tokenizer


def selected_rows(path):
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    if not n:
        return []
    indices = {0, n // 2, n - 1}
    if 'duration' in pf.schema_arrow.names:
        values = pf.read(columns=['duration'])['duration'].to_numpy()
        indices.add(int(values.argmax()))
    result = []
    offset = 0
    for group in range(pf.num_row_groups):
        count = pf.metadata.row_group(group).num_rows
        local = sorted(i - offset for i in indices if offset <= i < offset + count)
        if local:
            table = pf.read_row_group(group)
            result.extend(table.take(pa.array(local)).to_pylist())
        offset += count
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', default=['librispeech', 'aishell1', 'llava', 'libritts_r', 'aishell3'])
    args = p.parse_args()
    tokenizer = load_omni_tokenizer(ROOT / 'model/Qwen3-0.6B')
    _, audio_processor = MiniQwenOmni.load_sensevoice(ROOT / 'model/SenseVoiceSmall')
    vision_processor = SiglipImageProcessor.from_pretrained(ROOT / 'model/siglip2-base-p32-256-ve', local_files_only=True)
    result = {}
    catalog_path = ROOT / 'dataset/pretrain/manifest.json'
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else None
    directory = ROOT / '.runtime/pretrain/validation-inputs'
    directory.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        base = ROOT / 'dataset/pretrain/processed' / source
        manifest = json.loads((base / 'manifest.json').read_text())
        if manifest['status'] != 'training_format_ready':
            raise RuntimeError(f'{source} is not in training format yet')
        file_spec = catalog['sources'][source]['data_files'] if catalog else manifest.get('data_files', {})
        files = [ROOT / f for paths in file_spec.values() for f in paths]
        if not files:
            files = [base / 'train.parquet', base / 'dev.parquet']
        rows = [row for path in files for row in selected_rows(path)]
        path = directory / (source + '.parquet')
        pq.write_table(pa.Table.from_pylist(rows), path)
        ds = OmniDataset(str(path), tokenizer, audio_processor=audio_processor, vision_processor=vision_processor,
                         max_length=8192, training=False, use_modality_boundaries=True, use_talker_ref_boundaries=True)
        checks = []
        for i, row in enumerate(rows):
            ids, attention, text, codes, audio, audio_len, pixels, speaker = ds[i]
            assert (text != -100).any(), row['sample_id']
            if row['task_type'] == 'asr':
                assert audio_len > 0 and (ids[8] == ds.audio_token_id).any(), row['sample_id']
                user_text = json.loads(row['conversations'])[0]['content']
                decoded = tokenizer.decode(ids[8].tolist(), skip_special_tokens=False)
                assert user_text in decoded, row['sample_id']
            elif row['task_type'] == 'i2t':
                assert int((ids[8] == ds.image_token_id).sum()) == 64, row['sample_id']
                assert pixels is not None and pixels.any(), row['sample_id']
            elif row['task_type'] == 'tts':
                assert int((codes[0] != -100).sum()) == row['codec_frames'] + 1, row['sample_id']
                assert int((codes[0] == ds.audio_stop_token).sum()) == 1, row['sample_id']
            checks.append({'sample_id': row['sample_id'], 'task': row['task_type'],
                           'text_labels': int((text != -100).sum()), 'audio_labels': int((codes != -100).sum()),
                           'audio_input_frames': int(audio_len), 'text_context_tokens': int(attention.sum())})
        result[source] = {'status': 'passed', 'max_length': 8192, 'samples': checks}
        print('DATASET_SAMPLE_VALIDATION_PASS', source, len(checks), flush=True)
    target = ROOT / '.runtime/pretrain' / ('validation-' + '-'.join(args.sources) + '.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
