#!/usr/bin/env python3
"""Build one catalog; optionally derive a non-destructive official-clean TTS view."""
import argparse
import json
import os
from pathlib import Path
import tarfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dataset/pretrain'


def official_flags():
    archive = BASE / 'raw/libritts_r/libritts_r_failed_speech_restoration_examples.tar.gz'
    flags = {}
    with tarfile.open(archive) as t:
        for member in t:
            for split, name in [('train', 'train-clean-100_bad_sample_list.txt'), ('dev', 'dev-clean_bad_sample_list.txt')]:
                if member.isfile() and Path(member.name).name == name:
                    lines = t.extractfile(member).read().decode().splitlines()
                    flags[split] = sorted('libritts_r:' + Path(line.strip()).stem for line in lines if line.strip())
    return flags


def clean_file(source, destination, excluded):
    pf = pq.ParquetFile(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.partial')
    stats = {'rows': 0, 'seconds': 0.0, 'codec_frames': 0, 'max_duration': 0.0}
    with pq.ParquetWriter(temp, pf.schema_arrow, compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=512):
            table = pa.Table.from_batches([batch])
            table = table.filter(pc.invert(pc.is_in(table['sample_id'], value_set=pa.array(excluded, type=pa.string()))))
            if not table.num_rows:
                continue
            writer.write_table(table, row_group_size=128)
            stats['rows'] += table.num_rows
            stats['seconds'] += float(pc.sum(table['duration']).as_py())
            stats['codec_frames'] += int(pc.sum(table['codec_frames']).as_py())
            stats['max_duration'] = max(stats['max_duration'], float(pc.max(table['duration']).as_py()))
    os.replace(temp, destination)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libritts-quality', choices=['pending', 'exclude_official', 'keep_all'], default='pending')
    args = p.parse_args()
    flags = official_flags()
    flag_path = BASE / 'processed/libritts_r/official_restoration_flags.json'
    flag_path.write_text(json.dumps({'source': 'OpenSLR 141 official restoration-failure lists',
                                    'policy': args.libritts_quality, 'sample_ids': flags}, indent=2) + '\n')
    catalog = {'format_version': 1, 'libritts_quality_policy': args.libritts_quality,
               'quality_decision_pending': args.libritts_quality == 'pending', 'sources': {}}
    for name in ['librispeech', 'aishell1', 'llava', 'libritts_r', 'aishell3']:
        base = BASE / 'processed' / name
        metadata = json.loads((base / 'manifest.json').read_text())
        if metadata['status'] != 'training_format_ready':
            raise RuntimeError(f'{name}: data is not ready')
        files = metadata.get('data_files') or {s: [str((base / (s + '.parquet')).relative_to(ROOT))] for s in metadata['splits']}
        selected = {s: dict(v) if isinstance(v, dict) else {'rows': v} for s, v in metadata['splits'].items()}
        full_counts = {s: v['rows'] for s, v in selected.items()}
        if name == 'libritts_r' and args.libritts_quality == 'exclude_official':
            new_files = {}
            for split, paths in files.items():
                totals = {'rows': 0, 'seconds': 0.0, 'codec_frames': 0, 'max_duration': 0.0}
                new_files[split] = []
                for path in paths:
                    destination = base / 'official_clean' / (ROOT / path).relative_to(base)
                    stats = clean_file(ROOT / path, destination, flags.get(split, []))
                    new_files[split].append(str(destination.relative_to(ROOT)))
                    for k in ['rows', 'seconds', 'codec_frames']:
                        totals[k] += stats[k]
                    totals['max_duration'] = max(totals['max_duration'], stats['max_duration'])
                selected[split] = totals
            files = new_files
        catalog['sources'][name] = {'source_manifest': str((base / 'manifest.json').relative_to(ROOT)),
                                     'data_files': files, 'splits': selected, 'all_encoded_or_converted_rows': full_counts,
                                     'quarantined': metadata.get('quarantined', 0)}
        if name == 'libritts_r':
            catalog['sources'][name]['official_flagged_counts'] = {s: len(ids) for s, ids in flags.items()}
            catalog['sources'][name]['quality_policy'] = args.libritts_quality
            catalog['sources'][name]['excluded_counts'] = {s: full_counts[s] - selected[s]['rows'] for s in selected}
        print('CATALOG_SOURCE', name, {s: v['rows'] for s, v in selected.items()}, flush=True)
    target = BASE / 'manifest.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, target)
    print('CATALOG_READY', target, flush=True)


if __name__ == '__main__':
    main()
