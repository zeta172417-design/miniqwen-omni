#!/usr/bin/env python3
"""Columnar Parquet writer used by the canonical pretrain reorganizer."""
import collections
import pyarrow as pa
import pyarrow.parquet as pq


def convert_table(table, metadata, schema, source, split, raw_path, selected):
    import reorganize_pretrain_data as prep
    rows = [prep.normalized(row, source, split, raw_path,
                            int(info.get('_row', 0)), selected)
            for row, info in zip(table.to_pylist(), metadata)]
    return pa.Table.from_pylist(rows, schema=schema)


def build_source_columnar(source, spec, context):
    """Build one source using the reorganizer's normalization and Writer."""
    (root, work, meta, stage_root, schema, writer_cls, partition, family,
     save, rel, digest) = context
    import reorganize_pretrain_data as prep
    writer = writer_cls(spec['stage'], source)
    raw_counts = collections.Counter()
    speaker_sets = collections.defaultdict(set)
    for split, paths in spec['raw_paths'].items():
        for raw_path in paths:
            offset = 0
            for batch in pq.ParquetFile(root / raw_path).iter_batches(
                    batch_size=128, use_threads=False):
                accepted = []
                for index, row in enumerate(batch.to_pylist()):
                    absolute = offset + index
                    raw_counts[split] += 1
                    speaker = str(row.get('speaker_id') or '')
                    selected = (partition(source, speaker)
                                if spec['supplement'] and split == 'train'
                                else 'pretrain' if split == 'train' else 'dev')
                    if selected == 'sft_reserved':
                        speaker_sets[selected].add(speaker)
                        continue
                    accepted.append(prep.normalized(
                        row, source, split, raw_path, absolute, selected))
                    speaker_sets[selected].add(speaker)
                writer.add(split, accepted)
                offset += len(batch)
    for split, expected in spec.get('expected', {}).items():
        if raw_counts[split] != expected:
            raise ValueError((source, split, raw_counts[split], expected))
    if speaker_sets['pretrain'] & speaker_sets['sft_reserved']:
        raise ValueError('speaker_phase_overlap')
    result = writer.finish()
    result.update(input_rows=dict(raw_counts),
                  speaker_counts={k: len(v - {''}) for k, v in speaker_sets.items()})
    save(work / 'source-results' / f'{source}.json', result)
    return result
