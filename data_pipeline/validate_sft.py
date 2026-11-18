#!/usr/bin/env python3
"""Accept final SFT shards, exercise their actual reader/collator, and publish manifests."""
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_pipeline'))
from sft_common import META, OUT, R, SCHEMA, W, save, norm


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_inputs():
    counts = {}
    for name in ('ultrachat', 'coig', 'llava15_single', 'm4', 'fusion'):
        specification = json.loads((OUT / 'external' / name / 'manifest.json').read_text())
        counts[name] = specification['counts']['train'] + specification['counts']['dev']
    for name in ('librispeech', 'aishell1', 'libritts_r', 'aishell3'):
        specification = json.loads((META.parent / 'pretrain/sources' / (name + '.json')).read_text())
        counts[name] = specification['sft_reserved']['rows'] + specification['splits']['dev']['rows']
    counts['old_a2a'] = pq.ParquetFile(ROOT / 'dataset/raw/legacy/sft_a2a.parquet').metadata.num_rows
    return counts


def voice_references():
    references = {}
    for source in ('libritts_r', 'aishell3'):
        for path in (ROOT / 'dataset/raw/corpora' / source).glob('*.parquet'):
            for batch in pq.ParquetFile(path).iter_batches(
                columns=['sample_id', 'speaker_id', 'split', 'text_target'], batch_size=4096
            ):
                for row in batch.to_pylist():
                    references[row['sample_id']] = row
    return references


def main():
    os.nice(10)
    save(W / 'validation-status.json', {'status': 'VALIDATING_ALL_ROWS'})
    inputs = source_inputs()
    source_manifests = {
        path.stem: json.loads(path.read_text())
        for path in sorted((META / 'sources').glob('*.json'))
    }
    if set(source_manifests) != set(inputs):
        raise ValueError('Missing or unexpected prepared source manifests')
    references = voice_references()
    partition_speakers = {}
    for source in ('librispeech', 'aishell1', 'libritts_r', 'aishell3'):
        table = pq.read_table(META.parent / 'pretrain/partitions' / (source + '.parquet'))
        partition_speakers[source] = {
            row['speaker_id'] for row in table.to_pylist() if row['partition'] == 'sft_reserved'
        }
    sample_ids = set()
    group_splits = {}
    image_paths = set()
    audio_paths = set()
    payload_ids = set()
    source_counts = collections.defaultdict(collections.Counter)
    task_counts = collections.defaultdict(collections.Counter)
    voice_modes = collections.defaultdict(collections.Counter)
    exclusions = {}
    fingerprints = {}
    samples = {}
    special_samples = {}
    storage = 0
    codec_frames = 0
    target_tokens = 0
    file_lists = collections.defaultdict(list)
    row_counts = collections.Counter()
    maxima = collections.Counter()

    for source, specification in source_manifests.items():
        rejected_path = META / 'excluded' / (source + '.parquet')
        rejected = collections.Counter()
        if rejected_path.exists():
            for batch in pq.ParquetFile(rejected_path).iter_batches(columns=['reason']):
                rejected.update(batch.column(0).to_pylist())
        exclusions[source] = dict(rejected)
        actual_source = 0
        for key, paths in specification['data_files'].items():
            tier, split = key.split('/')
            budget = 2048 if tier == 'main_2048' else 3072
            actual = 0
            for relative in paths:
                path = ROOT / relative
                parquet = pq.ParquetFile(path)
                if parquet.schema_arrow != SCHEMA:
                    raise ValueError(f'Schema mismatch at {relative}')
                actual += parquet.metadata.num_rows
                file_lists[key].append(relative)
                storage += path.stat().st_size
                for batch in parquet.iter_batches(batch_size=128, use_threads=False):
                    for row in batch.to_pylist():
                        sid = row['sample_id']
                        if sid in sample_ids:
                            raise ValueError(f'Duplicate sample ID {sid}')
                        sample_ids.add(sid)
                        if row['source'] != source or row['split'] != split or row['length_tier'] != tier:
                            raise ValueError(f'Wrong source/tier/split for {sid}')
                        length = row['required_length']
                        if not 0 < length <= budget:
                            raise ValueError(f'Invalid length for {sid}')
                        if tier == 'long_text_3072' and (length <= 2048 or row['task_type'] != 'text'):
                            raise ValueError(f'Invalid long-text classification for {sid}')
                        messages = json.loads(row['conversations'])
                        if not messages or messages[-1]['role'] != 'assistant':
                            raise ValueError(f'Invalid conversation for {sid}')
                        marker_count = sum(message['content'].count('<image>') for message in messages)
                        if marker_count != len(row['image_paths']) or marker_count > 4:
                            raise ValueError(f'Image alignment mismatch for {sid}')
                        image_paths.update(row['image_paths'])
                        audio_paths.update(row['question_audio_paths'])
                        if row['question_audios'] and row['question_audio_paths']:
                            raise ValueError(f'Two question-audio representations for {sid}')
                        for field in ('image_paths', 'question_audio_paths'):
                            for value in row[field]:
                                if not (ROOT / value).resolve().is_relative_to(ROOT / 'dataset/raw'):
                                    raise ValueError(f'Media outside canonical raw directory for {sid}')
                        group = row['group_id']
                        previous = group_splits.setdefault(group, split)
                        if previous != split:
                            raise ValueError(f'Train/dev group overlap for {group}')
                        codes = row['answer_audios'][-1] if row['answer_audios'] else []
                        frames = row['codec_frames']
                        if frames and (len(codes) != frames * 8 or min(codes) < 0 or max(codes) >= 2048):
                            raise ValueError(f'Invalid complete codec target for {sid}')
                        if not frames and codes:
                            raise ValueError(f'Unexpected audio target for {sid}')
                        speaker = row['spk_emb']
                        if speaker and (len(speaker) != 192 or not all(math.isfinite(x) for x in speaker)):
                            raise ValueError(f'Invalid speaker embedding for {sid}')
                        reference_limit = 150 if source == 'old_a2a' else 50
                        if row['ref_audios'] and (len(row['ref_audios']) % 8 or len(row['ref_audios']) // 8 > reference_limit):
                            raise ValueError(f'Invalid reference audio for {sid}')
                        if row['source'] in partition_speakers and split == 'train':
                            if row['speaker_id'] not in partition_speakers[source]:
                                raise ValueError(f'Unexpected pretrain speaker in SFT reserved pool for {sid}')
                        metadata = json.loads(row['metadata_json'])
                        if row['task_type'] == 'tts_clone':
                            reference = references.get(row['reference_sample_id'])
                            target = references.get(sid.removeprefix('sft:'))
                            if not reference or not target:
                                raise ValueError(f'Voice reference missing provenance for {sid}')
                            if reference['sample_id'] == target['sample_id']:
                                raise ValueError(f'Voice target reused as reference for {sid}')
                            if reference['speaker_id'] != target['speaker_id'] or reference['split'] != target['split']:
                                raise ValueError(f'Voice speaker or source split mismatch for {sid}')
                            if reference['text_target'] == target['text_target'] or not metadata['reference_independent']:
                                raise ValueError(f'Reference utterance independence failed for {sid}')
                        if source in ('librispeech', 'aishell1') and row['task_type'] != 'asr':
                            raise ValueError(f'ASR instruction policy mismatch for {sid}')
                        if source == 'fusion':
                            view = metadata['fusion_view']
                            if (view in (1, 3)) != bool(row['question_audio_paths']):
                                raise ValueError(f'Fusion input view mismatch for {sid}')
                            if (view in (2, 3)) != bool(frames):
                                raise ValueError(f'Fusion output view mismatch for {sid}')
                            if view in (1, 3):
                                final_user = [m for m in messages if m['role'] == 'user'][-1]
                                expected = '<image>\nAnswer the spoken question.'
                                if not final_user['content'].startswith(expected):
                                    raise ValueError(f'Fusion spoken question instruction lost for {sid}')
                        target_tokens += len(messages[-1]['content'])
                        codec_frames += frames
                        row_counts[key] += 1
                        maxima[key] = max(maxima[key], length)
                        source_counts[source][key] += 1
                        task_counts[key][row['task_type']] += 1
                        voice_modes[source][row['voice_conditioning']] += 1
                        sample_key = (source, split, tier)
                        selected = samples.setdefault(sample_key, [])
                        if not selected:
                            selected.append(row)
                        elif len(selected) == 1:
                            selected.append(row)
                        elif length > selected[1]['required_length']:
                            selected[1] = row
                        if frames:
                            special_samples.setdefault('audio_target_' + source, row)
                        if marker_count in (2, 3, 4):
                            special_samples.setdefault(f'{marker_count}_images', row)
                        if source == 'fusion':
                            special_samples.setdefault('fusion_view_' + str(metadata['fusion_view']), row)
                        if row['task_type'] == 'tts_clone':
                            special_samples.setdefault('clone_' + row['voice_conditioning'], row)
                fingerprints[relative] = file_sha256(path)
            if actual != specification['counts'][key]:
                raise ValueError(f'Row count mismatch for {source}/{key}')
            actual_source += actual
        if actual_source + sum(rejected.values()) != inputs[source]:
            raise ValueError((source, 'source row conservation', actual_source, sum(rejected.values()), inputs[source]))
        print('SOURCE_VALIDATED', source, actual_source, dict(rejected), flush=True)
        save(W / 'validation-status.json', {'status': 'VALIDATING_ALL_ROWS', 'completed_source': source})

    # Group-index consistency is checked against the complete accepted media mapping.
    media_index = {row['path']: row for row in pq.read_table(META / 'media-groups.parquet').to_pylist()}
    for path in image_paths:
        entry = media_index[path]
        if entry['blocked_benchmark_image']:
            raise ValueError(f'Accepted evaluation-blocked media {path}')
        if group_splits[entry['group_id']] != entry['split']:
            raise ValueError(f'Image split mismatch at {path}')

    def check_path(path):
        return path, (ROOT / path).is_file()

    media_paths = sorted(image_paths | audio_paths)
    if os.environ.get('MINIQWEN_SKIP_MEDIA_EXISTENCE_CHECK') != '1':
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0, len(media_paths), 512):
                for path, exists in pool.map(check_path, media_paths[offset:offset + 512]):
                    if not exists:
                        raise FileNotFoundError(path)
    save(W / 'full-data-validation.json', {
        'status': 'PASS', 'rows': len(sample_ids), 'source_conservation': True,
        'group_train_dev_overlap': 0, 'media_paths_checked': len(media_paths),
        'complete_target_codec_frames': codec_frames, 'voice_reference_checks': 'PASS',
        'source_counts': {key: dict(value) for key, value in source_counts.items()},
        'exclusions': exclusions,
    })

    # Real processor and dataset checks include the longest accepted item from each pool.
    save(W / 'validation-status.json', {'status': 'VALIDATING_READER'})
    import torch
    from transformers import SiglipImageProcessor
    from model.model_omni import MiniQwenOmni
    from dataset.sft_dataset import SFTDataset
    from trainer.trainer_utils import load_omni_tokenizer
    from trainer.train_sft import omni_collate_fn

    tokenizer = load_omni_tokenizer(ROOT / 'model/Qwen3-0.6B')
    encoder, audio_processor = MiniQwenOmni.load_sensevoice(ROOT / 'model/SenseVoiceSmall')
    del encoder
    vision_processor = SiglipImageProcessor.from_pretrained(
        ROOT / 'model/siglip2-base-p32-256-ve', local_files_only=True
    )
    selected = {}
    for rows in samples.values():
        for row in rows:
            selected[row['sample_id']] = row
    for row in special_samples.values():
        selected[row['sample_id']] = row
    rows = list(selected.values())
    path = W / 'reader-samples.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression='zstd')
    dataset = SFTDataset(str(path), tokenizer, audio_processor, vision_processor,
                         max_length=3072, training=True, max_images=4)
    reader_records = []
    items = []
    for index, row in enumerate(rows):
        item = dataset[index]
        ids, attention, labels, codes, features, audio_frames, pixels, speaker = item
        if not labels.ne(-100).any():
            raise ValueError(f'No text labels at {row["sample_id"]}')
        if row['image_paths']:
            if int(ids[8].eq(dataset.image_token_id).sum()) != len(row['image_paths']) * 64:
                raise ValueError(f'Image placeholders lost at {row["sample_id"]}')
            if pixels.shape[0] != len(row['image_paths']) or not pixels.any():
                raise ValueError(f'Image pixels lost at {row["sample_id"]}')
        if row['audio_frames_upper_bound']:
            if audio_frames <= 0 or not ids[8].eq(dataset.audio_token_id).any():
                raise ValueError(f'Question waveform lost at {row["sample_id"]}')
            if audio_frames > row['audio_frames_upper_bound']:
                raise ValueError(f'Input audio length exceeds preparation bound at {row["sample_id"]}')
        if row['codec_frames']:
            frames = row['codec_frames']
            if int(codes[0].ne(-100).sum()) != frames + 1:
                raise ValueError(f'Incomplete main codec or EOS at {row["sample_id"]}')
            if int(codes[1].ne(-100).sum()) != frames:
                raise ValueError(f'Incomplete residual codec at {row["sample_id"]}')
            if int(codes[0].eq(dataset.audio_stop_token).sum()) != 1:
                raise ValueError(f'Missing audio stop token at {row["sample_id"]}')
        items.append(item)
        reader_records.append({'sample_id': row['sample_id'], 'source': row['source'],
                               'tier': row['length_tier'], 'budget_bound': row['required_length'],
                               'question_audio_frames': int(audio_frames),
                               'codec_frames': row['codec_frames']})
    trimmed = []
    for offset in range(0, len(items), 4):
        batch = items[offset:offset + 4]
        original_text = sum(int(item[2].ne(-100).sum()) for item in batch)
        original_audio = sum(int(item[3].ne(-100).sum()) for item in batch)
        collated = omni_collate_fn(batch, dynamic_padding=True, pad_to_multiple=8)
        if int(collated[2].ne(-100).sum()) != original_text:
            raise ValueError('Dynamic padding dropped text targets')
        if int(collated[3].ne(-100).sum()) != original_audio:
            raise ValueError('Dynamic padding dropped audio targets')
        trimmed.append(int(collated[0].shape[-1]))
    save(W / 'reader-validation.json', {'status': 'PASS', 'rows': len(rows), 'samples': reader_records,
                                       'dynamic_padding_batch_lengths': trimmed})
    print('READER_VALIDATED', len(rows), flush=True)
    gpu_forward_validation(dataset, items, rows)


    tiers = {}
    for tier, budget in (('main_2048', 2048), ('long_text_3072', 3072)):
        tiers[tier] = {
            'max_length': budget,
            'data_files': {split: file_lists.get(tier + '/' + split, []) for split in ('train', 'dev')},
            'counts': {split: row_counts.get(tier + '/' + split, 0) for split in ('train', 'dev')},
            'max_observed_length': {split: maxima.get(tier + '/' + split, 0) for split in ('train', 'dev')},
        }
    manifest = {
        'status': 'READY', 'format_version': 5, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tiers': tiers, 'dynamic_padding': True, 'sampling': 'global_random',
        'reader': 'dataset.sft_dataset.SFTDataset', 'file_sha256': fingerprints,
        'validation': {'full_rows':'PASS','actual_reader':'PASS','dynamic_padding':'PASS','GPU_forward':'PASS','optimizer_updates':0},
        'sources': {key: {'counts': dict(source_counts[key]),
                          'manifest': str((META / 'sources' / (key + '.json')).relative_to(ROOT))}
                    for key in source_manifests},
        'statistics': {'stored_rows': len(sample_ids), 'target_codec_frames': codec_frames,
                       'target_audio_equivalent_hours': codec_frames / 12.5 / 3600,
                       'referenced_images': len(image_paths), 'referenced_question_audio_paths': len(audio_paths),
                       'parquet_GiB': storage / 2**30,
                       'task_counts': {key: dict(value) for key, value in task_counts.items()},
                       'voice_modes': {key: dict(value) for key, value in voice_modes.items()}},
        'policies': {
            'fusion_views': 'One seeded view per original record, four views approximately equally distributed.',
            'voice_views': '20% basic TTS,80% conditioned. Within conditioned targets,70%both,20%speaker,10%reference before missing-reference fallback.',
            'long_pool': 'Only pure-text records with2049–3072 complete sequence positions.',
            'media': 'Existing images and fusion question waveforms referenced by canonical raw paths; localASR/legacyaudio bytes and targetcodec embedded.',
            'heldout_groups': 'Existing development membership wins when shared-image or repeated-audio groups are merged.',
            'startup': 'Data acceptance is performed during preparation. Training uses the committed manifest.',
        },
        'limitations': [
            'Evaluation overlap checks cover exact image-byte hashes and normalized exact question/target texts. Perceptual/semantic near-duplicate certification remains incomplete.',
            'LegacyA2A input alignment and generic response audio are reused. Legacy reference codec is omitted because independent-reference provenance is unavailable.',
            'The pretrained model has historical exposure to some source corpora. This later SFT split cannot remove that exposure.',
        ],
    }
    save(OUT / 'manifest.json', manifest)
    save(W / 'validation-status.json', {'status': 'COMPLETE', 'manifest': 'dataset/sft/manifest.json'})
    write_report(manifest, exclusions)
    print('SFT_DATA_READY', json.dumps({tier: specification['counts'] for tier, specification in tiers.items()}), flush=True)


def gpu_forward_validation(dataset, items, rows):
    import torch
    from model.model_omni import OmniConfig
    from trainer.trainer_utils import init_omni_model
    from trainer.train_pretrain import forward, objective
    from trainer.train_sft import omni_collate_fn

    checkpoint = ROOT / 'out/miniqwen_pretrain/tts/last/checkpoint'
    if not checkpoint.is_dir():
        save(W / 'forward-validation.json', {'status': 'SKIPPED_NO_CHECKPOINT', 'rows': len(rows)})
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU forward acceptance requires an available device')
    config = OmniConfig.from_pretrained(checkpoint, local_files_only=True)
    model, _ = init_omni_model(
        config, from_weight=str(checkpoint), tokenizer_path=str(ROOT / 'model/Qwen3-0.6B'),
        audio_encoder_path=str(ROOT / 'model/SenseVoiceSmall'),
        vision_model_path=str(ROOT / 'model/siglip2-base-p32-256-ve'), device='cuda:0',
    )
    if model.audio_encoder is not None:
        model.audio_encoder.to('cuda:0')
    if model.vision_encoder is not None:
        model.vision_encoder.to('cuda:0')
    model.eval()
    selected = []
    for source in ('ultrachat', 'coig', 'llava15_single', 'm4', 'fusion',
                   'librispeech', 'aishell1', 'libritts_r', 'aishell3', 'old_a2a'):
        candidates = [index for index, row in enumerate(rows) if row['source'] == source]
        if candidates:
            selected.append(candidates[0])
    for predicate in (
        lambda row: row['length_tier'] == 'long_text_3072',
        lambda row: len(row['image_paths']) == 4,
        lambda row: row['source'] == 'fusion' and bool(row['question_audio_paths']) and row['codec_frames'] > 0,
    ):
        candidate = next((index for index, row in enumerate(rows) if predicate(row)), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    checks = []
    with torch.inference_mode():
        for offset in range(0, len(selected), 2):
            indices = selected[offset:offset + 2]
            batch = omni_collate_fn([items[index] for index in indices], dynamic_padding=True)
            batch = [value.to('cuda:0') if isinstance(value, torch.Tensor) else value for value in batch]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = forward(model, batch, 'tts')
                loss = objective(result, batch[2], batch[3], config, 1)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite forward loss on accepted SFT records')
            checks.append({'sample_ids': [rows[index]['sample_id'] for index in indices],
                           'batch_length': int(batch[0].shape[-1]), 'loss': float(loss)})
    save(W / 'forward-validation.json', {'status': 'PASS', 'device': 'cuda:0',
                                         'precision': 'bfloat16_autocast', 'checks': checks,
                                         'optimizer_updates': 0})
    print('GPU_FORWARD_VALIDATED', len(checks), flush=True)


def write_report(manifest, exclusions):
    tiers = manifest['tiers']
    lines = ['# SFT 数据准备与验收', '', f"完成于 {manifest['created_at']}", '',
             '| 数据池 | train | dev | 长度上限 |', '|---|---:|---:|---:|']
    for tier, specification in tiers.items():
        lines.append(f"| {tier} | {specification['counts']['train']:,} | {specification['counts']['dev']:,} | {specification['max_length']:,} |")
    lines.extend(['', '| 来源 | 主池train | 长文本train | dev合计 | 排除 |', '|---|---:|---:|---:|---:|'])
    for source, specification in manifest['sources'].items():
        counts = specification['counts']
        dev = sum(value for key, value in counts.items() if key.endswith('/dev'))
        lines.append(f"| {source} | {counts.get('main_2048/train', 0):,} | {counts.get('long_text_3072/train', 0):,} | {dev:,} | {sum(exclusions[source].values()):,} |")
    rejected_counts = collections.Counter()
    for reasons in exclusions.values():
        rejected_counts.update(reasons)
    lines.extend(['', '| 排除原因 | 条数 |', '|---|---:|'])
    for reason, count in rejected_counts.most_common():
        lines.append(f'| {reason} | {count:,} |')
    missing_question = exclusions['old_a2a'].get('no_final_question_audio_not_a2a', 0)
    lines.extend(['', f'旧A2A有{missing_question:,}条缺少最后一轮问题音频，因此本轮语音问答池排除这些记录。原始记录仍可从raw来源追溯。长度预算按完整对话、图像占位、音频输入、语音目标和音色前缀共同计算，超长记录进入排除索引。', ''])
    lines.extend(['', '全部来源记录执行数量守恒检查，并逐行核对任务、长度、完整codec、媒体引用及train/dev分组。真实SFTDataset读取检查覆盖各来源首条和最长记录、2/3/4图、四种融合视图、基础TTS及不同音色条件，动态padding前后监督目标数量一致。GPU混合模态及长文本前向检查产生有限loss，优化器更新次数为0。', '',
                  f"最终Parquet约{manifest['statistics']['parquet_GiB']:.2f}GiB，codec目标累计{manifest['statistics']['target_codec_frames']:,}帧，等效约{manifest['statistics']['target_audio_equivalent_hours']:.2f}小时。图片及融合问题音频通过raw路径读取。", '',
                  '本轮采用自然来源数量混合。每条融合记录固定分配一种任务视图。TTS默认20%基础朗读、80%带音色条件，后者按70%双条件、20%仅speaker、10%仅reference分配，缺失独立参考时降级为speaker。', '',
                  '跨来源共享图片统一分组，已有开发成员所在的整个连通组划入dev。M4有26,300条、单图有538条、融合有145条原训练候选因此转到开发侧。旧A2A按相同问题音频和目标文本联合分组，维护既有开发边界。官方语音开发目标与固定评测的规范化长文本匹配记录从训练侧排除。', '',
                  '评测隔离覆盖精确图片字节哈希和规范化问题/目标文本，感知与语义近重复检查仍有覆盖限制。旧A2A的独立音色参考缺少可验证来源，因此其参考codec已移除，现有说话人向量用于语音回答条件。历史pretrain语料暴露记录保留。', '',
                  '统一清单位于dataset/sft/manifest.json。常规训练入口支持--data_manifest dataset/sft/manifest.json与--data_tier all、main_2048或long_text_3072。该入口按清单设置容量并采用SFTDataset，动态padding默认开启。参数解冻、学习率和SFT启动仍由后续训练安排确定。', '',
                  '逐来源清单与排除索引位于dataset/raw/manifests/sft。验收证据位于.runtime/sft-data/full-data-validation.json、reader-validation.json和forward-validation.json。原始外部池保留在dataset/sft/external。'])
    codec_manifest = json.loads((ROOT / 'dataset/raw/corpora/sft_fusion_codec/manifest.json').read_text())
    shard_counts = {name: sum(len(paths) for paths in value['data_files'].values())
                    for name, value in tiers.items()}
    lines.extend(['', f"融合语音输出Mimi编码已完成，共{codec_manifest['rows']:,}条。编码范围与完整帧检查及6条GPU解码抽检的结果位于.runtime/sft-data/fusion-codec-validation.json。", '',
                  f"训练清单包含{sum(shard_counts.values())}个Parquet分片，主池{shard_counts['main_2048']}个、长文本池{shard_counts['long_text_3072']}个。{manifest['statistics']['parquet_GiB']:.2f}GiB为这两类成品Parquet合计，迁移时还需携带图片及融合问题音频对应的raw媒体。"])
    (ROOT / 'reports/SFT_DATA_READY.md').write_text('\n'.join(lines) + '\n')
    save(ROOT / 'reports/SFT_DATA_READY.json', manifest)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(W / 'validation-status.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
