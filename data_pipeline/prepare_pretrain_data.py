#!/usr/bin/env python3
"""Download pinned public corpora and build lossless pretrain manifests.

Run with the project Python. Downloads, caches and generated data stay under
dataset/pretrain. Does not install packages or touch training checkpoints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import zipfile
from urllib.parse import urlencode

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / os.environ.get('MINIQWEN_PRETRAIN_DATA', 'dataset/pretrain')
LIBRI_SUBSET = os.environ.get('MINIQWEN_LIBRI_SUBSET', 'clean100')
if LIBRI_SUBSET not in {'clean100', 'clean360'}:
    raise ValueError('MINIQWEN_LIBRI_SUBSET must be clean100 or clean360')
REVISIONS = {
    'openslr/librispeech_asr': '71cacbfb7e2354c4226d01e70d77d5fca3d04ba1',
    'liuhaotian/LLaVA-Pretrain': '70f9d1e5e1a697fe35830875cfc7de1dd590d727',
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def digest_file(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def verify_archive_md5(path, expected):
    """Reuse a verified immutable archive; any size/mtime change rechecks it."""
    receipt_path=path.with_name(path.name+'.download.json')
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    info=path.stat();identity={'size':info.st_size,'mtime_ns':info.st_mtime_ns}
    cached=(receipt.get('official_md5_verified') and receipt.get('official_md5')==expected
            and receipt.get('official_md5_file_stat')==identity)
    if not cached and digest_file(path,'md5')!=expected:
        raise RuntimeError(f'Official MD5 mismatch: {path}')
    if receipt_path.exists() and not cached:
        receipt.update(official_md5=expected,official_md5_verified=True,official_md5_file_stat=identity)
        write_json(receipt_path,receipt)
    print('[official MD5 verified'+(' cached' if cached else '')+']',path.name,flush=True)


def get_json(url):
    r = requests.get(url, timeout=(20, 120))
    r.raise_for_status()
    return r.json()


def domestic_files(source, files):
    """Keep upstream identity; use a domestic transport only for matching files."""
    if source == 'libritts_r':
        for entry in files:
            entry.update(download_url=entry['url'].replace('www.openslr.org', 'openslr.magicdatatech.com'),
                         direct_domestic=True, mirror_repo='OpenSLR official CN mirror')
        return files
    mirrors = {
        'librispeech': ('openslr/librispeech_asr', 62471),
        'llava': ('AI-ModelScope/LLaVA-Pretrain', 19091),
        'aishell3': ('OmniData/AISHELL-3', 25141),
        'aishell1': ('OmniData/AISHELL-1', 25140),
    }
    if source not in mirrors:
        return files
    repo, dataset_id = mirrors[source]
    session = requests.Session()
    session.trust_env = False  # Domestic OSS access does not need the external proxy.
    r = session.get(f'https://modelscope.cn/api/v1/datasets/{dataset_id}/repo/tree',
                    params={'Revision': 'master', 'Root': '/', 'Recursive': 'True',
                            'PageNumber': 1, 'PageSize': 1000}, timeout=(15, 60))
    r.raise_for_status()
    tree = {f['Path']: f for f in r.json()['Data']['Files'] if f['Type'] == 'blob'}
    for entry in files:
        mirror_path = ('raw/33/' + entry['path'] if source == 'aishell1' else
                       'raw/' + entry['path'] if source == 'aishell3' else entry['path'])
        item = tree.get(mirror_path)
        if not item or not item.get('Sha256'):
            continue
        if entry.get('size') and item['Size'] != entry['size']:
            continue
        if entry.get('sha256') and item['Sha256'] != entry['sha256']:
            continue
        if not entry.get('sha256') and source not in ('aishell1', 'aishell3'):
            continue
        entry.update(size=item['Size'], sha256=item['Sha256'], mirror_repo=repo,
                     mirror_revision=item['Revision'], direct_domestic=True,
                     download_url=f'https://modelscope.cn/api/v1/datasets/{repo}/repo?' +
                     urlencode({'Revision': item['Revision'], 'FilePath': mirror_path}))
    return files


def source_files(source):
    if source == 'librispeech':
        repo = 'openslr/librispeech_asr'
        prefix = ('clean/train.360/',) if LIBRI_SUBSET == 'clean360' else ('clean/train.100/', 'clean/validation/')
    elif source == 'llava':
        repo = 'liuhaotian/LLaVA-Pretrain'
        prefix = ('README.md', 'blip_laion_cc_sbu_558k.json', 'images.zip')
    else:
        official = {
            'aishell1': (33, ['data_aishell.tgz']),
            'libritts_r': (141, ['train_clean_100.tar.gz', 'dev_clean.tar.gz', 'doc.tar.gz',
                                'md5sum.txt', 'libritts_r_failed_speech_restoration_examples.tar.gz']),
            'aishell3': (93, ['data_aishell3.tgz']),
        }
        number, names = official[source]
        if source == 'libritts_r' and LIBRI_SUBSET == 'clean360':
            names = ['train_clean_360.tar.gz', 'md5sum.txt']
        return [{'path': name, 'url': f'https://www.openslr.org/resources/{number}/{name}',
                 'source_page': f'https://www.openslr.org/{number}/'} for name in names]
    rev = REVISIONS[repo]
    tree = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=true&limit=1000')
    files = []
    for f in tree:
        if f['type'] != 'file' or not any(f['path'].startswith(p) for p in prefix):
            continue
        files.append({'path': f['path'], 'size': f['size'],
                      'sha256': f.get('lfs', {}).get('oid'), 'revision': rev,
                      'url': f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{f["path"]}',
                      'source_page': f'https://huggingface.co/datasets/{repo}'})
    if not files:
        raise RuntimeError(f'No files for {source}')
    return files


def download(entry, source, workers):
    dest = DATA / 'raw' / source / entry['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_name(dest.name + '.download.json')
    if dest.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('size') == dest.stat().st_size and previous.get('url') == entry['url']:
            return
    url = entry.get('download_url', entry['url'])
    def fetch(**kwargs):
        kwargs['headers'] = {'Accept-Encoding': 'identity', **kwargs.get('headers', {})}
        if entry.get('direct_domestic'):
            s = requests.Session()
            s.trust_env = False
            return s.get(url, **kwargs)
        return requests.get(url, **kwargs)
    for attempt in range(6):
        try:
            with fetch(headers={'Range': 'bytes=0-0'}, stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                ranged = r.status_code == 206
                content_range = r.headers.get('Content-Range', '')
                size = int(content_range.rsplit('/', 1)[-1]) if ranged else int(r.headers.get('Content-Length', 0))
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            print(f'[probe retry] {source}/{entry["path"]} attempt={attempt+1}', flush=True)
            time.sleep(min(2 ** attempt, 20))
    if entry.get('size') and size != entry['size']:
        raise RuntimeError(f'Size mismatch for {url}: {size} != {entry["size"]}')
    parts = dest.with_name(dest.name + '.parts')
    parts.mkdir(exist_ok=True)
    chunk_size = 32 * 1024 * 1024
    count = math.ceil(size / chunk_size) if ranged and size else 1
    completed = [0]
    lock = threading.Lock()
    started = time.monotonic()
    last_log = [started]

    def part(index):
        path = parts / f'{index:06d}'
        start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        expected = end - start + 1 if ranged else size
        if not path.exists() or (expected and path.stat().st_size != expected):
            for attempt in range(6):
                try:
                    tmp = path.with_suffix('.tmp')
                    offset = tmp.stat().st_size if ranged and tmp.exists() else 0
                    if expected and offset == expected:
                        os.replace(tmp, path)
                        break
                    if expected and offset > expected:
                        raise RuntimeError('Existing partial part exceeds expected size')
                    request_start = start + offset
                    headers = {'Range': f'bytes={request_start}-{end}'} if ranged else {}
                    request_started = time.monotonic()
                    with fetch(headers=headers, stream=True, timeout=(30, 120)) as rr:
                        rr.raise_for_status()
                        if ranged and (rr.status_code != 206 or not rr.headers.get('Content-Range', '').startswith(f'bytes {request_start}-{end}/')):
                            raise RuntimeError('Server ignored or changed requested byte range')
                        with open(tmp, 'ab' if offset else 'wb') as out:
                            for b in rr.iter_content(1024 * 1024):
                                out.write(b)
                                if ranged and time.monotonic() - request_started > 90 and out.tell() < expected:
                                    raise TimeoutError('Slow range transfer; reconnect and resume its partial bytes')
                        if expected and tmp.stat().st_size != expected:
                            raise RuntimeError('Incomplete download part')
                        os.replace(tmp, path)
                    break
                except Exception as exc:
                    print(f'[retry] {source}/{entry["path"]} part={index} attempt={attempt+1}: {type(exc).__name__}', flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(min(2 ** attempt, 20))
        with lock:
            completed[0] += path.stat().st_size
            if time.monotonic() - last_log[0] >= 20 or completed[0] == size:
                elapsed = max(0.01, time.monotonic() - started)
                print(f'[download] {source}/{entry["path"]}: {completed[0]/1e9:.2f}/{size/1e9:.2f} GB, {completed[0]/1e6/elapsed:.1f} MB/s', flush=True)
                last_log[0] = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers if ranged else 1) as pool:
        list(pool.map(part, range(count)))
    temp = dest.with_name(dest.name + '.partial')
    h = hashlib.sha256()
    with open(temp, 'wb') as out:
        for i in range(count):
            with open(parts / f'{i:06d}', 'rb') as f:
                for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(b)
                    out.write(b)
    sha = h.hexdigest()
    if entry.get('sha256') and sha != entry['sha256']:
        raise RuntimeError(f'SHA256 mismatch: {dest}')
    os.replace(temp, dest)
    write_json(receipt, {**entry, 'size': dest.stat().st_size, 'sha256': sha,
                         'upstream_sha256_verified': bool(entry.get('sha256'))})
    shutil.rmtree(parts)
    print(f'[verified] {source}/{entry["path"]}', flush=True)


def extract_archive(path, target):
    """Stream gzip once, write independent files concurrently on network storage."""
    base = target.resolve()
    checked = set()
    lock = threading.Lock()
    def write_member(name, data):
        destination = base / name
        parent = destination.parent
        with lock:
            if parent not in checked:
                parent.mkdir(parents=True, exist_ok=True)
                if parent.resolve() != parent or not parent.is_relative_to(base):
                    raise ValueError(f'Unsafe existing archive directory: {parent}')
                checked.add(parent)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
    tasks = set()
    count = 0
    # The tarfile stream default is 10 KiB, causing millions of tiny reads on
    # the shared filesystem for a 29 GB archive. Keep sequential reads large.
    with tarfile.open(path, 'r|gz', bufsize=4*1024*1024) as archive, ThreadPoolExecutor(max_workers=16) as pool:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk() or member.isdev():
                raise ValueError(f'Unsafe archive member: {member.name}')
            if not member.isfile():
                continue
            data = archive.extractfile(member).read()
            tasks.add(pool.submit(write_member, str(name), data))
            count += 1
            if len(tasks) >= 32:
                completed, tasks = wait(tasks, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
            if count % 10000 == 0:
                print('[extract progress]', path.name, count, flush=True)
        for future in tasks:
            future.result()


def extract(source):
    raw = DATA / 'raw' / source
    target = DATA / 'media' / source
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw.glob('*.tar.gz')) + sorted(raw.glob('*.tgz')):
        marker = target / (path.name + '.extracted.json')
        if marker.exists():
            continue
        print('[extract]', path, flush=True)
        extract_archive(path, target)
        write_json(marker, {'archive': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size})
    return target


def audio_info(raw):
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    import numpy as np
    if not len(wav) or not np.isfinite(wav).all():
        raise ValueError('Empty or nonfinite waveform')
    return {'sample_rate': sr, 'samples': len(wav), 'channels': wav.shape[1],
            'duration': len(wav) / sr}


def write_asr(source):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = DATA / 'processed' / source
    out.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([('sample_id', pa.string()), ('source', pa.string()), ('split', pa.string()),
                        ('task_type', pa.string()), ('language', pa.string()), ('speaker_id', pa.string()),
                        ('text_original', pa.string()), ('text_target', pa.string()), ('duration', pa.float64()),
                        ('sample_rate', pa.int32()), ('conversations', pa.string()),
                        ('question_audios', pa.list_(pa.binary()))])
    stats, errors, writers, pending = {}, [], {}, {}

    def accept(sid, split, speaker, original, text, audio, language):
        try:
            info = audio_info(audio)
            instruction = 'Transcribe the speech verbatim.' if language == 'en' else '请逐字转写这段语音。'
            row = dict(sample_id=f'{source}:{sid}', source=source, split=split, task_type='asr',
                       language=language, speaker_id=str(speaker), text_original=original, text_target=text,
                       duration=info['duration'], sample_rate=info['sample_rate'],
                       conversations=json.dumps([{'role': 'user', 'content': instruction},
                                                 {'role': 'assistant', 'content': text}], ensure_ascii=False),
                       question_audios=[audio])
            if not text:
                raise ValueError('Empty target')
        except Exception as exc:
            errors.append({'sample_id': sid, 'split': split, 'error': repr(exc)})
            return
        if split not in writers:
            writers[split] = pq.ParquetWriter(out / (split + '.parquet.partial'), schema, compression='zstd')
            pending[split] = []
            stats[split] = {'rows': 0, 'seconds': 0.0, 'max_duration': 0.0, 'speakers': set()}
        pending[split].append(row)
        st = stats[split]
        st['rows'] += 1
        st['seconds'] += info['duration']
        st['max_duration'] = max(st['max_duration'], info['duration'])
        st['speakers'].add(str(speaker))
        if len(pending[split]) >= 128:
            writers[split].write_table(pa.Table.from_pylist(pending[split], schema=schema))
            pending[split].clear()
        if st['rows'] % 10000 == 0:
            print('[convert]', source, split, st['rows'], flush=True)

    if source == 'librispeech':
        splits = [('train', 'train.360')] if LIBRI_SUBSET == 'clean360' else [('train', 'train.100'), ('dev', 'validation')]
        for split, folder in splits:
            for path in sorted((DATA / 'raw' / source / 'clean' / folder).glob('*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        accept(row['id'], split, row['speaker_id'], row['text'], row['text'], row['audio']['bytes'], 'en')
    else:
        media = extract(source)
        transcripts = next(media.rglob('aishell_transcript_v0.8.txt'))
        mapping = {}
        for line in transcripts.read_text().splitlines():
            if line.strip():
                sid, text = line.split(maxsplit=1)
                mapping[sid] = text
        # The official archive contains train/dev/test: retain test as raw only.
        # Read the official per-speaker archives directly, avoiding 178k tiny
        # extracted WAV files and their network-filesystem metadata overhead.
        for path in sorted(media.rglob('*.tar.gz')):
            with tarfile.open(path, 'r|gz') as archive:
                for member in archive:
                    wav = PurePosixPath(member.name)
                    split = 'train' if 'train' in wav.parts else 'dev' if 'dev' in wav.parts else None
                    if split and member.isfile() and wav.suffix == '.wav':
                        original = mapping.get(wav.stem, '')
                        accept(wav.stem, split, wav.parent.name, original, ''.join(original.split()), archive.extractfile(member).read(), 'zh')
    for split, writer in writers.items():
        if pending[split]:
            writer.write_table(pa.Table.from_pylist(pending[split], schema=schema))
        writer.close()
        os.replace(out / (split + '.parquet.partial'), out / (split + '.parquet'))
    expected_splits = ('train',) if source == 'librispeech' and LIBRI_SUBSET == 'clean360' else ('train', 'dev')
    if not stats or not all(s in stats for s in expected_splits):
        raise RuntimeError(f'Missing train/dev data for {source}')
    train_speakers, dev_speakers = stats['train']['speakers'], stats.get('dev', {}).get('speakers', set())
    if train_speakers & dev_speakers:
        raise RuntimeError('ASR train/dev speaker overlap')
    for split, st in stats.items():
        st['speakers'] = sorted(st['speakers'])
        st['parquet_sha256'] = digest_file(out / (split + '.parquet'))
    write_json(out / 'quarantine.json', errors)
    write_json(out / 'manifest.json', {'source': source, 'status': 'training_format_ready',
                                     'splits': stats, 'quarantined': len(errors), 'task_aware_loader_required': True,
                                     'policy': 'original segments; no case rewrite/punctuation/cuts/duration filtering'})


def write_llava():
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw = DATA / 'raw/llava'
    out = DATA / 'processed/llava'
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((raw / 'blip_laion_cc_sbu_558k.json').read_text())
    schema = pa.schema([('sample_id', pa.string()), ('source', pa.string()), ('split', pa.string()),
                        ('task_type', pa.string()), ('image_id', pa.string()), ('conversations', pa.string()),
                        ('image_bytes', pa.binary())])
    writers = {s: pq.ParquetWriter(out / (s + '.parquet.partial'), schema, compression='zstd') for s in ('train', 'dev')}
    pending = {s: [] for s in writers}
    counts = {s: 0 for s in writers}
    errors, seen, duplicates = [], set(), 0
    archives = []
    thread_state = threading.local()
    def init_archive():
        # A shared ZipFile serializes seeks/reads under its own lock. Separate
        # handles let network-filesystem latency overlap across image workers.
        thread_state.archive = zipfile.ZipFile(raw / 'images.zip')
        archives.append(thread_state.archive)
    with zipfile.ZipFile(raw / 'images.zip') as z, ThreadPoolExecutor(max_workers=8, initializer=init_archive) as pool:
        names = set(z.namelist())
        def convert(row):
            key = row['image']
            try:
                name = key if key in names else 'images/' + key
                data = thread_state.archive.read(name)
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                messages = [{'role': {'human': 'user', 'gpt': 'assistant'}[t['from']], 'content': t['value']}
                            for t in row['conversations']]
                if sum(t['content'].count('<image>') for t in messages) != 1:
                    raise ValueError('Expected one image marker')
                # Exact byte duplicates share a split even under different paths.
                split = 'dev' if int(hashlib.sha256(data).hexdigest()[:8], 16) % 100 == 0 else 'train'
                return dict(sample_id='llava:' + str(row['id']), source='llava', split=split,
                            task_type='i2t', image_id=key, conversations=json.dumps(messages, ensure_ascii=False), image_bytes=data), None
            except Exception as exc:
                return None, {'sample_id': row.get('id'), 'image': key, 'error': repr(exc)}
        processed = 0
        for start in range(0, len(records), 512):
            batch = records[start:start + 512]
            for row, (converted, error) in zip(batch, pool.map(convert, batch)):
                key = row['image']
                if key in seen:
                    duplicates += 1
                seen.add(key)
                if error:
                    errors.append(error)
                    continue
                split = converted['split']
                pending[split].append(converted)
                counts[split] += 1
                if len(pending[split]) >= 128:
                    writers[split].write_table(pa.Table.from_pylist(pending[split], schema=schema))
                    pending[split].clear()
            if start + len(batch) - processed >= 20000 or start + len(batch) == len(records):
                processed = start + len(batch)
                print('[convert] llava', processed, '/', len(records), flush=True)
    for archive in archives:
        archive.close()
    for s, w in writers.items():
        if pending[s]:
            w.write_table(pa.Table.from_pylist(pending[s], schema=schema))
        w.close()
        os.replace(out / (s + '.parquet.partial'), out / (s + '.parquet'))
    write_json(out / 'quarantine.json', errors)
    write_json(out / 'manifest.json', {'source': 'llava', 'status': 'training_format_ready',
                                     'revision': REVISIONS['liuhaotian/LLaVA-Pretrain'], 'source_rows': len(records),
                                     'splits': counts, 'duplicate_image_rows': duplicates, 'quarantined': len(errors),
                                     'split_policy': 'sha256(image bytes) mod 100 == 0 => dev; repeated-image captions retained; no caption rewrite',
                                     'sha256': {s: digest_file(out / (s + '.parquet')) for s in writers}})


def write_tts(source):
    media = extract(source)
    out = DATA / 'processed' / source
    out.mkdir(parents=True, exist_ok=True)
    mapping = {}
    if source == 'aishell3':
        for txt in media.rglob('content.txt'):
            for line in txt.read_text().splitlines():
                if not line.strip():
                    continue
                sid, body = line.split(maxsplit=1)
                fields = body.split()
                mapping[Path(sid).stem] = (''.join(fields[0::2]), line)
    stats = {}
    errors = []
    def convert(wav):
        if source == 'libritts_r':
            if LIBRI_SUBSET == 'clean360':
                split = 'train' if 'train-clean-360' in wav.parts else None
            else:
                split = 'train' if 'train-clean-100' in wav.parts else 'dev' if 'dev-clean' in wav.parts else None
        else:
            split = 'train' if 'train' in wav.parts else None
        if not split:
            return None, None
        try:
            if source == 'libritts_r':
                original = wav.with_suffix('.original.txt').read_text().strip()
                normalized = wav.with_suffix('.normalized.txt').read_text().strip()
                transcript = original
            else:
                normalized, transcript = mapping[wav.stem]
                original = normalized
            info = audio_info(wav.read_bytes())
            if not normalized:
                raise ValueError('Empty TTS target')
            speaker = wav.stem.split('_')[0] if source == 'libritts_r' else wav.stem[:7]
            if source == 'aishell3' and int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 20 == 0:
                split = 'dev'
            row = dict(sample_id=f'{source}:{wav.stem}', source=source, split=split, task_type='tts',
                       speaker_id=speaker,
                       language='en' if source == 'libritts_r' else 'zh', audio_path=str(wav.relative_to(ROOT)),
                       text_original=original, text_normalized=normalized, transcript_original=transcript, **info)
            return row, None
        except Exception as exc:
            return None, {'sample_id': wav.stem, 'error': repr(exc)}
    wavefiles = sorted(media.rglob('*.wav'))
    processed = 0
    with open(out / 'segments.jsonl.partial', 'w') as f, ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(wavefiles), 256):
            for row, error in pool.map(convert, wavefiles[start:start + 256]):
                if error:
                    errors.append(error)
                    continue
                if row is None:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                split = row['split']
                st = stats.setdefault(split, {'rows': 0, 'seconds': 0.0, 'max_duration': 0.0})
                st['rows'] += 1
                st['seconds'] += row['duration']
                st['max_duration'] = max(st['max_duration'], row['duration'])
            if start + 256 - processed >= 10000:
                processed = start + 256
                print('[scan TTS]', source, min(processed, len(wavefiles)), '/', len(wavefiles), flush=True)
    if not stats:
        raise RuntimeError(f'No TTS segments for {source}')
    os.replace(out / 'segments.jsonl.partial', out / 'segments.jsonl')
    write_json(out / 'quarantine.json', errors)
    write_json(out / 'manifest.json', {'source': source, 'status': 'raw_segments_ready_codec_pending',
                                     'splits': stats, 'quarantined': len(errors), 'speaker_reference': False,
                                     'split_policy': 'official train/dev' if source == 'libritts_r' else 'official train speakers: sha256(speaker) mod 20 == 0 => dev; official test excluded',
                                     'policy': 'official text and segments preserved; Mimi encoding pending'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', choices=['librispeech', 'aishell1', 'llava', 'libritts_r', 'aishell3'], required=True)
    p.add_argument('--stage', choices=['download', 'convert', 'all'], default='all')
    p.add_argument('--download-workers', type=int, default=8)
    p.add_argument('--file-workers', type=int, default=1)
    p.add_argument('--domestic', action='store_true', help='Use matching ModelScope transport; retain upstream hashes')
    args = p.parse_args()
    for source in args.sources:
        if args.stage in ('download', 'all'):
            cached = DATA / 'raw' / source / 'source_manifest.json'
            files = json.loads(cached.read_text())['files'] if cached.exists() else source_files(source)
            if args.domestic:
                files = domestic_files(source, files)
            write_json(DATA / 'raw' / source / 'source_manifest.json', {'source': source, 'files': files})
            with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
                list(pool.map(lambda entry: download(entry, source, args.download_workers), files))
        if args.stage in ('convert', 'all'):
            if source == 'libritts_r':
                checksums = DATA / 'raw' / source / 'md5sum.txt'
                for line in checksums.read_text().splitlines():
                    md5, name = line.split(maxsplit=1)
                    path = checksums.parent / name.lstrip('*')
                    if path.exists():
                        verify_archive_md5(path,md5)
            if source in ('librispeech', 'aishell1'):
                write_asr(source)
            elif source == 'llava':
                write_llava()
            else:
                write_tts(source)
        print('[complete]', source, args.stage, flush=True)


if __name__ == '__main__':
    main()
