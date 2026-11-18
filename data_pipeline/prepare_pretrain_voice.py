#!/usr/bin/env python3
"""Build deterministic cross-utterance voice conditions without changing targets."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / os.environ.get('MINIQWEN_PRETRAIN_DATA', 'dataset/pretrain')

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def choose_pools(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row['source'], row['split'], row['speaker_id'])].append(row)
    pools = {}
    for key, group in groups.items():
        candidates = sorted(group, key=lambda r: (r['duration'] < 3, hashlib.sha256(r['sample_id'].encode()).hexdigest()))
        selected, texts = [], set()
        for r in candidates:
            if r['text_normalized'] not in texts:
                selected.append(r); texts.add(r['text_normalized'])
            if len(selected) == 4: break
        pools[key] = selected
    for source in {r['source'] for r in rows}:
        train = {k[2] for k in pools if k[:2] == (source, 'train')}
        dev = {k[2] for k in pools if k[:2] == (source, 'dev')}
        if train & dev: raise ValueError(f'{source}: train/dev speaker overlap')
    return pools


def choose_reference(target, pool):
    candidates = [r for r in pool if r['sample_id'] != target['sample_id'] and r['text_normalized'] != target['text_normalized']]
    if not candidates: return target, True
    value = int(hashlib.sha256(target['sample_id'].encode()).hexdigest()[:8], 16)
    return candidates[value % len(candidates)], False


def crop_reference(row, destination):
    wav, sr = sf.read(ROOT / row['audio_path'], dtype='float32', always_2d=True)
    wav = wav.mean(1)
    size = min(len(wav), 4*sr)
    start = (len(wav)-size)//2
    wav = wav[start:start+size]
    if sr != 24000:
        g = math.gcd(sr, 24000); wav = resample_poly(wav, 24000//g, sr//g).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, wav, 24000, format='FLAC', subtype='PCM_24')
    # Read back the materialized reference: both encoders consume identical audio.
    wav, _ = sf.read(destination, dtype='float32')
    return wav, {'crop_start_seconds': start/sr, 'duration': size/sr,
                 'reference_audio_path': str(destination.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = BASE / 'voice'; out.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((BASE/'manifest.json').read_text())
    if catalog['libritts_quality_policy'] != 'exclude_official': raise ValueError('Expected agreed official-clean view')
    rows, files = [], {}
    for source in ['libritts_r', 'aishell3']:
        files[source] = catalog['sources'][source]['data_files']
        selected = {r['sample_id'] for paths in files[source].values() for p in paths
                    for b in pq.ParquetFile(ROOT/p).iter_batches(columns=['sample_id']) for r in b.to_pylist()}
        for line in (BASE/'processed'/source/'segments.jsonl').open():
            row = json.loads(line)
            if row['sample_id'] in selected: rows.append(row)
    index = {r['sample_id']: r for r in rows}
    pools = choose_pools(rows)
    pairs = {r['sample_id']: choose_reference(r, pools[(r['source'],r['split'],r['speaker_id'])]) for r in rows}
    references = {r['sample_id']: r for r, _ in pairs.values()}
    fingerprint = {'format': 1, 'catalog_sha256': digest(BASE/'manifest.json'),
                   'campplus_sha256': digest(ROOT/'model/campplus/campplus_cn_common.pt'),
                   'mimi_sha256': digest(ROOT/'model/mimi/model.safetensors'),
                   'crop': 'center, max 4 seconds; prefer >=3s; no target cutting',
                   'speaker_normalization': 'layer_norm_192', 'reference_count': len(references)}
    cache_path = out/'reference_cache.json'
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache['fingerprint'] != fingerprint: raise ValueError('Voice cache fingerprint differs')
        conditions = cache['conditions']
    else:
        from funasr.models.campplus.model import CAMPPlus
        from funasr.models.campplus.utils import extract_feature
        from transformers import MimiModel
        camp = CAMPPlus().eval()
        state = torch.load(ROOT/'model/campplus/campplus_cn_common.pt', map_location='cpu', weights_only=True)
        camp.load_state_dict(state, strict=True)
        camp = camp.to(args.device).requires_grad_(False)
        mimi = MimiModel.from_pretrained(ROOT/'model/mimi', local_files_only=True).eval().float().to(args.device).requires_grad_(False)
        mimi.config.use_streaming = False
        conditions, prepared = {}, []
        for i, (sid, row) in enumerate(sorted(references.items())):
            path = out/'references'/(hashlib.sha256(sid.encode()).hexdigest()[:24]+'.flac')
            wav, meta = crop_reference(row, path)
            wav16 = resample_poly(wav, 2, 3).astype(np.float32)
            # Tiny references get encoder-input padding only, recorded explicitly.
            spk_padding = max(0, 16000-len(wav16)); wav16 = np.pad(wav16, (0, spk_padding))
            feature = extract_feature([torch.from_numpy(wav16)])[0][0]
            meta.update(sample_id=sid, speaker_id=row['speaker_id'], split=row['split'],
                        campplus_padding_samples=spk_padding)
            prepared.append((sid, wav, feature, meta))
            if (i+1)%200 == 0: print('VOICE_REFERENCES_DECODED',i+1,'/',len(references),flush=True)
        # CAMPPlus stats pooling has no padding mask: batch only equal fbank lengths.
        groups = defaultdict(list)
        for item in prepared: groups[item[2].shape[0]].append(item)
        with torch.inference_mode():
            for group in groups.values():
                for start in range(0,len(group),16):
                    batch = group[start:start+16]
                    embeddings = camp(torch.stack([b[2] for b in batch]).to(args.device)).float()
                    embeddings = F.layer_norm(embeddings, (192,)).cpu()
                    for item, embedding in zip(batch,embeddings):
                        if not torch.isfinite(embedding).all(): raise ValueError('Nonfinite speaker vector')
                        conditions[item[0]] = {**item[3], 'spk_emb': embedding.tolist()}
            for start in range(0,len(prepared),16):
                batch = prepared[start:start+16]
                waves = [np.pad(b[1], (0,(-len(b[1]))%1920)) for b in batch]
                values = torch.zeros(len(batch),1,max(map(len,waves)),device=args.device)
                for i,w in enumerate(waves): values[i,0,:len(w)] = torch.from_numpy(w).to(args.device)
                codes = mimi.encode(values,num_quantizers=8,use_streaming=False).audio_codes.cpu()
                for i,item in enumerate(batch):
                    n=len(waves[i])//1920
                    conditions[item[0]]['ref_audios']=codes[i,:,:n].T.reshape(-1).tolist()
                print('VOICE_REFERENCES_ENCODED',min(start+16,len(prepared)),'/',len(prepared),flush=True)
        temp=cache_path.with_suffix('.tmp');temp.write_text(json.dumps({'fingerprint':fingerprint,'conditions':conditions}));os.replace(temp,cache_path)
    sources, fallback_ids = {}, []
    extra = [pa.field('spk_emb',pa.list_(pa.float32())),pa.field('ref_audios',pa.list_(pa.int64())),
             pa.field('reference_sample_id',pa.string()),pa.field('reference_speaker_id',pa.string()),
             pa.field('reference_split',pa.string()),pa.field('reference_duration',pa.float64()),
             pa.field('voice_self_fallback',pa.bool_())]
    for source,splits in files.items():
        output_files,counts={},{}
        for split,paths in splits.items():
            output_files[split]=[];counts[split]=0
            for i,path in enumerate(paths):
                target=out/source/f'{split}-{i:02d}.parquet';target.parent.mkdir(parents=True,exist_ok=True)
                pf=pq.ParquetFile(ROOT/path);schema=pa.schema(list(pf.schema_arrow)+extra)
                temp=target.with_suffix('.partial')
                with pq.ParquetWriter(temp,schema,compression='zstd') as writer:
                    for batch in pf.iter_batches(batch_size=128):
                        converted=[]
                        for row in batch.to_pylist():
                            ref,fallback=pairs[row['sample_id']];cond=conditions[ref['sample_id']]
                            if not fallback:
                                assert ref['speaker_id']==row['speaker_id'] and ref['split']==row['split']
                                assert ref['sample_id']!=row['sample_id'] and ref['text_normalized']!=index[row['sample_id']]['text_normalized']
                            else: fallback_ids.append(row['sample_id'])
                            row.update(spk_emb=cond['spk_emb'],ref_audios=[] if fallback else cond['ref_audios'],
                                       reference_sample_id=ref['sample_id'],reference_speaker_id=ref['speaker_id'],
                                       reference_split=ref['split'],reference_duration=cond['duration'],voice_self_fallback=fallback)
                            converted.append(row)
                        counts[split]+=len(converted);writer.write_table(pa.Table.from_pylist(converted,schema=schema))
                os.replace(temp,target);output_files[split].append(str(target.relative_to(ROOT)))
        sources[source]={'data_files':output_files,'counts':counts,
                         'sha256':{p:digest(ROOT/p) for paths in output_files.values() for p in paths}}
    manifest={'fingerprint':fingerprint,'sources':sources,'fallback_sample_ids':fallback_ids,
              'policy':'same speaker/split, different utterance AND text; fallback speaker-only; exclude fallback from cloning metrics',
              'mixture':{'both':0.7,'speaker':0.2,'reference':0.1},'prefix_tokens':160}
    temp=out/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2));os.replace(temp,out/'manifest.json')
    print('VOICE_DATA_READY', {s:d['counts'] for s,d in sources.items()},'fallbacks',len(fallback_ids),flush=True)


if __name__=='__main__': main()
