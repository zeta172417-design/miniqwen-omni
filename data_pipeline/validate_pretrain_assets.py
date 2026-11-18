#!/usr/bin/env python3
"""Explicit data-preparation acceptance; never called by training startup."""
import hashlib,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def preflight(plan):
    base=ROOT/plan.get('data_root','dataset/pretrain')
    if (base/'RETIRED_AFTER_CLEANUP.json').exists():
        raise ValueError('This dataset view was retired by cleanup; use dataset/pretrain_v2/pretrain_plan.json')
    if not plan.get('asset_sha256'):raise ValueError('Model asset fingerprints missing; rebuild plan')
    for path,expected in {**plan['asset_sha256'],**plan.get('code_sha256',{})}.items():
        if sha(ROOT/path)!=expected:raise ValueError(f'Model asset changed: {path}')
    for key,path in [('data_catalog_sha256',base/'manifest.json'),('voice_manifest_sha256',base/'voice/manifest.json')]:
        if sha(path)!=plan[key]:raise ValueError(f'Changed manifest: {path}; rebuild and review plan')
    import pyarrow.parquet as pq
    catalog=json.loads((base/'manifest.json').read_text())
    for phase in plan['phases']:
        if type(phase['epochs']) is not int or phase['epochs']<1:raise ValueError('Expected positive integer epochs')
        if phase['batch_size']*phase['accumulation_steps']*plan['world_size']!=plan['global_batch']:raise ValueError('Batch mismatch')
        counts={}
        for name,src in phase['sources'].items():
            counts[name]={split:sum(pq.ParquetFile(ROOT/f).metadata.num_rows for f in files) for split,files in src.items()}
            for split,count in counts[name].items():
                if count!=catalog['sources'][name]['splits'][split]['rows']:raise ValueError(f'Row count mismatch: {name}/{split}')
            if phase['name']!='tts':
                source=json.loads((ROOT/catalog['sources'][name]['source_manifest']).read_text())
                for split,files in src.items():
                    for file in files:
                        expected=catalog['sources'][name].get('file_sha256',{}).get(file) or source.get('file_sha256',{}).get(file)
                        if expected is None:
                            expected=source['sha256'][split] if name=='llava' else source['splits'][split]['parquet_sha256']
                        if sha(ROOT/file)!=expected:raise ValueError(f'Data changed: {file}')
        print(phase['name'],json.dumps(counts),'max_length',phase['max_seq_len'],'modules',phase['modules'],flush=True)
    # Verify every voice shard against the recorded encoding/pairing output.
    voice=json.loads((base/'voice/manifest.json').read_text())
    for src in voice['sources'].values():
        for path,expected in src['sha256'].items():
            if sha(ROOT/path)!=expected:raise ValueError(f'Voice shard changed: {path}')
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='dataset/raw/manifests/pretrain/plan.json');args=parser.parse_args()
    preflight(json.loads(Path(args.plan).read_text()))
    print('PREFLIGHT_PASS. No training started. Data preparation acceptance only.')
