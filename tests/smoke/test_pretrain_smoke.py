#!/usr/bin/env python3
"""Reproducible bounded four-card tests; no formal training outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='.runtime/pretrain/acceptance')
    parser.add_argument('--plan',default='dataset/raw/manifests/pretrain/plan.json')
    parser.add_argument('--keep-batch',action='store_true')
    parser.add_argument('--train-samples',type=int,default=64)
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'.runtime') or out.exists():raise ValueError('Use a new directory below .runtime')
    out.mkdir(parents=True)
    plan=json.loads((ROOT/args.plan).read_text())
    for phase in plan['phases']:
        phase.update(eval_samples_per_source=1)
        if not args.keep_batch:phase.update(batch_size=8,accumulation_steps=1)
        for name,source in phase['sources'].items():
            for split,files in source.items():
                batch=next(pq.ParquetFile(ROOT/files[0]).iter_batches(batch_size=args.train_samples if split=='train' else 2))
                path=out/f'{name}-{split}.parquet';pq.write_table(pa.Table.from_batches([batch]),path)
                source[split]=[str(path.relative_to(ROOT))]
    if not args.keep_batch:plan['global_batch']=32
    path=out/'plan.json';path.write_text(json.dumps(plan,indent=2))
    env={**os.environ,'OMP_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    previous='qwen'
    for phase in plan['phases']:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'trainer/train_pretrain.py'),'--plan',str(path),'--phase',phase['name'],'--from-weight',previous,'--output',str(out/phase['name']),'--smoke','--max-steps','1','--num-workers','0']
        with open(out/f"{phase['name']}.log",'w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if phase['name']=='tts':
            cmd[cmd.index('--max-steps')+1]='2';cmd.append('--resume')
            with open(out/'tts-resume.log','w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        previous=str(out/phase['name']/'last/checkpoint')
        print('SMOKE_PASS',phase['name'],flush=True)
if __name__=='__main__':main()
