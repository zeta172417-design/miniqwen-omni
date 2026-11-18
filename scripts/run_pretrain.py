#!/usr/bin/env python3
"""Show configuration by default. Formal execution requires explicit --run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trainer.pretrain_checkpoint import recover_checkpoint
from trainer.pretrain_plan import validate_checkpoint_signature
from trainer.runtime_env import training_environment

def validate_config(plan):
    for phase in plan['phases']:
        if type(phase['epochs']) is not int or phase['epochs']<1:raise ValueError('Expected positive integer epochs')
        if phase['batch_size']*phase['accumulation_steps']*plan['world_size']!=plan['global_batch']:raise ValueError('Batch mismatch')
        print('PHASE_CONFIG',phase['name'],'epochs',phase['epochs'],'modules',phase['modules'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default=str(ROOT/'dataset/raw/manifests/pretrain/plan.json'))
    p.add_argument('--run',action='store_true')
    p.add_argument('--use-swanlab',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--num-workers',type=int,default=None)
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());validate_config(plan)
    if not args.run:
        print('CONFIG_READY. No data verification performed; no training started.');return
    output=ROOT/plan['output_root'];weight='qwen'
    for phase in plan['phases']:
        target=output/phase['name']
        for kind in ['last']:recover_checkpoint(target/kind)
        state=target/'last/checkpoint/trainer_state.pt'
        resume=state.exists()
        if resume:
            import torch
            saved=torch.load(state,map_location='cpu',weights_only=False)
            migration=validate_checkpoint_signature(plan,phase['name'],plan['world_size'],saved.get('pretrain_signature'),ROOT)
            if migration:print('EXECUTION_MIGRATION',phase['name'],migration,flush=True)
            if saved.get('phase_complete'):
                weight=str(target/'last/checkpoint')
                if not Path(weight,'config.json').exists():raise ValueError('Missing last checkpoint')
                print('COMPLETE',phase['name']);continue
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f"--nproc_per_node={plan['world_size']}",str(ROOT/'trainer/train_pretrain.py'),'--plan',str(Path(args.plan).resolve()),'--phase',phase['name'],'--from-weight',weight,'--output',str(target),'--execute']
        if args.num_workers is not None:cmd+=['--num-workers',str(args.num_workers)]
        if resume:cmd.append('--resume')
        if args.use_swanlab:cmd.append('--use-swanlab')
        target.mkdir(parents=True,exist_ok=True)
        with open(target/'training.log','a',buffering=1) as log:
            print('START',phase['name'],'log',log.name,flush=True)
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env=training_environment(plan.get('execution',{}).get('cpu_threads_per_process',1)))
        weight=str(target/'last/checkpoint')
    print('PRETRAIN_COMPLETE',weight)
if __name__=='__main__':main()
