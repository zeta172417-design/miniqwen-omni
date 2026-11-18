#!/usr/bin/env python3
"""Show the SFT plan by default; --run explicitly launches distributed training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.runtime_env import training_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(ROOT / 'dataset/raw/manifests/sft/plan.json'))
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', default='',
                        help='Optional downloaded V1-Base checkpoint under model/')
    parser.add_argument('--use-swanlab', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    path = Path(args.plan).resolve()
    plan = json.loads(path.read_text())
    for key in ['world_size', 'epochs', 'batch_size', 'accumulation_steps']:
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    batch = plan['world_size'] * plan['batch_size'] * plan['accumulation_steps']
    if batch != plan['global_batch']:
        raise ValueError('Batch mismatch')
    print('SFT_CONFIG', 'epochs', plan['epochs'], 'world_size', plan['world_size'],
          'batch_per_rank', plan['batch_size'], 'accumulation', plan['accumulation_steps'],
          'global_batch', batch, flush=True)
    if not args.run:
        print('CONFIG_READY. No training started.')
        return
    output = ROOT / plan['output']
    output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           f"--nproc_per_node={plan['world_size']}", str(ROOT / 'trainer/train_sft.py'),
           '--plan', str(path)]
    if args.resume:
        cmd.append('--resume')
    if args.checkpoint:
        cmd += ['--checkpoint', args.checkpoint]
    if not args.use_swanlab:
        cmd.append('--no-tracking')
    with (output / 'training.log').open('a', buffering=1) as log:
        print('START SFT', 'log', log.name, flush=True)
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=training_environment(plan.get('execution', {}).get('cpu_threads_per_process', 1)))


if __name__ == '__main__':
    main()
