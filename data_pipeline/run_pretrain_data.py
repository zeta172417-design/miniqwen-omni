#!/usr/bin/env python3
"""Prepare the five initial corpora concurrently, with separate resumable logs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    log = ROOT / '.runtime/pretrain'
    log.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', default=['librispeech', 'aishell1', 'llava', 'libritts_r', 'aishell3'])
    sources = parser.parse_args().sources

    def run(source):
        with open(log / (source + '.log'), 'a') as f:
            f.write('\n[run/resume]\n')
            f.flush()
            command = [sys.executable, '-B', '-u', str(ROOT / 'data_pipeline/prepare_pretrain_data.py'),
                       '--sources', source, '--download-workers', '8' if source == 'librispeech' else '32',
                       '--file-workers', '16' if source == 'librispeech' else '1', '--stage', 'download', '--domestic']
            p = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if p.returncode == 0:
                # Start a fresh interpreter for conversion, independently resumable.
                p = subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'data_pipeline/prepare_pretrain_data.py'),
                                    '--sources', source, '--stage', 'convert'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        return source, p.returncode

    results = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(run, s) for s in sources]):
            source, code = future.result()
            results[source] = code
            print('SOURCE_EXIT', source, code, flush=True)
            (log / 'exit_codes.json').write_text(json.dumps(results, indent=2))
    raise SystemExit(any(results.values()))


if __name__ == '__main__':
    main()
