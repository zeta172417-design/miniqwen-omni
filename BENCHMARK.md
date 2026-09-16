# MiniQwen-Omni objective benchmark

The first benchmark compares MiniQwen V0, MiniQwen V0.1, the larger
Qwen2.5-Omni-3B upper bound, and Mini-Omni2. V0.1 is the frozen-encoder release;
experimental architecture candidates are not part of the first full run. The
headline score uses objective metrics only; inference
failures and empty outputs remain in the denominator.

## Workspace

The default layout is:

```text
/mnt/workspace/zhaozetao/multimodel/Omni/
├── miniqwen-omni -> ../miniqwen-omni
├── mini-omni2/{source,checkpoint}
├── Qwen2.5-Omni-3B/
├── benchmark-data/
├── benchmark-results/
└── models-manifest.json
```

Qwen2.5-Omni-3B is downloaded from the official ModelScope repository.
Mini-Omni2 has no author-maintained ModelScope repository, so its code and
checkpoint are pinned to the official GitHub and Hugging Face revisions.

Recreate or verify model downloads with:

```bash
cd /mnt/workspace/zhaozetao/multimodel/miniqwen-omni
source envs/Omni-ppu/bin/activate
bash scripts/download_benchmark_models.sh
```

## Data

The standard English suite contains:

| Category | Data | Samples | Metric |
|---|---|---:|---|
| Text | ARC-Easy, BoolQ, PIQA | 500 | accuracy |
| Spoken MCQ | fixed Flite rendering of held-out MCQs | 300 | accuracy |
| ASR | LibriSpeech test-clean/test-other | 500 | 1-WER |
| Vision | MMBench-EN | 350 | accuracy |
| Image + spoken question | POPE | 150 | accuracy |
| Instruction following | generated fixed constraints | 200 | constraint pass rate |
| Robustness | paired 10 dB noisy speech | 200 | preservation rate |

The same offline English voices (`slt`, `rms`) are used for every model.
Mini-Omni2's published vision interface requires a spoken question, so all
models receive identical image + spoken-question inputs for vision rows.
All input audio is capped at 30 seconds. Generated Flite files are reused only
when their stored content signature matches the current prompt and voice.

Prepare the data through ModelScope:

```bash
source envs/Omni-ppu/bin/activate
python -u benchmark/prepare_data.py
```

For a quick data-format check, append `--skip-audio --skip-decontamination`.
The final command scans the MiniQwen training parquet files and removes exact
normalized prompt overlaps. The manifest and its SHA256 are recorded under
`Omni/benchmark-data/standard`.

## Run

First verify the scoring/report path without loading a model:

```bash
source envs/Omni-ppu/bin/activate
python -m benchmark.run_benchmark \
  --config benchmark/configs/smoke.json --model mock
python -m benchmark.build_report \
  --run-dir /mnt/workspace/zhaozetao/multimodel/Omni/benchmark-results/smoke-v1
```

Run one real model directly when debugging an adapter:

```bash
source envs/Omni-ppu/bin/activate
RUN_ID=standard-$(date -u +%Y%m%dT%H%M%SZ)
python -u benchmark/run_benchmark.py --config benchmark/configs/models.json \
  --model miniqwen-v0.1 --run-id "$RUN_ID" --resume
```

Run the complete four-model suite in parallel on four PPUs:

```bash
source envs/Omni-ppu/bin/activate
bash scripts/run_omni_benchmark.sh full
```

By default, MiniQwen V0, MiniQwen V0.1, Qwen2.5-Omni-3B and Mini-Omni2 are
assigned to PPU 0/1/2/3 respectively. Inference finishes before the shared-ASR
phase begins; ASR scoring is then also distributed one process per PPU. Runs
are resumable and share one `RUN_ID` across all models. Progress is printed on
the first sample, every 10 samples, on failures, and on the final sample.

Select the available cards with `OMNI_BENCH_DEVICES=0,1,2,3`. For a strictly
sequential efficiency run, use `OMNI_BENCH_PARALLEL=0`; this avoids shared
CPU/I/O contention and therefore gives the cleanest latency comparison. The
default parallel mode still isolates each model's PPU memory because each model
has its own card, and substantially reduces total wall-clock time.

Before a full run, validate the runner and report builder without loading a
model, then optionally run one real Mini-Omni2 sample:

```bash
bash scripts/run_omni_benchmark.sh smoke
bash scripts/run_omni_benchmark.sh mini-omni2-smoke
```

On the PPU host, Mini-Omni2 uses the existing PPU Torch plus isolated
compatibility packages from `.runtime/mini_omni2_ppu_deps`; it never replaces
the training environment's Torch. An official Python 3.10/CUDA virtualenv can
still be selected with `MINIOMNI_BENCH_PYTHON=/path/to/python`.

Mini-Omni2's official environment pins Python 3.10 and PyTorch 2.3.1. The
included compatibility setup keeps the host's PPU Torch 2.11 build and places
only Python-level dependencies in `.runtime/mini_omni2_ppu_deps`. Run
`bash scripts/setup_benchmark_envs.sh` once before evaluation. The adapter has
been smoke-tested on PPU-ZW810E; an official CUDA environment remains an
optional fallback and should be identified as cross-device in efficiency
comparisons.

Qwen inference needs `qwen-omni-utils`; install `benchmark/requirements-qwen.txt`
without replacing the PPU build of Torch. Mini-Omni2 dependencies are listed in
`benchmark/requirements-mini-omni2.txt` for its separate environment.

## Shared speech scoring and report

Run the same ASR over every model's generated speech. On the PPU host the
default pipeline uses PPU inference (`cuda:0` inside each process's visible
device):

```bash
python benchmark/evaluate_audio.py \
  --input /path/to/model/per_sample.jsonl \
  --asr model/SenseVoiceSmall \
  --device cuda:0
python benchmark/build_report.py --run-dir /path/to/run
```

The first PPU ASR batch includes graph/kernel warm-up. On this host a warmed
20-sample comparison measured about 12.43 samples/s on PPU versus 4.37
samples/s on CPU. Use `OMNI_BENCH_ASR_DEVICE=cpu` for a CPU fallback.

The run directory contains per-sample JSONL, aggregate JSON/CSV, failures,
generated WAV files, and `REPORT.md` with directly playable examples. UTMOS is
optional and can be enabled with a local SpeechMOS checkout via `--utmos-repo`;
without it, speech quality is explicitly limited to validity, silence and
clipping diagnostics.

The weighted quality score is text 10%, audio 20%, vision 20%, image+audio 15%,
text-speech consistency 15%, speech quality 15%, and robustness 5%. Missing
capabilities are not silently dropped after a full run: they appear as failed
samples. Qwen2.5-Omni-3B is an upper bound rather than a parameter-matched model.

## MiniQwen voice cloning

The full runner also evaluates MiniQwen V0 and V0.1 on a fixed 48-sample
LibriSpeech test-clean subset (12 speakers and four target utterances per
speaker). One different utterance supplies the CAMPPlus/Mimi reference
condition. The natural target recording is never passed to the model.

Speaker similarity is scored by the independent English VoxCeleb ERes2Net
model under `model/eres2net_sv_en_voxceleb_16k`; content accuracy uses the
shared SenseVoice ASR and 1-WER. Voice cloning remains a separate diagnostic
section and is not added to the four-model headline score because Qwen2.5-Omni
and Mini-Omni2 do not expose equivalent arbitrary-reference cloning inputs.

To rerun only this small section and rebuild an existing main report:

```bash
bash scripts/run_voice_clone_benchmark.sh standard-YYYYMMDDTHHMMSSZ
```
