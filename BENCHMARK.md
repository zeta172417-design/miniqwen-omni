# Omni Benchmark V0

This suite focuses on Omni capabilities rather than re-evaluating the pretrained
Qwen text backbone. It compares MiniQwen V0, MiniQwen V0.1, MiniMind-3O-MoE,
Mini-Omni2, and Qwen2.5-Omni-3B with objective English metrics.

V0 uses scorer 2.3.0 and data-builder 2.3.0. The scorer separates semantic
correctness from output-format compliance. MCQ
answers may be an explicit option letter, one uniquely stated option text, or
one unambiguous standalone capital option letter. A verbose correct answer is
therefore not scored as wrong merely for ignoring the letter-only instruction;
strict format compliance remains visible as a diagnostic. TextVQA retains the
annotation-consensus rule while allowing a normalized reference answer as a
complete token span in a brief explanation.

## Capability set

The standard profile contains 752 main samples plus 16 reference voice-cloning
samples. The quick profile contains 110 main samples plus 8 cloning samples.

The frozen first official result is summarized in
[`benchmark/reports/OMNI_BENCHMARK_V0.md`](benchmark/reports/OMNI_BENCHMARK_V0.md).
MiniQwen V0.1 is the architecture baseline for subsequent data and training
pipeline improvements; MiniQwen V0 remains the historical parallel-delay
comparison model.

| Task | Standard | Input → output | Primary diagnostics |
|---|---:|---|---|
| LibriSpeech test-clean/test-other ASR | 160 | audio → text | WER |
| Spoken QA A2T | 80 | audio → text | answer accuracy |
| Spoken QA A2A | 80 | audio → text + audio | spoken-answer accuracy, text/speech consistency |
| TTS | 64 | text → audio | ASR-WER, UTMOS, validity |
| MMBench-EN | 120 | image + text → text | accuracy |
| POPE | 60 | image + text → text | F1 and accuracy |
| TextVQA | 60 | image + text → text | VQA score |
| Image + spoken question | 64 | image + audio → text | paired accuracy |
| NLVR2 | 32 | two images + text → text | accuracy |
| Text regression guard | 32 | text → text | accuracy, excluded from Omni score |
| Voice clone | 16 | reference audio + text → audio | speaker cosine, 1-WER, UTMOS |

Every manifest row declares `output_modality` and `require_audio`. ASR, image
understanding, audio-vision understanding, multi-image, and text-regression rows
never start Talker. Only A2A, TTS, and voice cloning generate speech.

Unsupported public APIs remain `N/A` in task diagnostics, but count as zero in
the headline Fixed Omni score. Mini-Omni2's public API, for example, does not
expose text-question image inference or multi-image input. The secondary Common
Task score averages the exact non-text tasks completed by every model in the
same report, so unlike a capability-name intersection it compares like with
like.

## Prepare data

The runner automatically prepares a missing manifest. To prepare explicitly:

```bash
cd /mnt/workspace/zhaozetao/multimodel/miniqwen-omni
source envs/Omni-ppu/bin/activate

python benchmark/prepare_data.py \
  --config benchmark/configs/quick.json \
  --output /mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/omni-core-v2.3-quick \
  --skip-decontamination

python benchmark/prepare_data.py \
  --config benchmark/configs/standard.json \
  --output /mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/omni-core-v2.3-standard
```

The first standard preparation scans the three training parquet files for raw
and modality-tag-stripped normalized overlap. Its compact hash set is cached
under `.runtime/benchmark_cache`; later standard preparation reuses it. Quick
is intended for frequent architecture iteration and explicitly skips the costly
scan; this is recorded in its metadata. An official annotated TextVQA
validation shard is retained in the shared data directory; other source media
use the ModelScope cache. To reproduce that download through the domestic
Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download lmms-lab-encoder/textvqa \
  --repo-type dataset \
  --include 'data/validation-00002-of-00003.parquet' \
  --local-dir /mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/hf-textvqa-validation
```

## Run

Quick architecture comparison:

```bash
source envs/Omni-ppu/bin/activate
bash scripts/run_omni_benchmark.sh quick
```

Formal comparison:

```bash
source envs/Omni-ppu/bin/activate
bash scripts/run_omni_benchmark.sh standard
```

Select models or devices without editing code:

```bash
OMNI_BENCH_MODELS=miniqwen-v0.1,minimind-3o-moe \
OMNI_BENCH_DEVICES=0,1,2,3 \
bash scripts/run_omni_benchmark.sh quick
```

Evaluation is data parallel: each process owns a complete model and a disjoint
manifest shard. No model weights are split across PPUs. MiniQwen V0.1 uses
native batch 4 by default; adapters without safe batching use batch 1. Runs are
resumable with the same `OMNI_BENCH_RUN_ID`. Resume is rejected if the manifest,
checkpoint, adapter, seed, shard layout, batch size, or decoding mode changed.
Text generation is greedy so repeated runs are deterministic at the decoding
level. Qwen2.5-Omni audio generation uses its model-card-required system prompt
and a label-independent Talker safety cap; natural-label duration is never fed
to model inference.

The same SenseVoice checkpoint scores generated speech on PPU. Set
`OMNI_BENCH_ASR_DEVICE=cpu` only for fallback. UTMOS predicts naturalness MOS
on a roughly 1–5 scale and is enabled by default from the shared SpeechMOS
checkout. ASR-WER measures content, speaker cosine measures voice identity,
and UTMOS measures perceptual quality; all three are reported because none is
a substitute for the others. To override its location:

```bash
OMNI_BENCH_UTMOS_REPO=/path/to/SpeechMOS \
bash scripts/run_omni_benchmark.sh standard
```

Set `OMNI_BENCH_UTMOS_REPO=0` only when MOS scoring is intentionally disabled.

## Outputs

Results are written under:

```text
/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-results/<run-id>/
```

Each run contains per-sample JSONL, task/capability metrics, generated WAV files,
`summary.csv`, `failures.csv`, and `REPORT.md`. The report includes directly
playable TTS natural labels, generated TTS, and voice-cloning reference/label/
generated examples.

Run the code-only smoke test without loading a real model:

```bash
bash scripts/run_omni_benchmark.sh smoke
```
