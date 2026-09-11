# MiniQwen-Omni architecture evaluation

This pipeline is a fixed-data, fixed-seed gate for comparing Talker, bridge and
multimodal architecture changes. It is separate from `trainer/train.sh` and
never writes to `out/miniqwen_omni_full`.

## Preserved pre-MRoPE baseline

The completed `baseline_6l` experiment is retained read-only for comparison.
Do not rerun it with the format-v4 code.

## Format-v5 Main codec + Code Predictor candidate

```bash
cd /path/to/miniqwen-omni
source envs/Omni-ppu/bin/activate
bash trainer/train_arch_eval.sh
```

The default experiment name is `main_codec_cp_2l_v1`. It starts from
Qwen3-0.6B, retains the completed `mrope_special_multiimage_v1` architecture,
and replaces only the delayed eight-head codec output with a same-frame Main
codec head plus a 2 × 768 Code Predictor. The report compares against
`mrope_special_multiimage_v1` by default.

T2A and A2A directly reuse the complete `sft_t2a_mini.parquet` and
`sft_a2a_mini.parquet`; no duplicate training parquet is created. Their fixed
English dev samples are drawn from those mini sources. I2T uses 32,768 English
training samples and 512 group-disjoint English dev samples. Prepared dev/I2T
data is stored under `dataset/arch_eval`; every later architecture candidate
reuses those exact parquet files.

The five stages are:

1. Full T2A mini: all parameters, with Qwen at `1e-5` and new omni modules at
   `5e-4`.
2. Full A2A mini alignment: `audio_proj` only at `5e-4`.
3. Full A2A mini joint tuning: all parameters, with Qwen at `2e-6` and omni
   modules at `2e-5`.
4. English I2T alignment: `vision_proj` only at `5e-4`.
5. English I2T joint tuning: Thinker, text head and vision projector only, with
   low differential learning rates. Talker is frozen and skipped on I2T
   batches to preserve the audio result.

Each stage overwrites the experiment checkpoint, evaluates all three fixed dev
sets, then advances `pipeline_stage`. Re-running the same command resumes an
interrupted stage and skips completed stages.

## New candidates

Always use a new experiment name after changing architecture code:

```bash
ARCH_EVAL_NAME=talker_change_v1 bash trainer/train_arch_eval.sh
```

The common layer ablations can be launched without editing code:

```bash
ARCH_EVAL_NAME=talker_4l ARCH_EVAL_NUM_TALKER_LAYERS=4 bash trainer/train_arch_eval.sh
ARCH_EVAL_NAME=talker_8l ARCH_EVAL_NUM_TALKER_LAYERS=8 bash trainer/train_arch_eval.sh
```

Useful controls:

```bash
# Stop after A2A joint, then continue later with the same command.
ARCH_EVAL_NAME=my_test ARCH_EVAL_STOP_AFTER_STAGE=2 bash trainer/train_arch_eval.sh

# Skip the four-sample English qualitative generation suite when only metrics
# are needed. The standard complete run generates it and embeds it in REPORT.md.
ARCH_EVAL_NAME=my_test ARCH_EVAL_RUN_GENERATION=0 bash trainer/train_arch_eval.sh

# Metrics-only run (training still uses the standard fixed datasets).
ARCH_EVAL_NAME=metrics_only ARCH_EVAL_RUN_GENERATION=0 \
  ARCH_EVAL_MAX_SAMPLES=32 bash trainer/train_arch_eval.sh
```

Do not reuse one experiment directory after changing its architecture. A saved
optimizer is only compatible with the parameter structure that created it.

## Results

Training logs:

```text
.runtime/train_logs/arch_eval/<experiment>/
```

Per-stage deterministic metrics and generated samples:

```text
.runtime/arch_eval/<experiment>/
```

Every stage refreshes `.runtime/arch_eval/<experiment>/REPORT.md`. The report
contains the full metric table and automatically compares completed candidates
with `mrope_special_multiimage_v1`. When qualitative generation is enabled, it also embeds the
input image/audio, Thinker response, playable Talker MP3 files, generation RTF,
SenseVoice transcription and generated-speech/text WER.

Compare completed experiments:

```bash
python scripts/compare_arch_eval.py
```

Lower losses are better. Positive `audio_input_gain`, `speaker_only_gain`, and
`vision_gain` mean the model is using the corresponding condition. The report's
generated English text and playable MP3 files are required sanity checks; loss
alone does not prove intelligible speech. Compare candidates only when their
data manifest, stage, epoch budget and initialization are identical.

The active experiment keeps one overwrite-style checkpoint at
`out/arch_eval/<experiment>/checkpoint`. After recording metrics and listening
to generated samples, an old candidate checkpoint can be removed while keeping
the much smaller `.runtime/arch_eval/<experiment>` report.
