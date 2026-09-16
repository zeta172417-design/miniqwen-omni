# MiniQwen-Omni architecture evaluation

This pipeline is a fixed-data, fixed-seed gate for comparing Talker, bridge and
multimodal architecture changes. It is separate from `trainer/train.sh` and
never writes to `out/miniqwen_omni_full`.

## Archived pre-MRoPE baseline

The completed `baseline_6l` metrics and generated-sample report are retained
read-only for comparison. Its heavyweight checkpoint has been removed after
the result was recorded. Do not rerun it with the format-v4 code.

## Format-v5 Main codec + Code Predictor candidate

```bash
cd /path/to/miniqwen-omni
source envs/Omni-ppu/bin/activate
bash trainer/train_arch_eval.sh
```

The default experiment name is `main_codec_cp_2l_v1`. It starts from
Qwen3-0.6B, retains the completed `mrope_special_multiimage_v1` architecture,
and replaces only the delayed eight-head codec output with a same-frame Main
codec head plus a 2 × 768 Code Predictor. This completed experiment is the
V0.1 baseline; later candidate reports compare against
`main_codec_cp_2l_v1` by default on the full-corpus holdout.

T2A and A2A directly reuse the complete `sft_t2a_mini.parquet` and
`sft_a2a_mini.parquet`; no duplicate training parquet is created. The primary
English evaluation is instead sampled from the full corpora into
`dataset/arch_eval_holdout`: T2A excludes every conversation seen in mini
training, A2A excludes every mini-training speaker embedding, and I2T excludes
train-image hashes. The report records source counts, group keys, zero-overlap
checks and example rows. MiniMind's fixed local audio/image prompts remain as
the qualitative listening demo; they are not used as the quantitative gate.
I2T training still uses the fixed 32,768-row English subset in
`dataset/arch_eval`.

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

Every stage refreshes `.runtime/arch_eval/<experiment>/REPORT.md`. A completed
pipeline additionally writes `holdout-final.json`; its full-corpus holdout
section is the primary architecture-selection gate, while the historical
per-stage table is a training-domain diagnostic. The report automatically
compares completed candidates with the configured baseline when their holdout
manifests match. When qualitative generation is enabled, it also embeds the
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

The recorded `baseline_6l`, `mrope_special_multiimage_v1`, and
`main_codec_cp_2l_v1` reports are currently archived this way; their
`out/arch_eval` checkpoints are intentionally absent. The abandoned encoder
unfreeze candidate and its partial checkpoint are not part of the V0.1 release.
