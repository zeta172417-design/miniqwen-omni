# Omni Benchmark V0 baseline

This is the first frozen evaluation baseline for MiniQwen-Omni. It was produced
on 2026-09-18 with scorer 2.3.0, data-builder 2.3.0, 752 main samples, and 16
voice-cloning samples. All evaluated samples completed without runtime errors.

The full local report, generated audio, and per-sample records are intentionally
excluded from Git because they occupy about 345 MB. Their canonical run name is
`omni-benchmark-v0-standard-20260918T034920Z`.

## Headline results

| Model | Fixed Omni | Common Task | Coverage | Failure |
|---|---:|---:|---:|---:|
| Mini-Omni2 | 23.99 | 38.16 | 60% | 0% |
| MiniMind-3O-MoE | 8.10 | 8.22 | 90% | 0% |
| MiniQwen V0 | 23.70 | 18.02 | 90% | 0% |
| MiniQwen V0.1 | 23.55 | 16.70 | 100% | 0% |
| Qwen2.5-Omni-3B | 83.81 | 82.92 | 100% | 0% |

Fixed Omni is the equal-weight macro average of ASR, speech understanding,
speech generation, vision understanding, and cross-modal understanding.
Unsupported required tasks score zero. The text regression guard is reported
separately and is excluded from the aggregate.

## MiniQwen V0 versus V0.1

| Capability | V0 | V0.1 | V0.1 − V0 |
|---|---:|---:|---:|
| ASR | 10.49 | 5.46 | -5.03 |
| Speech understanding | 3.75 | 1.25 | -2.50 |
| Speech generation / TTS | 78.07 | 86.80 | +8.73 |
| Vision understanding | 25.41 | 19.56 | -5.85 |
| Cross-modal / multi-image | 0.78 | 4.69 | +3.91 |
| Fixed Omni | 23.70 | 23.55 | -0.15 |

The paired Fixed Omni difference is -0.15 points with a 95% bootstrap interval
of [-2.50, +2.25], so the two versions are tied on the overall benchmark.
V0.1 has a statistically reliable TTS-content improvement: WER falls from
23.02% to 13.69%, and exact text repetition rises from 47/64 to 61/64. Its
UTMOS naturalness score falls slightly from 3.66 to 3.54.

The pure-text regression guard is identical for V0 and V0.1: 65.62% accuracy,
100% answer parse rate, and 87.5% strict-format compliance. The lower audio and
vision scores therefore do not establish a general instruction-following
regression. The two checkpoints are not a controlled Talker-only ablation:
their Thinker and projector weights, modality boundaries, positional encoding,
world size, effective batch, and optimization trajectories differ.

## Voice cloning

| Model | Speaker cosine | Content 1-WER | UTMOS | Valid audio |
|---|---:|---:|---:|---:|
| MiniQwen V0 | 0.1177 | 80.50 | 3.43 | 100% |
| MiniQwen V0.1 | 0.1532 | 81.29 | 3.07 | 100% |
| MiniMind-3O-MoE | 0.1361 | 27.97 | 3.29 | 100% |

V0.1 shows a small, statistically inconclusive speaker-similarity gain and a
naturalness decline. The independent real reference-to-label cosine is 0.7766,
which shows that substantial voice-cloning headroom remains.

## Baseline decision

MiniQwen V0.1 is frozen as the architecture baseline for subsequent work. The
Main codec + Code Predictor design is retained because it materially improves
same-frame audio-code prediction and spoken-content accuracy. Future iterations
should focus on data and training rather than another immediate Talker redesign:

1. Add explicit English ASR/transcription supervision.
2. Improve audio-question and image-plus-audio instruction data.
3. Rebalance optimizer steps and effective batch across 4- and 16-PPU runs.
4. Preserve TTS content accuracy while optimizing perceptual naturalness and
   speaker similarity.
5. Keep a frozen text-regression guard to detect Thinker degradation.

Quick is intended for rapid screening. Standard remains the release-quality
comparison and should be rerun before accepting a future architecture or
training-pipeline version.
