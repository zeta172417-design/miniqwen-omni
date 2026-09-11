# MiniQwen-Omni

MiniQwen-Omni is an end-to-end multimodal training project built around a Qwen3-0.6B Thinker. It accepts text, speech, and images and jointly produces text and eight Mimi audio-code streams.

This repository contains code only. Datasets, pretrained auxiliary models, checkpoints, experiment logs, and generated media are excluded by `.gitignore`.

The current code release is **V0.1**. V0 denotes the initially trained and uploaded parallel-delay Talker; V0.1 replaces it with same-frame Main codec head + Code Predictor audio generation and is the new baseline. Releases are recorded in both the root `VERSION` file and Git tags.

## Architecture

- Qwen3-0.6B Thinker: 28 layers, hidden size 1024, native Qwen tokenizer/chat template/generation config.
- Bridge: hidden state from Thinker layer 14 by default, projected from 1024 to 768.
- Talker: 6 × 768 by default; 4/6/8 layers remain configurable for ablations. A Main head predicts c0 and a 2 × 768 Code Predictor autoregressively predicts c1–c7 within the same Mimi frame; the mean of all eight frame embeddings is fed back between Talker steps.
- SenseVoice audio projector: 512 → 1024.
- SigLIP2 vision projector: 768 → 1024, with 64 image tokens.
- Eight Mimi codebook output heads.
- FP32 master parameters with BF16 autocast and separate Qwen/Omni learning rates.

The Talker blocks live in `model/model_talker.py`; the obsolete MiniMind language backbone is not included.

## Train

Prepare the Qwen3, SenseVoice, SigLIP2 and Mimi directories described in `README.md`, then run:

```bash
cd trainer
source ../envs/Omni-ppu/bin/activate
bash train.sh
```

The seven-stage pipeline defaults to single-node 16-device DDP and supports `NPROC_PER_NODE`/`PPU_DEVICES` overrides. It uses SwanLab tracking, stable overwrite checkpoints, direct resume, dynamic removal of batch-wide padding tails, supervised-only vocabulary projection, and differential learning rates.

## Evaluate

```bash
source envs/Omni-ppu/bin/activate
python eval_omni.py --load_from out/miniqwen_omni_full_main_codec_cp_v5/checkpoint --mode 0 --max_samples 1
```

Frozen SenseVoice, SigLIP2, and Mimi weights are external dependencies and are not bundled into the core checkpoint.

## Password-protected Web demo

The Gradio app supports text, microphone/audio and image input, streamed text, multi-turn context, and Mimi speech output. Public binding requires a password read only from the environment:

```bash
read -rsp 'Web password: ' MINIQWEN_WEB_PASSWORD
echo
export MINIQWEN_WEB_PASSWORD
bash scripts/serve_web.sh
```

See `WEB_DEPLOYMENT.md` for tmux, temporary `gradio.live` sharing, and an Alibaba Cloud DSW public endpoint.

The UI has independent **Text / image** and **Audio** input tabs. Choose either fast text-only output or text plus speech. To send audio alone, stop recording, wait for the waveform, and click **Send this audio** without entering text.

## Export

```bash
python scripts/export_modelscope.py \
  --checkpoint out/miniqwen_omni_full_main_codec_cp_v5/checkpoint \
  --output releases/miniqwen-omni-bf16
```

The export converts FP32 training weights to BF16 inference weights and excludes optimizer/trainer state. See `PUBLISHING.md` for private GitHub and ModelScope commands.

## Test

```bash
python -m unittest tests.test_miniqwen_omni tests.test_web_demo
bash -n trainer/train.sh trainer/train_mini.sh scripts/serve_web.sh
```

## Attribution

This project evolved from the Omni data interface and Thinker–Talker design of [MiniMind-O](https://github.com/jingyaogong/minimind-o). The text backbone, tokenizer, cache semantics, precision policy, and training pipeline have been migrated to Qwen3-0.6B. See `LICENSE` for the retained license and copyright notice.
