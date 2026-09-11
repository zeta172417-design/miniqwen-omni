# MiniQwen-Omni

MiniQwen-Omni 是一个基于 Qwen3-0.6B Thinker 的端到端多模态训练项目，支持文本、语音和图像输入，并联合生成文本与 8 层 Mimi audio codes。

本仓库只保存训练与推理代码。训练数据、本地预训练模型、训练 checkpoint、实验日志和生成音频均由 `.gitignore` 排除。

当前代码版本为 **V0.1**。V0 是最初完成全量训练并上传的 parallel-delay Talker 版本；V0.1 将音频生成头升级为同帧 Main codec head + Code Predictor，并作为后续实验的新基线。发布版本同时记录在根目录 `VERSION` 和 Git tag 中。

## 当前架构

- Thinker：Qwen3-0.6B，28 层，hidden size 1024，完整使用 Qwen tokenizer、chat template 和 generation config。
- Bridge：默认接收 Thinker 第 14 层 hidden state，执行 1024 → 768 映射。
- Talker：默认 6 层、hidden size 768；层数可配置为 4/6/8，用于消融实验。
- Audio projector：SenseVoice 512 → Qwen 1024。
- Vision projector：SigLIP2 768 → Qwen 1024，每张图像对应 64 个 image tokens。
- Audio output：Main head 预测当前帧 c0；2 × 768 Code Predictor 在帧内自回归预测 c1～c7。上一完整 Mimi frame 的 8 路 embedding 取 mean 后反馈给 Talker。
- 训练精度：FP32 master parameters + BF16 autocast。
- 优化策略：Qwen 与新建 Omni 模块使用差分学习率。

Talker 使用的 Transformer block 位于 `model/model_talker.py`。它是独立声学解码结构，不包含旧 MiniMind Thinker 或语言模型。

## 目录结构

```text
miniqwen-omni/
├── dataset/
│   └── omni_dataset.py
├── model/
│   ├── model_omni.py
│   └── model_talker.py
├── trainer/
│   ├── train.sh
│   ├── train_mini.sh
│   ├── train_sft_omni.py
│   └── trainer_utils.py
├── scripts/
│   ├── export_modelscope.py
│   └── web_demo_omni.py
├── tests/
├── eval_omni.py
├── eval_mini.sh
├── requirements.txt
└── requirements-ppu.txt
```

以下本地目录需要自行准备，但不会提交到 Git：

```text
dataset/sft_t2a.parquet
dataset/sft_a2a.parquet
dataset/sft_i2t.parquet
model/Qwen3-0.6B/
model/SenseVoiceSmall/
model/siglip2-base-p32-256-ve/
model/mimi/
model/campplus/                 # 仅音色克隆需要
model/speaker/                  # 可选预设音色
model/vad/                      # 可选实时 VAD
```

## PPU 环境

项目已在以下环境完成训练闭环：

- 4 × PPU-ZW810E（每卡约 100GB）
- Python 3.12.3
- PPU SDK/PCCL 2.1.1
- torch 2.11.0+v0.1.0.ppu2.1.1
- torchaudio 2.11.0、torchvision 0.26.0（PPU SDK 源码编译）
- Pillow 11.3.0、FFmpeg 6.1.1

```bash
source envs/Omni-ppu/bin/activate
pip install -r requirements.txt
```

PPU 版本的核心包锁定值见 `requirements-ppu.txt`，不要用 PyPI 的标准 torch 覆盖 PPU 构建。

## 训练

完整七阶段流水线：

```bash
cd trainer
source ../envs/Omni-ppu/bin/activate
bash train.sh
```

训练脚本具有以下行为：

- 默认使用单机 16 卡 DDP 和 SwanLab；可通过 `NPROC_PER_NODE`、`PPU_DEVICES` 覆盖设备配置。
- 每个 stage 自动继承上一个 stage 的稳定 checkpoint。
- 每个大 epoch 结束保存一次，并原子覆盖同一目录，限制磁盘占用。
- checkpoint 保留 FP32 master weights、optimizer、scaler 和 RNG 状态，支持 `bash train.sh` 直接续训。
- 所有 stage 使用动态裁尾，但不会删除 attention、文本标签或音频标签覆盖的有效 token。
- 文本词表投影只计算 assistant label 位置。
- 纯 `vision_proj` 阶段跳过恒为零损失的 Talker 分支。

稳定训练目录：

```text
out/miniqwen_omni_full_main_codec_cp_v5/checkpoint/
```

快速验证音频链路：

```bash
cd trainer
bash train_mini.sh
```

## 推理

核心 checkpoint 不包含冻结的 SenseVoice、SigLIP2 和 Mimi 权重；运行推理前必须准备这些辅助模型。

```bash
source envs/Omni-ppu/bin/activate
python eval_omni.py \
  --load_from out/miniqwen_omni_full_main_codec_cp_v5/checkpoint \
  --mode 0 \
  --max_samples 1
```

`--mode` 支持：`0=text`、`1=multi-turn`、`2=audio`、`3=clone`、`4=image`、`5=mix`、`-1=all`。

## 带密码的 Web 体验

项目提供支持文本、麦克风、图片、流式文本和语音回复的 Gradio 服务。密码只从环境变量读取，公网启动时强制启用认证：

```bash
read -rsp 'Web password: ' MINIQWEN_WEB_PASSWORD
echo
export MINIQWEN_WEB_PASSWORD
bash scripts/serve_web.sh
```

tmux 后台运行、临时 `gradio.live` 地址以及 DSW 固定公网映射方法见 `WEB_DEPLOYMENT.md`。

Web 界面将“文字 / 图片”和“语音”作为两个独立输入页。回复形式可选择“仅文字（更快）”或“文字 + 语音”；录音停止并出现波形后可直接点击“发送这段语音”，无需填写文本。

## 导出与发布

训练 checkpoint 是用于精确续训的 FP32 版本，不应直接作为推理模型上传。使用下面的命令生成不含 optimizer 的 BF16 ModelScope 目录：

```bash
python scripts/export_modelscope.py \
  --checkpoint out/miniqwen_omni_full_main_codec_cp_v5/checkpoint \
  --output releases/miniqwen-omni-bf16
```

导出目录包含 BF16 safetensors、Qwen tokenizer、generation config、remote-code 模型文件、模型卡和 SHA256 manifest；不会包含 `trainer_state.pt`。

ModelScope 与私人 GitHub 的具体创建和上传命令见 `PUBLISHING.md`。

## 测试

```bash
python -m unittest tests.test_miniqwen_omni tests.test_web_demo
bash -n trainer/train.sh trainer/train_mini.sh scripts/serve_web.sh
```

## 来源与许可

本项目从 [MiniMind-O](https://github.com/jingyaogong/minimind-o) 的 Omni 数据接口与 Thinker–Talker 思路演化而来，文本主干、tokenizer、缓存、训练精度和训练流水线已迁移到 Qwen3-0.6B。原项目版权和许可信息保留在 `LICENSE` 中。
