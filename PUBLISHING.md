# 发布指南

代码仓库和模型仓库应分开发布：GitHub 只放源码，ModelScope 只上传导出的 BF16 推理模型。不要上传 `out/**/checkpoint/trainer_state.pt`，其中包含 optimizer、随机数状态和 SwanLab run 信息。

## 1. 发布到私人 GitHub 仓库

当前工作目录已连接私人仓库 `zeta172417-design/miniqwen-omni`，默认分支为 `main`，不再配置 MiniMind-O remote。V0 对应第一次上传的代码；V0.1 对应 Main codec head + Code Predictor、冻结 SenseVoice/SigLIP2 encoder 的正式基线。

发布后续代码版本时执行：

```bash
cd /path/to/miniqwen-omni
git add -A
git status --short

# 检查暂存区中不存在大文件。
git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(parquet|safetensors|bin|pth|pt|ckpt|onnx|mp3|wav|jpg|jpeg|png)$' && \
  echo 'ERROR: binary artifact staged' || echo 'OK: code-only staging area'

git commit -m "Describe the change"
git push -u origin main
```

创建新版本 tag 时：

```bash
git tag -a v0.2 -m "MiniQwen-Omni V0.2"
git push origin v0.2
```

训练数据、权重、日志、生成媒体和本地密钥均由 `.gitignore` 排除。不要使用 `git add -f` 强制添加这些内容。

## 2. 导出 ModelScope 模型

不要直接转换或覆盖训练 checkpoint。生成独立 BF16 目录：

```bash
cd /path/to/miniqwen-omni
source envs/Omni-ppu/bin/activate

python scripts/export_modelscope.py \
  --checkpoint out/miniqwen_omni_full_main_codec_cp_v5/checkpoint \
  --output releases/miniqwen-omni-bf16
```

检查导出内容：

```bash
find releases/miniqwen-omni-bf16 -maxdepth 1 -type f -printf '%f\n' | sort
grep -E '"model_type"|"dtype"|"num_talker_hidden_layers"' \
  releases/miniqwen-omni-bf16/config.json
```

## 3. 创建并上传私人 ModelScope 模型

从 ModelScope 个人设置页面创建 Access Token。本机安装的 ModelScope 1.37.0 要求通过 `--token` 登录；使用静默输入，避免 token 明文进入 shell history：

```bash
read -rsp 'ModelScope access token: ' MINIQWEN_MODELSCOPE_TOKEN
echo
modelscope login --token "$MINIQWEN_MODELSCOPE_TOKEN"
unset MINIQWEN_MODELSCOPE_TOKEN
```

V0 保留在 `peachPPP/MiniQwen-Omni`，不得用后续版本覆盖。V0.1 使用独立仓库
`peachPPP/MiniQwen-Omni-V0.1`：

```bash
modelscope create peachPPP/MiniQwen-Omni-V0.1 \
  --repo-type model \
  --visibility private \
  --license 'Apache License 2.0' \
  --description 'Qwen3-0.6B based text/audio/vision Omni model' \
  --exist-ok

modelscope upload peachPPP/MiniQwen-Omni-V0.1 \
  releases/miniqwen-omni-bf16 \
  --repo-type model \
  --commit-message 'Release MiniQwen-Omni V0.1 frozen-encoder BF16 checkpoint' \
  --max-workers 4
```

ModelScope 1.37.0 的 `upload` 子命令没有 `--use-cache` 或 `--sync` 参数，不要添加这两个选项。上传中断时，在确认 repo ID 正确后重新执行同一条 `modelscope upload` 命令。

## 4. 下载验证

下载到新目录，避免误用本地原文件：

```bash
modelscope download --model peachPPP/MiniQwen-Omni-V0.1 \
  --local_dir /tmp/miniqwen-omni-modelscope-check
```

加载配置、tokenizer 和核心权重：

```bash
python - <<'PY'
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

path = '/tmp/miniqwen-omni-modelscope-check'
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype='auto',
    audio_encoder_path=None,
    vision_model_path=None,
)
print(
    cfg.model_type,
    cfg.num_talker_hidden_layers,
    cfg.hidden_size,
    cfg.talker_hidden_size,
    len(tok),
    sorted({str(param.dtype) for param in model.parameters()}),
)
PY
```

完整语音/图像推理仍需要单独准备 SenseVoice-Small、SigLIP2 和 Mimi；这些冻结模型不会打包到 MiniQwen-Omni 核心仓库。
