# 发布指南

代码仓库和模型仓库应分开发布：GitHub 只放源码，ModelScope 只上传导出的 BF16 推理模型。不要上传 `out/**/checkpoint/trainer_state.pt`，其中包含 optimizer、随机数状态和 SwanLab run 信息。

## 1. 发布到私人 GitHub 仓库

当前目录继承了上游 MiniMind-O 的 Git 历史。为了确保旧历史中的二进制文件也不会进入你的私人仓库，不要直接推送当前 `master`。先在本地完成整理 commit，再用 `git archive` 生成无历史的纯代码仓库。

先在 GitHub 网页创建一个空的 private repository，例如 `YOUR_GITHUB_USER/miniqwen-omni`，不要自动添加 README、License 或 `.gitignore`。然后在本项目执行：

```bash
cd /mnt/workspace/zhaozetao/multimodel/miniqwen-omni

# 这三类文件在上游中曾被跟踪；只从 Git 索引移除，本地文件保留。
git rm -r --cached --ignore-unmatch dataset/eval_omni model/speaker model/vad

git add -A
git status --short

# 检查暂存区中不存在大文件。
git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(parquet|safetensors|bin|pth|pt|ckpt|onnx|mp3|wav|jpg|jpeg|png)$' && \
  echo 'ERROR: binary artifact staged' || echo 'OK: code-only staging area'

git commit -m "Prepare MiniQwen-Omni code release"

# 从当前快照创建一个没有上游历史的新仓库。
rm -rf /tmp/miniqwen-omni-code-release
mkdir -p /tmp/miniqwen-omni-code-release
git archive HEAD | tar -x -C /tmp/miniqwen-omni-code-release
cd /tmp/miniqwen-omni-code-release
git init -b main
git add .
git commit -m "Initial private release of MiniQwen-Omni"
git remote add origin git@github.com:YOUR_GITHUB_USER/miniqwen-omni.git
git push -u origin main
```

如果使用 HTTPS，将最后的 remote 地址改为：

```bash
git remote add origin https://github.com/YOUR_GITHUB_USER/miniqwen-omni.git
```

`git rm --cached` 只修改 Git 索引，不删除本地评测数据或辅助模型。`git archive` 只导出当前 commit 中的文件，因此新 GitHub 仓库不会携带上游历史。

## 2. 导出 ModelScope 模型

不要直接转换或覆盖训练 checkpoint。生成独立 BF16 目录：

```bash
cd /mnt/workspace/zhaozetao/multimodel/miniqwen-omni
source envs/Omni-ppu/bin/activate

python scripts/export_modelscope.py \
  --checkpoint out/miniqwen_omni_full/checkpoint \
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

将 `YOUR_MODELSCOPE_USER` 换成你的 ModelScope 用户名：

```bash
modelscope create YOUR_MODELSCOPE_USER/MiniQwen-Omni \
  --repo-type model \
  --visibility private \
  --license 'Apache License 2.0' \
  --description 'Qwen3-0.6B based text/audio/vision Omni model' \
  --exist-ok

modelscope upload YOUR_MODELSCOPE_USER/MiniQwen-Omni \
  releases/miniqwen-omni-bf16 \
  --repo-type model \
  --commit-message 'Upload MiniQwen-Omni BF16 checkpoint' \
  --max-workers 4
```

ModelScope 1.37.0 的 `upload` 子命令没有 `--use-cache` 或 `--sync` 参数，不要添加这两个选项。上传中断时，在确认 repo ID 正确后重新执行同一条 `modelscope upload` 命令。

## 4. 下载验证

下载到新目录，避免误用本地原文件：

```bash
modelscope download --model YOUR_MODELSCOPE_USER/MiniQwen-Omni \
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
