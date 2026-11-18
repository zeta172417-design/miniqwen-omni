# 发布指南

代码仓库和模型仓库应分开发布：GitHub 只放源码，ModelScope 只上传导出的 BF16 推理模型。不要上传 `out/**/checkpoint/trainer_state.pt`，其中包含 optimizer、随机数状态和 SwanLab run 信息。

## 1. 发布到私人 GitHub 仓库

当前工作目录已连接私人仓库 `zeta172417-design/miniqwen-omni`，默认分支为 `main`，不再配置 MiniMind-O remote。MiniQwen-Omni-V1-Base 对应 Pretrain checkpoint；MiniQwen-Omni-V1 对应完成 SFT 的发布模型。两者均使用 Main codec head + Code Predictor，并冻结 SenseVoice/SigLIP2 encoder。

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
git tag -a v1.0 -m "MiniQwen-Omni V1.0"
git push origin v1.0
```

训练数据、权重、日志、生成媒体和本地密钥均由 `.gitignore` 排除。不要使用 `git add -f` 强制添加这些内容。

## 2. 导出 ModelScope 模型

不要直接转换或覆盖训练 checkpoint。生成独立 BF16 目录：

```bash
cd /path/to/miniqwen-omni
source .venv/bin/activate

python scripts/export_modelscope.py \
  --checkpoint out/miniqwen_sft/last/checkpoint \
  --output releases/MiniQwen-Omni-V1 \
  --model-name MiniQwen-Omni-V1 \
  --repo-id peachPPP/MiniQwen-Omni-V1

python scripts/export_modelscope.py \
  --checkpoint out/miniqwen_pretrain/tts/last/checkpoint \
  --output releases/MiniQwen-Omni-V1-Base \
  --model-name MiniQwen-Omni-V1-Base \
  --repo-id peachPPP/MiniQwen-Omni-V1-Base
```

检查导出内容：

```bash
find releases/MiniQwen-Omni-V1 -maxdepth 1 -type f -printf '%f\n' | sort
grep -E '"model_type"|"dtype"|"num_talker_hidden_layers"' \
  releases/MiniQwen-Omni-V1/config.json
```

## 3. 创建并上传私人 ModelScope 模型

从 ModelScope 个人设置页面创建 Access Token。本实例安装的 ModelScope 1.40.2 支持通过 `--token` 登录；使用静默输入，避免 token 明文进入 shell history：

```bash
read -rsp 'ModelScope access token: ' MINIQWEN_MODELSCOPE_TOKEN
echo
modelscope login --token "$MINIQWEN_MODELSCOPE_TOKEN"
unset MINIQWEN_MODELSCOPE_TOKEN
```

使用独立仓库 `peachPPP/MiniQwen-Omni-V1` 发布最终模型；如需发布 Pretrain 基座，使用 `peachPPP/MiniQwen-Omni-V1-Base`：

```bash
modelscope create peachPPP/MiniQwen-Omni-V1 \
  --repo_type model \
  --visibility private \
  --license 'Apache License 2.0' \
  --description 'Qwen3-0.6B based text/audio/vision Omni model' \
  --exist_ok

modelscope upload peachPPP/MiniQwen-Omni-V1 \
  releases/MiniQwen-Omni-V1 \
  --repo-type model \
  --commit-message 'Release MiniQwen-Omni MiniQwen-Omni-V1 BF16 checkpoint' \
  --max-workers 4

modelscope create peachPPP/MiniQwen-Omni-V1-Base \
  --repo_type model \
  --visibility private \
  --license 'Apache License 2.0' \
  --description 'MiniQwen-Omni V1 Base pretrain checkpoint' \
  --exist_ok

modelscope upload peachPPP/MiniQwen-Omni-V1-Base \
  releases/MiniQwen-Omni-V1-Base \
  --repo-type model \
  --commit-message 'Release MiniQwen-Omni V1 Base BF16 checkpoint' \
  --max-workers 4
```

ModelScope 1.40.2 的 `create` 使用下划线参数 `--repo_type` / `--exist_ok`，而
`upload` 使用连字符参数 `--repo-type`。`upload` 没有 `--use-cache` 或 `--sync`
参数，不要添加。上传中断时，在确认 repo ID 正确后重新执行同一条命令。

## 4. 下载验证

下载到新目录，避免误用本地原文件：

```bash
modelscope download --model peachPPP/MiniQwen-Omni-V1 \
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

## 5. 发布训练数据

Pretrain 和 SFT 应作为独立的 ModelScope **数据集仓库**发布，不要和模型权重混在同一个仓库中。当前处理后的数据规模如下：

| 数据集 | Parquet 文件 | 样本量/大小 | 发布建议 |
|---|---:|---:|---|
| `dataset/pretrain` | 323 | 约 66.05 GiB | 可直接作为 Parquet 数据集上传；音频和 Mimi 编码已内嵌 |
| `dataset/sft` | 190 | 约 18.77 GiB | 需要同时打包 `dataset/raw/external` 中被引用的媒体，并保留相对路径 |

上传前先运行只读审计：

```bash
python scripts/audit_modelscope_upload.py \
  dataset/pretrain dataset/sft \
  --output dataset/modelscope-upload-audit.json
```

Pretrain 审计应显示 `READY`。SFT 如果显示 `MEDIA_PATHS_REQUIRE_PACKAGING`，说明 Parquet 中仍有 `dataset/raw/...` 媒体引用；发布包应使用如下结构，不能直接上传只含 Parquet 的目录：

在上传前生成两个下载即用的 staging 目录。脚本默认使用硬链接，不会再次占用一份 85 GiB；如果 staging 在不同文件系统，会自动退化为复制：

```bash
python scripts/stage_modelscope_datasets.py \
  --pretrain-output /tmp/MiniQwen-Omni-V1-Pretrain \
  --sft-output /tmp/MiniQwen-Omni-V1-SFT
```

```text
MiniQwen-Omni-V1-SFT/
├── README.md
├── manifest.json
├── parquet/
│   ├── main_2048/
│   └── long_text_3072/
└── media/
    ├── fusion/audio2/
    ├── fusion/images/
    └── ...
```

正式发布前还需要补一层数据卡，记录来源许可证、字段 schema、训练/开发划分、媒体相对路径规则、清洗策略和已知限制。原始下载压缩包、缓存、绝对路径、训练日志和 checkpoint optimizer 状态不应进入数据集仓库。

创建数据集仓库的命令示例：

```bash
modelscope create peachPPP/MiniQwen-Omni-V1-Pretrain \
  --repo_type dataset --visibility private \
  --description 'Prepared AIA, IIA and AOA Parquet data for MiniQwen-Omni-V1-Base' \
  --exist_ok

modelscope create peachPPP/MiniQwen-Omni-V1-SFT \
  --repo_type dataset --visibility private \
  --description 'Prepared mixed SFT data for MiniQwen-Omni-V1' \
  --exist_ok
```

数据集目录准备好后执行：

```bash
modelscope upload peachPPP/MiniQwen-Omni-V1-Pretrain \
  /tmp/MiniQwen-Omni-V1-Pretrain --repo-type dataset \
  --commit-message 'Prepared V1 Base pretrain data' --max-workers 4

modelscope upload peachPPP/MiniQwen-Omni-V1-SFT \
  /tmp/MiniQwen-Omni-V1-SFT --repo-type dataset \
  --commit-message 'Prepared V1 SFT data and referenced media' --max-workers 4
```

建议先上传一个小型 smoke 子集，下载并用项目 reader 验证后，再上传完整约 85 GiB 数据。

## 6. 下载即用

发布后的用户只需要在项目根目录执行：

```bash
bash scripts/download_modelscope_release.sh
python scripts/validate_downloaded_release.py
```

脚本会将数据下载到 `dataset/`，将两个训练好的模型直接下载到训练和推理默认路径：

```text
out/miniqwen_sft/last/checkpoint/
out/miniqwen_pretrain/tts/last/checkpoint/
```

下载后的 V1 可直接推理：

```bash
python eval_omni.py \
  --load_from out/miniqwen_sft/last/checkpoint \
  --mode -1 --max_samples 1
```

下载后的 V1-Base 可作为 SFT 初始化权重：

```bash
bash trainer/sft.sh --run \
  --checkpoint out/miniqwen_pretrain/tts/last/checkpoint
```

辅助模型 `Qwen3-0.6B`、SenseVoice、SigLIP2 和 Mimi 仍需按环境说明单独下载；它们不重复包含在 V1/V1-Base 核心权重仓库中。
