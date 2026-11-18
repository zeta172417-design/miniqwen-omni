# 数据目录

数据文件不提交 Git，按用途分为三层：

- `raw/`：原始 Parquet、开源语料、媒体文件、来源清单和划分索引。
- `pretrain/`：MiniQwen-Omni-V1-Base 的三类对齐池：`AIA`（音频输入）、`IIA`（视觉输入）、`AOA`（语音输出）。
- `sft/`：MiniQwen-Omni-V1 使用的混合 SFT Parquet 池及其媒体索引。

训练时只读取处理后的 Parquet；下载、清洗、编码、切分和验收代码统一放在 `data_pipeline/`。运行时读取器位于 `dataset/omni_dataset.py`、`dataset/sft_dataset.py` 和 `dataset/parquet_reader.py`。

新实例恢复时，优先运行 `scripts/download_modelscope_release.sh`，直接获取已验收的 Pretrain 和 SFT 数据。原始公开数据集只在需要重建数据时按 `README.md` 和 `PUBLISHING.md` 中的来源链接下载。
