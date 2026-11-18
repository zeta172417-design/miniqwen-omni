# 数据目录

数据文件不提交 Git，按用途分为三层：

- `raw/`：原始 Parquet、开源语料、媒体文件、来源清单和划分索引。
- `pretrain/`：MiniQwen-Omni-V1-Base 的三类对齐池：`AIA`（音频输入）、`IIA`（视觉输入）、`AOA`（语音输出）。
- `sft/`：MiniQwen-Omni-V1 使用的混合 SFT Parquet 池及其媒体索引。

训练时只读取处理后的 Parquet；下载、清洗、编码、切分和验收代码统一放在 `data_pipeline/`。运行时读取器位于 `dataset/omni_dataset.py`、`dataset/sft_dataset.py` 和 `dataset/parquet_reader.py`。

当前计划文件：

```text
raw/manifests/pretrain/plan.json
raw/manifests/sft/plan.json
```

旧的三份主数据源和补充语音数据保留在 `raw/`，便于重新构建；不会在训练时直接读取压缩包或临时缓存。数据比例、来源和验收结果见 `reports/` 下的数据报告。
