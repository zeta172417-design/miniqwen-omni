# Offline data pipeline

这里存放数据下载、清洗、分组、媒体处理、Parquet/Mimi 编码和验收代码。它们不参与模型运行时，也不应被 `trainer/` 直接导入。最终训练只依赖已验收的 Parquet 和对应 manifest。

当前补充池：`prepare_visual_textvqa.py` 生成 TextVQA 图像问答 Parquet；`prepare_speech_mcq_kokoro.py` 和 `finalize_speech_mcq_kokoro.py` 生成并验收 Kokoro-82M A2A 音频及 Mimi 目标。旧 Flite 和模型自生成目标流程已废弃。

训练时使用的 Dataset、Parquet reader 和 collator 保留在 `dataset/`；训练循环保留在 `trainer/`。

当前主入口：

- `prepare_sft.py`：构建最终 SFT Parquet
- `reorganize_pretrain_data.py`：构建统一 Pretrain Parquet
- `validate_sft.py`、`validate_pretrain_*.py`：数据验收
- `run_pretrain_data.py`、`run_sft_data_preparation.py`：可恢复的批处理协调器
