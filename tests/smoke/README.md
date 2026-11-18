# Smoke tests

这里放需要真实模型/GPU 的短流程测试。它们只允许写入 `.runtime/`，不能创建正式 checkpoint，也不能替代 `pytest` 单元测试。

SFT 性能基准直接复用 `trainer/train_sft.py` 的训练循环。给定一份输出目录位于 `.runtime/` 的计划后，使用 `--benchmark-steps 8 --benchmark-warmup 2`；没有 Pretrain checkpoint 时可加 `--benchmark-from-qwen`。基准模式禁止 resume，自动关闭开发集评测、checkpoint 与 SwanLab，最多允许 100 次更新，结果写入 `benchmark.json`。它会执行真实的优化器步骤，但模型仅存在于内存中。

CPU 数据读取比较使用 `benchmark_data_loading.py`。
