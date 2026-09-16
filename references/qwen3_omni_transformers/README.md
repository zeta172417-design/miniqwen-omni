# Qwen3-Omni architecture reference

This directory contains only the official Hugging Face Transformers source files for the Qwen3-Omni-MoE architecture. It does **not** contain model weights, tokenizer files, or datasets.

- Upstream repository: https://github.com/huggingface/transformers
- Upstream commit: `5474a55e920f358d8382f3ecd3377edca979baa1`
- Upstream path: `src/transformers/models/qwen3_omni_moe/`
- Retrieved: 2026-09-12

Files:

- `modeling_qwen3_omni_moe.py`: generated executable model implementation
- `modular_qwen3_omni_moe.py`: modular source used to generate the model implementation
- `configuration_qwen3_omni_moe.py`: architecture configuration classes
- `__init__.py`: upstream module exports

These files are retained as a read-only design reference for MiniQwen-Omni. Keep local implementation changes outside this directory so comparisons against the pinned upstream source remain reliable.

The copied source files retain their upstream copyright and Apache-2.0 license headers.
