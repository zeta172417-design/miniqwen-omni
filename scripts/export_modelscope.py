"""Export a resumable FP32 training checkpoint as a BF16 model repository."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.trainer_utils import get_omni_model_class, infer_omni_model_arch  # noqa: E402


MODEL_CARD = """---
frameworks:
  - PyTorch
tasks:
  - multi-modal-dialogue
license: apache-2.0
---

# MiniQwen-Omni

This is the **V0.1 frozen-encoder release**. MiniQwen-Omni uses Qwen3-0.6B as
its 28-layer, 1024-dimensional Thinker and a
6-layer, 768-dimensional Talker that predicts eight Mimi audio codebooks. The
default bridge consumes Thinker layer 14 and projects 1024-dimensional hidden
states to the Talker width. A Main codec head predicts c0 and a two-layer Code
Predictor autoregressively predicts c1-c7 within the same Mimi frame.

This repository contains the BF16 core model and tokenizer. Frozen auxiliary
models are intentionally external:

- SenseVoice-Small for audio encoding
- SigLIP2 base P32 256 for image encoding
- Mimi for audio-code decoding

Load custom model code with `trust_remote_code=True`. Attach the auxiliary
encoders with `MiniQwenOmni.load_sensevoice` and `MiniQwenOmni.load_vision` as
shown in the MiniQwen-Omni source repository. This is a research checkpoint;
evaluate safety, factuality, speech quality, and language coverage before use.

```python
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "peachPPP/MiniQwen-Omni-V0.1"
local_dir = snapshot_download(repo)
tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    trust_remote_code=True,
    dtype="auto",
    audio_encoder_path=None,
    vision_model_path=None,
)
```
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Export MiniQwen-Omni to a ModelScope-ready BF16 directory")
    parser.add_argument("--checkpoint", required=True, type=Path, help="FP32 HF training checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--max-shard-size", default="5GB", help="Transformers shard-size limit")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if checkpoint == output:
        raise ValueError("--output must differ from the resumable training checkpoint")
    required = [checkpoint / "config.json", checkpoint / "model.safetensors", checkpoint / "tokenizer_config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model_class = get_omni_model_class(infer_omni_model_arch(str(checkpoint)))
    model = model_class.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        audio_encoder_path=None,
        vision_model_path=None,
    )
    model.eval().to(dtype=torch.bfloat16)
    model.config.dtype = torch.bfloat16
    model.config.architectures = ["MiniQwenOmni"]

    # Some shared DSW images expose an unrelated/broken DeepSpeed install.
    # Accelerate imports it only to decide whether this plain nn.Module needs
    # unwrapping. This exporter never constructs a DeepSpeedEngine, so disable
    # that optional probe instead of making export depend on that installation.
    try:
        import accelerate.utils.other as accelerate_other
        accelerate_other.is_deepspeed_available = lambda: False
    except ImportError:
        pass
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output)
    (output / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    forbidden = [output / "trainer_state.pt", output / "optimizer.pt"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("training state leaked into the inference export")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "miniqwen-omni":
        raise RuntimeError(f"unexpected model_type: {config.get('model_type')}")
    if config.get("num_talker_hidden_layers") != 6:
        raise RuntimeError("exported checkpoint is not the expected 6-layer Talker")

    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "format": "MiniQwen-Omni ModelScope BF16 export",
        "source": "local FP32 MiniQwen-Omni training checkpoint",
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total_gib = sum(item["size"] for item in files) / 1024 ** 3
    print(f"ModelScope export ready: {output}")
    print(f"Files: {len(files)} | size before manifest: {total_gib:.2f} GiB | dtype: bfloat16")


if __name__ == "__main__":
    main()
