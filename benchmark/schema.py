from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    task: str
    metric: str
    prompt: str
    reference: Any
    choices: list[str] = field(default_factory=list)
    image: str | None = None
    audio: str | None = None
    perturbation: str = "clean"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any], root: Path | None = None) -> "BenchmarkSample":
        values = {key: row.get(key) for key in cls.__dataclass_fields__}
        values["choices"] = values["choices"] or []
        values["metadata"] = values["metadata"] or {}
        values["perturbation"] = values["perturbation"] or "clean"
        values["source"] = values["source"] or ""
        if root:
            for key in ("image", "audio"):
                value = values.get(key)
                if value and not Path(value).is_absolute():
                    values[key] = str((root / value).resolve())
        return cls(**values)


@dataclass
class GenerationResult:
    sample_id: str
    model_id: str
    generated_text: str = ""
    generated_audio: str | None = None
    latency_seconds: float | None = None
    first_token_seconds: float | None = None
    first_audio_seconds: float | None = None
    peak_memory_gb: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "ok"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
