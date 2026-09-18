from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from benchmark.schema import BenchmarkSample, GenerationResult


class ModelAdapter(ABC):
    def __init__(self, model_id: str, output_dir: Path, **_: object):
        self.model_id = model_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, sample: BenchmarkSample, seed: int) -> GenerationResult: ...

    def generate_batch(self, samples: list[BenchmarkSample], seeds: list[int]) -> list[GenerationResult]:
        """Generate a batch, with a sequential fallback for third-party adapters."""
        return [self.generate(sample, seed) for sample, seed in zip(samples, seeds)]

    def close(self) -> None:
        return None
