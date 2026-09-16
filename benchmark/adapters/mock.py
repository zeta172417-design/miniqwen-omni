from __future__ import annotations

from benchmark.adapters.base import ModelAdapter
from benchmark.schema import BenchmarkSample, GenerationResult


class MockAdapter(ModelAdapter):
    """Deterministic adapter used by CI and scoring smoke tests."""

    def load(self) -> None:
        return None

    def generate(self, sample: BenchmarkSample, seed: int) -> GenerationResult:
        if sample.metric == "mcq":
            text = str(sample.reference)
        elif isinstance(sample.reference, list):
            text = str(sample.reference[0])
        elif isinstance(sample.reference, dict):
            text = ""
        else:
            text = str(sample.reference)
        return GenerationResult(sample.sample_id, self.model_id, generated_text=text, latency_seconds=0.001)
