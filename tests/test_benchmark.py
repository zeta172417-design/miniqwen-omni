import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from benchmark.metrics import (
    aggregate, exact_match, extract_choice, token_f1, vqa_accuracy, word_error_rate,
)
from benchmark.adapters.mini_omni2 import NoValidSnacFrames, sanitize_snac_codes
from benchmark.adapters.qwen25_omni import generated_tokens_only
from benchmark.prepare_voice_clone import select_rows


class BenchmarkTests(unittest.TestCase):
    def test_voice_clone_selection_uses_distinct_reference_and_targets(self):
        rows = [
            {
                "audio": {"bytes": b"audio"}, "text": f"four useful words {speaker} {index}",
                "speaker_id": speaker, "id": f"{speaker}-{index}",
            }
            for speaker in (1, 2, 3) for index in range(5)
        ]
        with patch("benchmark.prepare_voice_clone.duration", return_value=3.0):
            selected = select_rows(rows, speakers=2, targets=4, seed=7)
        self.assertEqual(len(selected), 2)
        for group in selected:
            self.assertEqual(len(group["targets"]), 4)
            self.assertNotIn(group["reference"]["id"], {row["id"] for row in group["targets"]})

    def test_qwen_generation_removes_prompt_tokens(self):
        import torch
        prompt = torch.tensor([[1, 2, 3]])
        generated = torch.tensor([[1, 2, 3, 8, 9]])
        self.assertEqual(generated_tokens_only(generated, prompt).tolist(), [[8, 9]])
        continuation_only = torch.tensor([[8, 9]])
        self.assertIs(generated_tokens_only(continuation_only, prompt), continuation_only)

    def test_snac_codes_stop_before_invalid_frame(self):
        import torch
        codes = [
            torch.tensor([[1, 2, 4096, 4]]),
            torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            torch.tensor([[1] * 16]),
        ]
        cleaned = sanitize_snac_codes(codes)
        self.assertEqual([x.shape[-1] for x in cleaned], [2, 4, 8])
        self.assertTrue(all(int(x.max()) < 4096 for x in cleaned))

    def test_snac_codes_reject_invalid_first_frame_without_device_decode(self):
        import torch
        codes = [torch.tensor([[4096]]), torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3, 4]])]
        with self.assertRaises(NoValidSnacFrames):
            sanitize_snac_codes(codes)

    def test_text_metrics(self):
        self.assertEqual(exact_match("The blue cat.", "blue cat"), 1)
        self.assertEqual(token_f1("small blue cat", "blue cat"), 0.8)
        self.assertEqual(word_error_rate("a fast cat", "a slow cat"), 0.5)
        self.assertEqual(extract_choice("Answer: B", ["x", "y", "z"]), "B")
        self.assertEqual(vqa_accuracy("cat", ["cat", "cat", "cat", "dog"]), 1)

    def test_failures_and_robustness_stay_in_denominator(self):
        rows = [
            {"sample_id": "clean", "task": "audio", "status": "ok", "scores": {"primary": 1.0}},
            {"sample_id": "noisy", "task": "robustness", "status": "ok", "scores": {"primary": 0.0}, "metadata": {"base_id": "clean"}},
            {"sample_id": "failed", "task": "text", "status": "error", "scores": {"primary": 0.0}},
        ]
        metrics = aggregate(rows)
        self.assertEqual(metrics["failure_rate"], 1 / 3)
        self.assertEqual(metrics["task_metrics"]["robustness"]["score"], 0)

    def test_mock_runner_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "sample_id": "one", "task": "text", "metric": "em", "prompt": "x", "reference": "answer",
                "metadata": {"base_id": "clean-one", "require_audio": False},
            }) + "\n")
            config = tmp_path / "config.json"
            config.write_text(json.dumps({
                "run_id": "test", "manifest": str(manifest), "output_root": str(tmp_path / "results"),
                "models": [{"id": "mock", "adapter": "benchmark.adapters.mock:MockAdapter", "kwargs": {}}]
            }))
            subprocess.run([sys.executable, "-m", "benchmark.run_benchmark", "--config", str(config), "--model", "mock"], check=True)
            run_dir = tmp_path / "results/test"
            subprocess.run([sys.executable, "-m", "benchmark.build_report", "--run-dir", str(run_dir)], check=True)
            self.assertTrue((run_dir / "REPORT.md").exists())
            metrics = json.loads((run_dir / "mock/metrics.json").read_text())
            self.assertEqual(metrics["objective_score"], 100)
            row = json.loads((run_dir / "mock/per_sample.jsonl").read_text())
            self.assertEqual(row["metadata"]["base_id"], "clean-one")
            self.assertFalse(row["metadata"]["require_audio"])


if __name__ == "__main__":
    unittest.main()
