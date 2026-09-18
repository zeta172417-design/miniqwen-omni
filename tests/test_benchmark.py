import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from benchmark.metrics import (
    aggregate, exact_match, extract_choice, extract_explicit_choice, extract_spoken_choice,
    multiple_choice_accuracy,
    naturalize_tts_text, normalize_asr_text, token_f1, vqa_accuracy, word_edit_similarity,
    word_error_rate, paired_fixed_omni_comparison,
)
from benchmark.adapters.mini_omni2 import NoValidSnacFrames, sanitize_snac_codes
from benchmark.adapters.qwen25_omni import (
    QWEN_AUDIO_SYSTEM_PROMPT, generated_tokens_only, qwen_conversation, talker_token_budget,
)
from benchmark.prepare_voice_clone import select_rows
from benchmark.schema import BenchmarkSample
from benchmark.validate_manifest import validate_manifest


class BenchmarkTests(unittest.TestCase):
    def test_tts_text_naturalizes_librispeech_uppercase(self):
        self.assertEqual(
            naturalize_tts_text("FOR SOME TIME AFTER THAT I REMEMBERED NOTHING DISTINCTLY"),
            "For some time after that I remembered nothing distinctly.",
        )
        self.assertEqual(naturalize_tts_text("Already natural!"), "Already natural!")

    def test_v2_standard_budget_focuses_on_omni_tasks(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "benchmark/configs/standard.json").read_text())
        counts = config["counts"]
        main_samples = (
            counts["arc_easy"] + counts["boolq"] + counts["piqa"]
            + counts["spoken_qa"] * 2
            + counts["librispeech_clean"] + counts["librispeech_other"]
            + counts["tts"] + counts["mmbench_en"] + counts["pope"]
            + counts["textvqa"] + counts["audio_vision"] + counts["nlvr2"]
        )
        self.assertEqual(main_samples, 752)
        self.assertEqual(counts["arc_easy"] + counts["boolq"] + counts["piqa"], 32)

    def test_sample_resolves_ordered_multi_image_paths(self):
        sample = BenchmarkSample.from_dict({
            "sample_id": "multi", "task": "multi_image", "metric": "mcq",
            "prompt": "x", "reference": "A", "images": ["left.jpg", "right.jpg"],
        }, Path("/tmp/manifest"))
        self.assertEqual(sample.image_paths, ["/tmp/manifest/left.jpg", "/tmp/manifest/right.jpg"])

    def test_unsupported_samples_are_na_not_failures(self):
        rows = [
            {
                "sample_id": "ok", "task": "asr_clean", "metric": "wer", "reference": "hello",
                "generated_text": "hello", "status": "ok", "scores": {"primary": 1.0},
                "metadata": {"capability": "asr"},
            },
            {
                "sample_id": "na", "task": "multi_image", "status": "unsupported", "scores": {},
                "metadata": {"capability": "cross_modal"},
            },
        ]
        metrics = aggregate(rows)
        # Fixed Omni has ten required tasks. A perfect ASR-clean result with
        # every other task absent therefore contributes 1/10 of the score.
        self.assertEqual(metrics["objective_score"], 10)
        self.assertEqual(metrics["available_omni_score"], 100)
        self.assertEqual(metrics["failure_rate"], 0)
        self.assertEqual(metrics["unsupported_samples"], 1)
        self.assertNotIn("multi_image", metrics["task_metrics"])

    def test_minimind_3o_is_registered_for_main_and_clone_benchmarks(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "benchmark/configs/models.json").read_text())
        entry = next(model for model in config["models"] if model["id"] == "minimind-3o-moe")
        self.assertEqual(entry["adapter"], "benchmark.adapters.minimind3o:MiniMind3OAdapter")
        self.assertTrue(entry["kwargs"]["return_audio"])

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

    def test_qwen_talker_budget_is_separate_and_bounded(self):
        self.assertEqual(talker_token_budget("spoken_qa_a2a"), 600)
        self.assertEqual(talker_token_budget("tts"), 1536)
        self.assertEqual(talker_token_budget("tts", ceiling=800), 800)

    def test_qwen_audio_conversation_uses_required_official_system_prompt(self):
        content = [{"type": "text", "text": "hello"}]
        audio_conversation = qwen_conversation(content, return_audio=True)
        self.assertEqual(audio_conversation[0]["role"], "system")
        self.assertEqual(audio_conversation[0]["content"][0]["text"], QWEN_AUDIO_SYSTEM_PROMPT)
        self.assertEqual(
            qwen_conversation(content, return_audio=False),
            [{"role": "user", "content": content}],
        )

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
        self.assertAlmostEqual(word_error_rate("a fast cat", "a slow cat"), 1 / 3)
        self.assertEqual(normalize_asr_text("A cat, and the dog."), "a cat and the dog")
        self.assertEqual(extract_choice("Answer: B", ["x", "y", "z"]), "B")
        self.assertEqual(multiple_choice_accuracy("A", "1", ["x", "y", "z"]), 1)
        self.assertGreaterEqual(word_edit_similarity("many unrelated extra words", "cat"), 0)
        self.assertEqual(vqa_accuracy("cat", ["cat", "cat", "cat", "dog"]), 1)

    def test_verbose_multiple_choice_answers_are_scored_semantically(self):
        choices = ["The cat is running", "The cat is lying on the floor", "The cat is jumping"]
        self.assertEqual(extract_choice("The cat is lying on the floor.", choices), "B")
        self.assertEqual(multiple_choice_accuracy("The cat is lying on the floor.", "B", choices), 1)
        self.assertIsNone(extract_explicit_choice("The cat is lying on the floor.", choices))
        self.assertEqual(extract_choice("After checking the image, B is correct.", choices), "B")
        self.assertEqual(extract_choice("I believe B.", choices), "B")
        self.assertEqual(extract_choice("There is no person in the image.", ["Yes", "No"]), "B")

    def test_choice_parser_does_not_treat_english_article_as_option(self):
        choices = ["solid", "liquid", "gas", "plasma"]
        self.assertIsNone(extract_choice('The statement "A proton has a positive charge" is true.', choices))
        self.assertIsNone(extract_choice('The title "A Piece of Art" is visible.', choices))
        self.assertIsNone(extract_choice("The candidates are A, B, C, and D.", choices))

    def test_textvqa_accepts_answer_inside_brief_explanation(self):
        refs = ["astros"] * 9 + ["houston astros"]
        self.assertEqual(vqa_accuracy('The team name on the jersey is "Astros".', refs), 1)

    def test_spoken_option_letter_accepts_single_word_asr_homophone_only(self):
        choices = ["one", "two", "three", "four"]
        self.assertEqual(extract_spoken_choice("see.", choices), "C")
        self.assertEqual(extract_spoken_choice("Bee", choices), "B")
        self.assertIsNone(extract_spoken_choice("I can see the image", choices))

    def test_manifest_validator_rejects_empty_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "sample_id": "broken", "task": "vision_textvqa", "metric": "vqa",
                "prompt": "read it", "reference": [],
                "metadata": {"max_new_tokens": 64},
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "empty reference"):
                validate_manifest(manifest)

    def test_failures_and_robustness_stay_in_denominator(self):
        rows = [
            {"sample_id": "clean", "task": "audio", "status": "ok", "scores": {"primary": 1.0}},
            {"sample_id": "noisy", "task": "robustness", "status": "ok", "scores": {"primary": 0.0}, "metadata": {"base_id": "clean"}},
            {"sample_id": "failed", "task": "text", "status": "error", "scores": {"primary": 0.0}},
        ]
        metrics = aggregate(rows)
        self.assertEqual(metrics["failure_rate"], 1 / 3)
        self.assertEqual(metrics["task_metrics"]["robustness"]["score"], 0)

    def test_paired_comparison_uses_fixed_omni_weights_and_excludes_text(self):
        left = [
            {"sample_id": "asr", "task": "asr_clean", "status": "ok", "scores": {"primary": 0.0}},
            {"sample_id": "text", "task": "text_regression", "status": "ok", "scores": {"primary": 0.0}},
        ]
        right = [
            {"sample_id": "asr", "task": "asr_clean", "status": "ok", "scores": {"primary": 1.0}},
            {"sample_id": "text", "task": "text_regression", "status": "ok", "scores": {"primary": 1.0}},
        ]
        comparison = paired_fixed_omni_comparison(left, right, rounds=20)
        self.assertAlmostEqual(comparison["delta"], 0.1)
        self.assertEqual(comparison["samples"], 1)

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

    def test_sharded_runner_merges_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text("".join(
                json.dumps({
                    "sample_id": f"sample-{index}", "task": "text", "metric": "em",
                    "prompt": "x", "reference": f"answer-{index}",
                }) + "\n"
                for index in range(5)
            ))
            config = tmp_path / "config.json"
            config.write_text(json.dumps({
                "run_id": "sharded", "manifest": str(manifest),
                "output_root": str(tmp_path / "results"),
                "models": [{"id": "mock", "adapter": "benchmark.adapters.mock:MockAdapter", "kwargs": {}}],
            }))
            for shard_index in range(2):
                subprocess.run([
                    sys.executable, "-m", "benchmark.run_benchmark", "--config", str(config),
                    "--model", "mock", "--num-shards", "2", "--shard-index", str(shard_index),
                    "--batch-size", "2",
                ], check=True)
            model_dir = tmp_path / "results/sharded/mock"
            subprocess.run([
                sys.executable, "-m", "benchmark.merge_shards", "--run-dir", str(model_dir),
                "--num-shards", "2",
            ], check=True)
            rows = [json.loads(line) for line in (model_dir / "per_sample.jsonl").read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], [f"sample-{index}" for index in range(5)])
            self.assertEqual(json.loads((model_dir / "metrics.json").read_text())["objective_score"], 100)


if __name__ == "__main__":
    unittest.main()
