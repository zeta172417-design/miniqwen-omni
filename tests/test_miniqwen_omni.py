import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.omni_dataset import OmniDataset
from model.model_omni import MiniQwenOmni, OmniConfig, TalkerModule
from trainer.trainer_utils import configure_token_ids, load_omni_tokenizer
from trainer.train_sft_omni import build_talker_attention_mask, init_swanlab_tracking, trim_batch_to_active_length


QWEN_PATH = "model/Qwen3-0.6B"


class MiniQwenOmniConfigTest(unittest.TestCase):
    def test_parquet_streaming_loader_preserves_nested_audio_columns(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        rows = {
            'conversations': [
                json.dumps([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]),
                json.dumps([{'role': 'user', 'content': 'c'}, {'role': 'assistant', 'content': 'd'}]),
            ],
            'answer_audios': [[[1, 2, 3]], [[4, 5, 6]]],
            'question_audios': [[b'audio-a'], [b'audio-b']],
            'ref_audios': [[7, 8], [9, 10]],
            'spk_emb': [[0.1, 0.2], [0.3, 0.4]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, 'nested.parquet')
            pq.write_table(pa.table(rows), parquet_path)
            with patch.dict(os.environ, {
                'MINIQWEN_DATASET_CACHE': os.path.join(tmp, 'cache'),
                'MINIQWEN_PARQUET_BATCH_SIZE': '1',
            }):
                dataset = OmniDataset(parquet_path, tokenizer, max_length=32, scheduled_sampling=0)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.dataset[1]['question_audios'], [b'audio-b'])
            self.assertEqual(dataset.dataset[1]['answer_audios'], [[4, 5, 6]])

    @staticmethod
    def _tracking_args():
        return SimpleNamespace(
            swanlab_project='MiniQwen-Omni-Full',
            swanlab_run_name='01-t2a-warmup',
            epochs=6,
            batch_size=24,
            qwen_learning_rate=1e-5,
            omni_learning_rate=5e-4,
            stage_id=1,
        )

    def test_swanlab_checkpoint_resume_is_not_must(self):
        calls = []
        fake = SimpleNamespace(
            init=lambda **kwargs: calls.append(kwargs),
            finish=lambda: None,
        )
        # Current production checkpoint predates the CLI naming cleanup.
        checkpoint = {'wandb_id': 'old-run', 'epoch': 2, 'step': 0}
        with patch.dict('sys.modules', {'swanlab': fake}):
            self.assertIs(init_swanlab_tracking(self._tracking_args(), checkpoint), fake)
        self.assertEqual(calls[0]['id'], 'old-run')
        self.assertEqual(calls[0]['resume'], 'allow')

    def test_swanlab_resume_failure_starts_new_segment(self):
        calls = []

        def init(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError('stale run')

        fake = SimpleNamespace(init=init, finish=lambda: None)
        checkpoint = {'swanlab_id': 'old-run', 'epoch': 2, 'step': 0}
        with patch.dict('sys.modules', {'swanlab': fake}):
            self.assertIs(init_swanlab_tracking(self._tracking_args(), checkpoint), fake)
        self.assertEqual(len(calls), 2)
        self.assertNotIn('id', calls[1])
        self.assertEqual(calls[1]['resume'], 'never')
        self.assertEqual(calls[1]['name'], '01-t2a-warmup-resume-e3')

    def test_talker_fusion_scales_have_deterministic_missing_key_init(self):
        config = SimpleNamespace(
            talker_hidden_size=32,
            hidden_size=48,
            use_moe=False,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            rope_scaling=None,
            audio_vocab_size=64,
            num_talker_hidden_layers=1,
            spk_emb_size=16,
            initializer_range=0.02,
        )
        talker = TalkerModule(config)
        talker.text_scale.data.zero_()
        talker.audio_scale.data.zero_()
        owner = SimpleNamespace(config=config)
        MiniQwenOmni._init_weights(owner, talker)
        self.assertEqual(talker.text_scale.item(), 3.0)
        self.assertEqual(talker.audio_scale.item(), 1.0)

    def test_talker_defaults_and_ablations(self):
        default = OmniConfig.from_qwen_pretrained(QWEN_PATH)
        self.assertEqual(default.num_talker_hidden_layers, 6)
        self.assertEqual(default.talker_hidden_size, 768)
        self.assertEqual(default.accept_hidden_layer, 14)
        for layers in (4, 6, 8):
            config = OmniConfig.from_qwen_pretrained(QWEN_PATH, num_talker_hidden_layers=layers)
            self.assertEqual(config.num_talker_hidden_layers, layers)
            self.assertEqual(config.accept_hidden_layer, 14)

    def test_qwen_token_ids_and_assistant_labels(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        config = OmniConfig.from_qwen_pretrained(QWEN_PATH)
        configure_token_ids(config, tokenizer)
        self.assertEqual(len(config.audio_ids), 1)
        self.assertEqual(len(config.image_ids), 1)
        self.assertLess(config.audio_ids[0], config.vocab_size)

        dataset = object.__new__(OmniDataset)
        dataset.max_length = 128
        dataset.assistant_start_ids = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        dataset.assistant_end_ids = tokenizer.encode('<|im_end|>', add_special_tokens=False)
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'world'}],
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        labels, ranges = dataset.generate_text_labels(input_ids)
        self.assertEqual(len(ranges), 1)
        self.assertGreater(sum(label != -100 for label in labels), 0)
        self.assertEqual(labels[ranges[0][1]], tokenizer.eos_token_id)

    def test_long_prompt_keeps_assistant_and_complete_modality_runs(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        dataset = object.__new__(OmniDataset)
        dataset.max_length = 128
        dataset.audio_token_id = tokenizer.convert_tokens_to_ids('<|audio_pad|>')
        dataset.image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        dataset.assistant_start_ids = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        dataset.assistant_end_ids = tokenizer.encode('<|im_end|>', add_special_tokens=False)
        prompt = tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': ('long context ' * 300) + '<|image_pad|>' * 8},
                {'role': 'assistant', 'content': 'short supervised answer'},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = tokenizer.encode(prompt, add_special_tokens=False)
        truncated = dataset.truncate_for_training(full_ids)
        labels, ranges = dataset.generate_text_labels(truncated)
        self.assertEqual(len(truncated), dataset.max_length)
        self.assertEqual(len(ranges), 1)
        self.assertGreater(sum(label != -100 for label in labels), 0)
        self.assertEqual(truncated.count(dataset.image_token_id), 8)

    def test_scheduled_sampling_never_creates_qwen_special_tokens(self):
        import torch

        tokenizer = load_omni_tokenizer(QWEN_PATH)
        dataset = object.__new__(OmniDataset)
        dataset.scheduled_sampling_prob = 1.0
        dataset.base_text_vocab_size = tokenizer.vocab_size
        dataset.base_audio_vocab_size = 2048
        dataset.image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        input_ids = torch.zeros(9, 4096, dtype=torch.long)
        text_labels = torch.zeros(4096, dtype=torch.long)
        audio_labels = torch.zeros(8, 4096, dtype=torch.long)
        sampled = dataset.apply_scheduled_sampling(input_ids, audio_labels, text_labels)
        self.assertLess(int(sampled[8].max()), tokenizer.vocab_size)
        self.assertLess(int(sampled[:8].max()), 2048)
        self.assertFalse((sampled[8] == dataset.image_token_id).any())

    def test_orphan_image_marker_is_sanitized(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        dataset = object.__new__(OmniDataset)
        dataset.image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        dataset.image_token_len = 64
        dataset.safe_text_token_id = tokenizer.encode(' ', add_special_tokens=False)[0]
        cleaned, valid = dataset.sanitize_image_markers([1, dataset.image_token_id, 2], has_real_image=False)
        self.assertFalse(valid)
        self.assertNotIn(dataset.image_token_id, cleaned)

    def test_talker_mask_extends_through_audio_targets(self):
        import torch

        thinker_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ])
        audio_labels = torch.full((2, 8, 8), -100, dtype=torch.long)
        audio_labels[0, 0, 3:7] = 42
        audio_labels[0, 7, 6:8] = 84
        talker_mask = build_talker_attention_mask(thinker_mask, audio_labels)
        self.assertTrue(torch.equal(talker_mask[0], torch.ones(8, dtype=torch.long)))
        self.assertTrue(torch.equal(talker_mask[1], thinker_mask[1]))
        # The Thinker mask remains unchanged and is not reused by Talker.
        self.assertTrue(torch.equal(thinker_mask[0], torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])))

    def test_dynamic_padding_preserves_every_active_position(self):
        import torch

        input_ids = torch.arange(2 * 9 * 32).reshape(2, 9, 32)
        attention_mask = torch.zeros(2, 32, dtype=torch.long)
        labels = torch.full((2, 32), -100, dtype=torch.long)
        audio_labels = torch.full((2, 8, 32), -100, dtype=torch.long)
        attention_mask[0, :11] = 1
        labels[1, 14] = 7
        audio_labels[0, 7, 18] = 42

        trimmed = trim_batch_to_active_length(
            input_ids, attention_mask, labels, audio_labels, pad_to_multiple=8
        )
        self.assertEqual(trimmed[0].shape, (2, 9, 24))
        self.assertTrue(torch.equal(trimmed[0], input_ids[..., :24]))
        self.assertTrue(torch.equal(trimmed[1], attention_mask[..., :24]))
        self.assertTrue(torch.equal(trimmed[2], labels[..., :24]))
        self.assertTrue(torch.equal(trimmed[3], audio_labels[..., :24]))

    def test_masked_vocab_projection_matches_ignored_cross_entropy(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(7)
        hidden_full = torch.randn(2, 7, 11, requires_grad=True)
        weight_full = torch.randn(23, 11, requires_grad=True)
        labels = torch.full((2, 7), -100, dtype=torch.long)
        labels[0, 2:5] = torch.tensor([1, 4, 9])
        labels[1, 5:] = torch.tensor([2, 3])
        full_loss = F.cross_entropy(hidden_full.matmul(weight_full.t()).reshape(-1, 23), labels.reshape(-1))
        full_loss.backward()

        hidden_masked = hidden_full.detach().clone().requires_grad_(True)
        weight_masked = weight_full.detach().clone().requires_grad_(True)
        mask = labels.ne(-100)
        masked_loss = F.cross_entropy(hidden_masked[mask].matmul(weight_masked.t()), labels[mask])
        masked_loss.backward()

        self.assertTrue(torch.allclose(full_loss, masked_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(hidden_full.grad, hidden_masked.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(weight_full.grad, weight_masked.grad, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
