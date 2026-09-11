import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from dataset.omni_dataset import OmniDataset
from model.model_omni import CodePredictor, MimiCodecEmbedding, MiniQwenOmni, MiniQwenMRoPERotaryEmbedding, OmniConfig, TalkerModule
from trainer.trainer_utils import configure_token_ids, load_omni_tokenizer
from trainer.train_sft_omni import build_talker_attention_mask, compute_audio_losses, configure_trainable_modules, init_swanlab_tracking, omni_collate_fn, trim_batch_to_active_length


QWEN_PATH = "model/Qwen3-0.6B"


class MiniQwenOmniConfigTest(unittest.TestCase):
    @staticmethod
    def _png_bytes(color):
        buffer = __import__('io').BytesIO()
        Image.new('RGB', (16, 16), color=color).save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def _fake_vision_processor(images, return_tensors='pt'):
        pixels = torch.stack([
            torch.full((3, 256, 256), float(index + 1))
            for index, _ in enumerate(images)
        ])
        return {'pixel_values': pixels}

    def test_arch_eval_module_policy_excludes_unused_speaker_projection(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)
                self.lm_head = torch.nn.Linear(2, 2)
                self.audio_proj = torch.nn.Linear(2, 2)
                self.vision_proj = torch.nn.Linear(2, 2)
                self.talker = torch.nn.Module()
                self.talker.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.talker.embed_proj = torch.nn.Linear(2, 2)
                self.talker.spk_proj = torch.nn.Linear(2, 2)

        model = DummyModel()
        configure_trainable_modules(model, 'talker_core')
        self.assertTrue(model.talker.layers[0].weight.requires_grad)
        self.assertTrue(model.talker.embed_proj.weight.requires_grad)
        self.assertFalse(model.talker.spk_proj.weight.requires_grad)
        self.assertFalse(model.model.weight.requires_grad)

        configure_trainable_modules(model, 'talker,audio_proj')
        self.assertTrue(model.talker.spk_proj.weight.requires_grad)
        self.assertTrue(model.audio_proj.weight.requires_grad)
        self.assertFalse(model.model.weight.requires_grad)

    def test_eval_audio_prompt_is_deterministic_and_has_no_transcript(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        dataset = object.__new__(OmniDataset)
        dataset.training = False
        dataset.audio_token = '<|audio_pad|>'
        dataset.tokenizer = tokenizer
        conversations = [
            {'role': 'user', 'content': 'this transcript must not leak'},
            {'role': 'assistant', 'content': 'answer'},
        ]
        first = dataset.create_chat_prompt(conversations, audio_features_length=3)
        second = dataset.create_chat_prompt(conversations, audio_features_length=3)
        self.assertEqual(first, second)
        self.assertNotIn('this transcript must not leak', first)
        self.assertEqual(first.count('<|audio_pad|>'), 3)

    def test_text_only_stream_generation_skips_talker(self):
        calls = []

        class TextOnlyModel:
            config = SimpleNamespace(pad_token_id=0, think_end_ids=[])

            def forward(self, input_ids, **kwargs):
                calls.append((input_ids.clone(), kwargs.copy()))
                logits = torch.full((1, input_ids.shape[-1], 8), -100.0)
                logits[0, -1, 3] = 100.0
                return SimpleNamespace(
                    logits=logits,
                    past_key_values=object(),
                    audio_logits=None,
                )

        generated = list(MiniQwenOmni.stream_generate(
            TextOnlyModel(),
            torch.tensor([[1, 2]], dtype=torch.long),
            eos_token_id=3,
            max_new_tokens=4,
            temperature=1.0,
            top_p=1.0,
            top_k=4,
            rp=1.0,
            use_cache=True,
            return_audio_codes=False,
        ))

        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0][0].tolist(), [[3]])
        self.assertIsNone(generated[0][1])
        self.assertEqual(calls[0][0].ndim, 2)
        self.assertFalse(calls[0][1]["output_audio_logits"])
        self.assertEqual(calls[0][1]["logits_to_keep"], 1)

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
        self.assertEqual(default.audio_head_type, 'main_codec_predictor')
        self.assertEqual(default.code_predictor_num_layers, 2)
        self.assertEqual(default.code_predictor_hidden_size, 768)
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

    def test_modality_special_tokens_preserve_qwen_controls_and_ids(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        config = OmniConfig.from_qwen_pretrained(QWEN_PATH)
        configure_token_ids(config, tokenizer)

        self.assertEqual(config.audio_ids, [151669])
        self.assertEqual(len(config.audio_start_ids), 1)
        self.assertEqual(len(config.audio_end_ids), 1)
        self.assertLess(len(tokenizer), config.vocab_size + 1)
        for token in ('<|im_start|>', '<|vision_start|>', '<|image_pad|>',
                      '<|audio_pad|>', '<|audio_start|>', '<|audio_end|>'):
            self.assertIn(token, tokenizer.all_special_tokens)

        with tempfile.TemporaryDirectory() as tmp:
            tokenizer.save_pretrained(tmp)
            reloaded = load_omni_tokenizer(tmp)
            for token in ('<|audio_pad|>', '<|audio_start|>', '<|audio_end|>'):
                self.assertEqual(
                    tokenizer.convert_tokens_to_ids(token),
                    reloaded.convert_tokens_to_ids(token),
                )

    def test_mrope_positions_for_text_and_multiple_images(self):
        config = SimpleNamespace(
            image_ids=[10], vision_start_ids=[11], vision_end_ids=[12],
            image_grid_size=8, image_token_len=64,
        )
        owner = SimpleNamespace(config=config)
        text_only = torch.tensor([[3, 4, 5, 0]])
        text_pos, text_delta = MiniQwenOmni.build_mrope_position_ids(
            owner, text_only, torch.tensor([[1, 1, 1, 0]])
        )
        self.assertTrue(torch.equal(text_pos[:, 0, :3], torch.arange(3).expand(3, -1)))
        self.assertEqual(text_delta.item(), 0)

        image_block = [11] + [10] * 64 + [12]
        tokens = torch.tensor([[3] + image_block + image_block + [4]])
        positions, delta = MiniQwenOmni.build_mrope_position_ids(owner, tokens)
        first_image = positions[:, 0, 2:66]
        self.assertTrue(torch.equal(first_image[0], torch.full((64,), 2)))
        self.assertTrue(torch.equal(first_image[1], torch.arange(8).repeat_interleave(8) + 2))
        self.assertTrue(torch.equal(first_image[2], torch.arange(8).repeat(8) + 2))
        self.assertEqual(positions[0, 0, -1].item(), 21)
        self.assertTrue(torch.equal(positions[:, 0, -1], torch.full((3,), 21)))
        self.assertEqual(delta.item(), -112)

    def test_mrope_cached_forward_matches_full_forward(self):
        config = OmniConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1,
            head_dim=64, max_position_embeddings=256,
            num_talker_hidden_layers=1, talker_hidden_size=64,
            audio_vocab_size=64, spk_emb_size=16, accept_hidden_layer=1,
            image_ids=[10], vision_start_ids=[11], vision_end_ids=[12],
            image_token_len=64, image_grid_size=8,
            use_mrope=True, mrope_section=[12, 10, 10],
        )
        model = MiniQwenOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        self.assertIsInstance(model.model.rotary_emb, MiniQwenMRoPERotaryEmbedding)
        prompt = torch.tensor([[3, 11] + [10] * 64 + [12]])
        next_token = torch.tensor([[4]])
        with torch.inference_mode():
            first = model(prompt, use_cache=True, output_audio_logits=False, logits_to_keep=1)
            cached = model(
                next_token,
                attention_mask=torch.ones(1, prompt.size(1) + 1, dtype=torch.long),
                past_key_values=first.past_key_values,
                use_cache=True,
                output_audio_logits=False,
                logits_to_keep=1,
            )
            full = model(
                torch.cat((prompt, next_token), dim=1),
                use_cache=False,
                output_audio_logits=False,
                logits_to_keep=1,
            )
        self.assertEqual(first.past_key_values.rope_deltas.item(), -56)
        self.assertTrue(torch.allclose(cached.logits, full.logits, atol=2e-5, rtol=2e-5))

    def test_multi_image_dataset_and_collate_keep_source_order(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        rows = {
            'conversations': [
                json.dumps([
                    {'role': 'user', 'content': '<image> first then <image> second'},
                    {'role': 'assistant', 'content': 'two images'},
                ]),
                json.dumps([
                    {'role': 'user', 'content': 'only <image>'},
                    {'role': 'assistant', 'content': 'one image'},
                ]),
            ],
            'image_bytes': [
                [self._png_bytes('red'), self._png_bytes('blue')],
                [self._png_bytes('green')],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, 'multi-image.parquet')
            pq.write_table(pa.table(rows), parquet_path)
            with patch.dict(os.environ, {'MINIQWEN_DATASET_CACHE': os.path.join(tmp, 'cache')}):
                dataset = OmniDataset(
                    parquet_path,
                    tokenizer,
                    vision_processor=self._fake_vision_processor,
                    max_length=256,
                    scheduled_sampling=0,
                    training=False,
                    use_modality_boundaries=True,
                    max_images=4,
                )
                samples = [dataset[0], dataset[1]]

        image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.assertEqual(samples[0][6].shape, (2, 3, 256, 256))
        self.assertEqual(int((samples[0][0][8] == image_id).sum()), 128)
        self.assertEqual(int((samples[0][0][8] == vision_start_id).sum()), 2)
        self.assertEqual(int((samples[0][0][8] == vision_end_id).sum()), 2)
        batch = omni_collate_fn(samples, dynamic_padding=False)
        self.assertEqual(batch[6].shape, (2, 2, 3, 256, 256))
        self.assertFalse(batch[6][1, 1].any())

    def test_multi_image_features_are_injected_into_matching_runs(self):
        owner = SimpleNamespace(config=SimpleNamespace(image_ids=[10]))
        tokens = torch.tensor([[10] * 64 + [7] + [10] * 64])
        hidden = torch.zeros(1, 129, 4)
        features = torch.stack((
            torch.full((64, 4), 1.0),
            torch.full((64, 4), 2.0),
        )).unsqueeze(0)
        injected = MiniQwenOmni.count_vision_proj(
            owner, tokens, hidden, vision_tensors=features, seqlen=129
        )
        self.assertTrue(torch.equal(injected[0, :64], features[0, 0]))
        self.assertTrue(torch.equal(injected[0, 65:], features[0, 1]))
        self.assertFalse(injected[0, 64].any())

    def test_fp32_vision_encoder_is_cast_for_bfloat16_projector(self):
        class FakeVisionEncoder(torch.nn.Module):
            def forward(self, pixel_values):
                batch = pixel_values.size(0)
                return SimpleNamespace(
                    last_hidden_state=torch.ones(batch, 64, 8, dtype=torch.float32)
                )

        owner = SimpleNamespace(
            vision_encoder=FakeVisionEncoder(),
            vision_proj=torch.nn.Sequential(
                torch.nn.LayerNorm(8),
                torch.nn.Linear(8, 16),
            ).to(torch.bfloat16),
            config=SimpleNamespace(image_token_len=64, hidden_size=16),
        )
        encoded = MiniQwenOmni.encode_image_inputs(
            owner, torch.ones(2, 3, 16, 16, dtype=torch.float32)
        )
        padded = MiniQwenOmni.encode_image_inputs(
            owner, torch.zeros(2, 3, 16, 16, dtype=torch.float32)
        )
        self.assertEqual(encoded.dtype, torch.bfloat16)
        self.assertEqual(encoded.shape, (2, 64, 16))
        self.assertEqual(padded.dtype, torch.bfloat16)
        self.assertFalse(padded.any())

    def test_talker_reference_and_speaker_have_explicit_boundaries(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        rows = {
            'conversations': [json.dumps([
                {'role': 'user', 'content': 'say hello'},
                {'role': 'assistant', 'content': 'hello'},
            ])],
            'answer_audios': [[[100 + i for i in range(16)]]],
            'ref_audios': [[10 + i for i in range(16)]],
            'spk_emb': [[0.1] * 192],
        }
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, 'talker-prefix.parquet')
            pq.write_table(pa.table(rows), parquet_path)
            with patch.dict(os.environ, {'MINIQWEN_DATASET_CACHE': os.path.join(tmp, 'cache')}):
                dataset = OmniDataset(
                    parquet_path,
                    tokenizer,
                    max_length=128,
                    scheduled_sampling=0,
                    training=False,
                    use_talker_ref_boundaries=True,
                )
                audio_inputs = dataset[0][0][:8]

        main_row = audio_inputs[0].tolist()
        ref_start = main_row.index(2052)
        self.assertEqual(main_row[ref_start - 1], 2051)
        self.assertEqual(main_row[ref_start + 1:ref_start + 3], [10, 18])
        self.assertEqual(main_row[ref_start + 3], 2053)
        for layer in range(1, 8):
            row = audio_inputs[layer].tolist()
            self.assertNotIn(2051, row)
            self.assertNotIn(2052, row)
            self.assertNotIn(2053, row)
            self.assertEqual(row[ref_start + 1:ref_start + 3], [10 + layer, 18 + layer])

    def test_main_codec_dataset_is_same_frame_bos_to_eos(self):
        tokenizer = load_omni_tokenizer(QWEN_PATH)
        rows = {
            'conversations': [json.dumps([
                {'role': 'user', 'content': 'say two frames'},
                {'role': 'assistant', 'content': 'hello'},
            ])],
            'answer_audios': [[[100 + index for index in range(16)]]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, 'same-frame.parquet')
            pq.write_table(pa.table(rows), parquet_path)
            with patch.dict(os.environ, {'MINIQWEN_DATASET_CACHE': os.path.join(tmp, 'cache')}):
                dataset = OmniDataset(
                    parquet_path, tokenizer, max_length=128,
                    scheduled_sampling=0, training=False,
                    audio_head_type='main_codec_predictor',
                )
                input_ids, _, _, labels, *_ = dataset[0]

        audio_inputs = input_ids[:8]
        bos_pos = audio_inputs[0].tolist().index(2048)
        self.assertTrue(torch.equal(audio_inputs[1:, bos_pos], torch.full((7,), 2049)))
        self.assertTrue(torch.equal(labels[:, bos_pos], torch.arange(100, 108)))
        self.assertTrue(torch.equal(audio_inputs[:, bos_pos + 1], torch.arange(100, 108)))
        self.assertTrue(torch.equal(labels[:, bos_pos + 1], torch.arange(108, 116)))
        self.assertTrue(torch.equal(audio_inputs[:, bos_pos + 2], torch.arange(108, 116)))
        self.assertEqual(labels[0, bos_pos + 2].item(), 2050)
        self.assertTrue(labels[1:, bos_pos + 2].eq(-100).all())

    def test_same_frame_embedding_uses_mean_and_residual_vocab_is_strict(self):
        embedding = MimiCodecEmbedding(2112, 2048, 4)
        with torch.no_grad():
            embedding.main.weight.zero_()
            for index, residual in enumerate(embedding.residual, start=1):
                residual.weight.fill_(float(index))
        frame = torch.zeros(1, 8, 1, dtype=torch.long)
        output = embedding(frame)
        self.assertTrue(torch.allclose(output, torch.full_like(output, 3.5)))
        self.assertEqual(embedding.residual[0].num_embeddings, 2048)
        with self.assertRaises((IndexError, RuntimeError)):
            embedding.codebook(1, torch.tensor([2050]))

    def test_code_predictor_is_causal_and_backpropagates_to_talker(self):
        config = OmniConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            talker_hidden_size=32, code_predictor_hidden_size=32,
            code_predictor_num_layers=2,
            code_predictor_num_attention_heads=4,
            code_predictor_num_key_value_heads=2,
            code_predictor_intermediate_size=64,
            audio_vocab_size=2112, audio_codebook_size=2048,
        )
        embedding = MimiCodecEmbedding(2112, 2048, 32)
        predictor = CodePredictor(config, embedding).eval()
        hidden = torch.randn(2, 32, requires_grad=True)
        frames = torch.randint(0, 2048, (2, 8))
        changed = frames.clone()
        changed[:, 2] = (changed[:, 2] + 1) % 2048
        logits = predictor(hidden, frames)
        changed_logits = predictor(hidden, changed)
        self.assertTrue(torch.allclose(logits[0], changed_logits[0]))
        self.assertTrue(torch.allclose(logits[1], changed_logits[1]))
        loss = sum(torch.nn.functional.cross_entropy(logit, frames[:, index + 1])
                   for index, logit in enumerate(logits)) / 7
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(hidden.grad.abs().sum()), 0)

    def test_code_predictor_cached_steps_match_teacher_forcing(self):
        config = OmniConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            talker_hidden_size=32, code_predictor_hidden_size=32,
            code_predictor_num_layers=2,
            code_predictor_num_attention_heads=4,
            code_predictor_num_key_value_heads=2,
            code_predictor_intermediate_size=64,
            audio_vocab_size=2112, audio_codebook_size=2048,
        )
        embedding = MimiCodecEmbedding(2112, 2048, 32)
        predictor = CodePredictor(config, embedding).eval()
        hidden = torch.randn(1, 32)
        frame = torch.randint(0, 2048, (1, 8))
        teacher_logits = predictor(hidden, frame)

        prefix = torch.cat((
            hidden.unsqueeze(1), embedding.codebook(0, frame[:, 0]).unsqueeze(1)
        ), dim=1)
        cached_hidden, cache = predictor._run(prefix, use_cache=True)
        self.assertTrue(torch.allclose(
            predictor.heads[0](cached_hidden[:, -1]), teacher_logits[0], atol=1e-5, rtol=1e-5
        ))
        for index in range(1, 7):
            next_embedding = embedding.codebook(index, frame[:, index]).unsqueeze(1)
            cached_hidden, cache = predictor._run(
                next_embedding, start_pos=index + 1,
                past_key_values=cache, use_cache=True,
            )
            self.assertTrue(torch.allclose(
                predictor.heads[index](cached_hidden[:, -1]), teacher_logits[index],
                atol=1e-5, rtol=1e-5,
            ))

    def test_tiny_same_frame_model_forward_shapes(self):
        config = OmniConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            num_talker_hidden_layers=1, talker_hidden_size=32,
            code_predictor_hidden_size=32, code_predictor_num_layers=1,
            code_predictor_num_attention_heads=4,
            code_predictor_num_key_value_heads=2,
            code_predictor_intermediate_size=64,
            audio_vocab_size=2112, audio_codebook_size=2048,
            spk_emb_size=16, accept_hidden_layer=1,
            use_mrope=False, audio_head_type='main_codec_predictor',
        )
        model = MiniQwenOmni(config, audio_encoder_path=None, vision_model_path=None)
        inputs = torch.full((1, 9, 5), 0, dtype=torch.long)
        inputs[:, :8] = 2049
        inputs[0, 0, 1] = 2048
        inputs[0, :8, 2] = torch.arange(8)
        targets = torch.full((1, 8, 5), -100, dtype=torch.long)
        targets[0, :, 1] = torch.arange(8)
        targets[0, 0, 2] = 2050
        result = model(
            inputs, attention_mask=torch.ones(1, 5, dtype=torch.long),
            talker_attention_mask=torch.ones(1, 5, dtype=torch.long),
            audio_targets=targets, output_audio_logits=True,
        )
        self.assertEqual(result.main_audio_logits.shape, (1, 5, 2112))
        self.assertEqual(len(result.residual_audio_logits), 7)
        self.assertEqual(result.residual_audio_logits[0].shape, (1, 2048))

    def test_same_frame_checkpoint_reload_and_direct_generation(self):
        config = OmniConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            num_talker_hidden_layers=1, talker_hidden_size=32,
            code_predictor_hidden_size=32, code_predictor_num_layers=1,
            code_predictor_num_attention_heads=4,
            code_predictor_num_key_value_heads=2,
            code_predictor_intermediate_size=64,
            audio_vocab_size=2112, audio_codebook_size=2048,
            spk_emb_size=16, accept_hidden_layer=1,
            use_mrope=False, audio_head_type='main_codec_predictor',
            eos_token_id=2, pad_token_id=0, think_end_ids=[],
        )
        model = MiniQwenOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        with torch.no_grad():
            model.talker.lm_head.weight.zero_()
        # Deterministic argmax makes the zeroed Main head choose real codec 0,
        # never EOS, so the second temporal step must emit one direct frame.
        model._sample_from_logits = lambda logits, temperature=1.0, top_k=0, top_p=1.0: int(logits.argmax())
        generated = list(model.stream_generate(
            torch.tensor([[1, 3]], dtype=torch.long), eos_token_id=2,
            max_new_tokens=2, temperature=1.0, top_p=1.0, top_k=1,
            rp=1.0, use_cache=True, return_audio_codes=True,
            newline_token_id=4, pad_token_id=0,
        ))
        frames = [frame for _, frame in generated if frame is not None]
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]), 8)
        self.assertTrue(all(0 <= code < 2048 for code in frames[0]))

        with tempfile.TemporaryDirectory() as tmp:
            # The DSW CPU-only test subprocess can make Accelerate import an
            # unrelated broken DeepSpeed CPU backend. Training checkpoints run
            # with PPU available; bypass only that wrapper in this unit test.
            with patch('transformers.modeling_utils.unwrap_model', lambda value, **_: value):
                model.save_pretrained(tmp, safe_serialization=True)
            reloaded = MiniQwenOmni.from_pretrained(
                tmp, audio_encoder_path=None, vision_model_path=None,
            )
        self.assertEqual(reloaded.config.audio_head_type, 'main_codec_predictor')
        self.assertIsNotNone(reloaded.talker.code_predictor)

    def test_main_codec_loss_formula_has_unweighted_eos(self):
        main_logits = torch.randn(1, 3, 2112, requires_grad=True)
        residual_logits = [torch.randn(2, 2048, requires_grad=True) for _ in range(7)]
        labels = torch.full((1, 8, 3), -100, dtype=torch.long)
        labels[:, :, :2] = torch.tensor([
            [10, 11], [20, 21], [30, 31], [40, 41],
            [50, 51], [60, 61], [70, 71], [80, 81],
        ])
        labels[0, 0, 2] = 2050
        result = SimpleNamespace(
            logits=torch.zeros(1, 1), audio_logits=main_logits,
            main_audio_logits=main_logits,
            residual_audio_logits=residual_logits,
            residual_audio_mask=torch.tensor([[True, True, False]]),
        )
        config = SimpleNamespace(
            audio_head_type='main_codec_predictor', audio_stop_token=2050,
            residual_codec_loss_weight=0.3,
        )
        objective, main_loss, residual_loss, _ = compute_audio_losses(result, labels, config)
        self.assertTrue(torch.allclose(objective, main_loss + 0.3 * residual_loss))
        expected_main = torch.nn.functional.cross_entropy(
            main_logits.reshape(-1, 2112), labels[:, 0].reshape(-1), ignore_index=-100
        )
        self.assertTrue(torch.allclose(main_loss, expected_main))

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
