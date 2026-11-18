import unittest
from types import SimpleNamespace
import torch

from trainer.train_pretrain import objective
from trainer.train_sft import lr_factor, parameter_groups
from model.model_omni import MimiCodecEmbedding


class SFTTrainingTest(unittest.TestCase):
    def test_special_only_audio_inputs_keep_residual_gradients_present(self):
        embedding = MimiCodecEmbedding(2112, 2048, 4)
        codes = torch.full((2, 8, 5), 2049)
        expected = embedding.main(codes[:, 0]).detach()
        actual = embedding(codes)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        for residual in embedding.residual:
            self.assertIsNotNone(residual.weight.grad)
            self.assertEqual(int(residual.weight.grad.count_nonzero()), 0)

    def test_window_normalization_matches_concatenated_token_gradient(self):
        config = SimpleNamespace(residual_codec_loss_weight=0.3)
        first = torch.tensor([[0.1, 0.4]], requires_grad=True)
        second = torch.tensor([[1.0, 0.1], [0.2, 0.3], [0.1, 0.7]], requires_grad=True)
        targets = [torch.tensor([[1]]), torch.tensor([[0, 1, 1]])]
        counts = torch.tensor([4., 0., 0.], dtype=torch.float64)
        combined = 0
        for logits, labels in zip((first, second), targets):
            result = SimpleNamespace(logits=logits, audio_logits=None, aux_loss=logits.sum() * 0)
            audio = torch.full((1, 8, labels.shape[1]), -100)
            combined = combined + objective(result, labels, audio, config, 1,
                                            normalization_counts=counts)
        expected = torch.nn.functional.cross_entropy(torch.cat((first, second)),
                                                     torch.cat([value.flatten() for value in targets]))
        actual_grads = torch.autograd.grad(combined, (first, second), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (first, second))
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference)

    def test_speaker_group_and_tied_vocabulary_parameters_are_unique(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(3, 4)
        model.lm_head = torch.nn.Linear(4, 3, bias=False)
        model.lm_head.weight = model.model.embed_tokens.weight
        model.talker = torch.nn.Module()
        model.talker.spk_proj = torch.nn.Linear(2, 4, bias=False)
        model.talker.proj = torch.nn.Linear(4, 4)
        rates = {'thinker': 1e-6, 'omni': 1e-5, 'speaker': 2e-5}
        groups = parameter_groups(model, {'learning_rates': rates})
        parameters = [parameter for group in groups for parameter in group['params']]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        by_name = {group['name']: group for group in groups}
        self.assertIs(by_name['speaker']['params'][0], model.talker.spk_proj.weight)
        self.assertEqual(by_name['speaker']['lr'], 2e-5)
        self.assertEqual(len(by_name['thinker']['params']), 1)

    def test_warmup_and_final_learning_rate(self):
        self.assertAlmostEqual(lr_factor(1, 1000, .03), 1 / 30)
        self.assertAlmostEqual(lr_factor(30, 1000, .03), 1)
        self.assertAlmostEqual(lr_factor(1000, 1000, .03), .1)
