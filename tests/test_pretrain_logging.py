import unittest
from types import SimpleNamespace
import torch
from trainer.train_pretrain import objective, training_metrics

class LoggingTest(unittest.TestCase):
    def test_uneven_token_weighting_and_partial_window(self):
        # Two uneven microbatches/ranks, with separate codec denominators.
        a=torch.tensor([6,2,10,2,12,3,0.4,1,2],dtype=torch.float64)
        b=torch.tensor([4,8,2,1,8,2,0.2,1,1],dtype=torch.float64)
        m=training_metrics(a+b,0.5)
        self.assertAlmostEqual(m['train/loss'],1+4+0.5*4+0.3)
        self.assertEqual(m['train/samples'],3)
        self.assertEqual(m['train/rank_microbatches'],2)
        self.assertAlmostEqual(training_metrics(a,0.5)['train/loss'],3+5+2+0.4)

    def test_logging_preserves_gradient_and_handles_no_audio(self):
        labels=torch.tensor([[1,0,-100]])
        audio=torch.full((1,8,3),-100)
        config=SimpleNamespace(residual_codec_loss_weight=0.5)
        logits=torch.tensor([[1.,2.],[3.,1.]],requires_grad=True)
        result=SimpleNamespace(logits=logits,audio_logits=None,aux_loss=logits.sum()*0)
        loss=objective(result,labels,audio,config,1)
        expected=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        logged,stats=objective(result,labels,audio,config,1,return_logging=True)
        actual=torch.autograd.grad(logged,logits)[0]
        self.assertTrue(torch.equal(expected,actual))
        self.assertFalse(stats.requires_grad)
        metrics=training_metrics(stats,0.5)
        self.assertAlmostEqual(metrics['train/loss'],loss.item(),places=6)
        self.assertEqual(metrics['train/main_codec_loss'],0)
