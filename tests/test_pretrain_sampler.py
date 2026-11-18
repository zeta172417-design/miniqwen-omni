import unittest
from torch.utils.data import DistributedSampler
from trainer.pretrain_sampler import EpochBatchSampler

class EpochResumeTest(unittest.TestCase):
 def test_resume_and_epoch_reshuffle(self):
  def batches(skip=0):return list(EpochBatchSampler(DistributedSampler(range(23),num_replicas=1,rank=0,seed=42),4,2,skip))
  full=batches();self.assertEqual(len(full),12)
  self.assertEqual(sorted(i for b in full[:6] for i in b),list(range(23)))
  self.assertEqual(sorted(i for b in full[6:] for i in b),list(range(23)))
  self.assertNotEqual(full[:6],full[6:])
  for skip in (0,2,5,6,7,12):self.assertEqual(batches(skip),full[skip:])
 def test_distributed_ranks_and_single_epoch(self):
  for rank in range(4):
   s=DistributedSampler(range(32),num_replicas=4,rank=rank,seed=42)
   expected=list(s)
   got=list(EpochBatchSampler(s,3,1))
   self.assertEqual([i for b in got for i in b],expected)

class GlobalRandomTest(unittest.TestCase):
 def test_matches_original_global_distributed_order_and_resume(self):
  from trainer.pretrain_sampler import make_pretrain_sampler
  for rank in range(4):
   ds=range(127)
   sampler=make_pretrain_sampler(ds,4,rank,42)
   original=DistributedSampler(ds,num_replicas=4,rank=rank,seed=42)
   for epoch in (0,1):
    sampler.set_epoch(epoch);original.set_epoch(epoch)
    self.assertEqual(list(sampler),list(original))
   full=list(EpochBatchSampler(sampler,8,2))
   self.assertEqual(list(EpochBatchSampler(sampler,8,2,5)),full[5:])
