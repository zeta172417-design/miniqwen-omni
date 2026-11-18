"""Deterministic epoch shuffling with a global batch cursor for resume."""
import math
from torch.utils.data import BatchSampler

class EpochBatchSampler:
    def __init__(self, sampler, batch_size, epochs, skip_batches=0, condition_epochs=False):
        self.condition_epochs=condition_epochs
        self.sampler=sampler;self.batch_size=batch_size;self.epochs=epochs;self.skip_batches=skip_batches
        if epochs<1 or batch_size<1 or not 0<=skip_batches<=self.total_batches:
            raise ValueError('Invalid epoch batch cursor')
    @property
    def total_batches(self):return math.ceil(len(self.sampler)/self.batch_size)*self.epochs
    def __len__(self):return self.total_batches-self.skip_batches
    def __iter__(self):
        per_epoch=math.ceil(len(self.sampler)/self.batch_size)
        first,offset=divmod(self.skip_batches,per_epoch)
        for epoch in range(first,self.epochs):
            self.sampler.set_epoch(epoch)
            for i,batch in enumerate(BatchSampler(self.sampler,self.batch_size,drop_last=False)):
                if epoch==first and i<offset:continue
                yield [(index, epoch) for index in batch] if self.condition_epochs else batch

class ParquetDistributedSampler:
    """Shuffle groups and rows each epoch, preserving locality without dropping rows."""
    def __init__(self, rows, num_replicas, rank, seed=0):
        self.rows=rows;self.num_replicas=num_replicas;self.rank=rank;self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return math.ceil(len(self.rows)/self.num_replicas)
    def __iter__(self):
        import torch
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        ranges=self.rows.row_group_ranges;head=[];position=0
        padding=len(self)*self.num_replicas-len(self.rows)
        for group in torch.randperm(len(ranges),generator=g).tolist():
            start,end=ranges[group]
            for offset in torch.randperm(end-start,generator=g).tolist():
                index=start+offset
                if len(head)<padding:head.append(index)
                if position%self.num_replicas==self.rank:yield index
                position+=1
        if len(self.rows):
            for j in range(padding):
                if (position+j)%self.num_replicas==self.rank:yield head[j%len(head)]

def make_pretrain_sampler(rows, num_replicas, rank, seed, mode='global_random'):
    if mode=='global_random':
        from torch.utils.data import DistributedSampler
        return DistributedSampler(rows,num_replicas=num_replicas,rank=rank,shuffle=True,seed=seed,drop_last=False)
    if mode=='shuffle_row_groups_then_rows':
        return ParquetDistributedSampler(rows,num_replicas,rank,seed)
    raise ValueError(f'Unknown sampling mode: {mode}')
