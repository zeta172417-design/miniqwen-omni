import pickle,tempfile,unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from dataset.parquet_reader import ParquetRows
from trainer.pretrain_sampler import ParquetDistributedSampler,EpochBatchSampler

class DirectParquetTest(unittest.TestCase):
 def fixture(self,tmp):
  paths=[];expected=[]
  for k in range(2):
   rows=[{'sample_id':str(k*11+i),'question_audios':[bytes([i])],'answer_audios':[[i,i+1]],'optional':None} for i in range(11)]
   path=Path(tmp)/f'{k}.parquet';pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=3);paths.append(str(path));expected+=rows
  return ParquetRows(paths,cache_bytes=1000),expected
 def test_random_access_payloads_workers_and_no_disk_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   ds,expected=self.fixture(tmp)
   for i in [21,0,3,11,4,0,-1]:self.assertEqual(ds[i],expected[i])
   with self.assertRaises(IndexError):ds[22]
   self.assertEqual(pickle.loads(pickle.dumps(ds))[5],expected[5])
   got=list(DataLoader(ds,batch_size=None,num_workers=2))
   self.assertEqual(got,expected)
   self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),['0.parquet','1.parquet'])
 def test_distributed_shuffle_resume_coverage(self):
  with tempfile.TemporaryDirectory() as tmp:
   ds,_=self.fixture(tmp);all_indices=[]
   for rank in range(4):
    s=ParquetDistributedSampler(ds,4,rank,42);full=list(EpochBatchSampler(s,2,2));self.assertEqual(len(full),6)
    self.assertEqual(list(EpochBatchSampler(s,2,2,4)),full[4:])
    all_indices += [i for batch in full[:3] for i in batch]
   self.assertEqual(set(all_indices),set(range(22)));self.assertEqual(len(all_indices),24)
   s=ParquetDistributedSampler(ds,1,0,42);a=list(s);s.set_epoch(1);b=list(s)
   self.assertNotEqual(a,b);self.assertEqual(sorted(a),sorted(b))
 def test_smaller_than_world(self):
  class Rows:
   row_group_ranges=[(0,1)]
   def __len__(self):return 1
  for rank in range(4):self.assertEqual(list(ParquetDistributedSampler(Rows(),4,rank)),[0])
