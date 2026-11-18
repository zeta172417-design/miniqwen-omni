import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from trainer.pretrain_plan import plan_signature,validate_checkpoint_signature

class PlanMigrationTest(unittest.TestCase):
 def test_execution_migration_preserves_training_contract(self):
  parent={'version':3,'output_root':'out/same','world_size':4,'seed':42,'global_batch':128,'code_sha256':{'trainer':'old'},'phases':[{'name':'vision','batch_size':8,'accumulation_steps':4,'sources':{'source':{'train':['train.parquet']}},'learning_rate':.0003}]}
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);path=root/'parent.json';path.write_text(json.dumps(parent));plan=copy.deepcopy(parent)
   plan.update(version=4,code_sha256={'trainer':'new'},execution={'distributed_eval':True},tracking={'project':'test'},resume_parent_plan='parent.json',resume_parent_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   plan['phases'][0].update(ddp_no_sync=True,compress_control_gradients=True)
   saved=plan_signature(parent,'vision',4)
   self.assertEqual(validate_checkpoint_signature(plan,'vision',4,saved,root),saved)
   self.assertIsNone(validate_checkpoint_signature(plan,'vision',4,plan_signature(plan,'vision',4),root))
   for field,value in [('batch_size',16),('learning_rate',.01),('sources',{})]:
    changed=copy.deepcopy(plan);changed['phases'][0][field]=value
    with self.assertRaisesRegex(ValueError,'training contract'):validate_checkpoint_signature(changed,'vision',4,saved,root)
   changed=copy.deepcopy(plan);changed['output_root']='out/other'
   with self.assertRaisesRegex(ValueError,'training contract'):validate_checkpoint_signature(changed,'vision',4,saved,root)
   with self.assertRaisesRegex(ValueError,'declared resume parent'):validate_checkpoint_signature(plan,'vision',4,'wrong',root)
   path.write_text('{}')
   with self.assertRaisesRegex(ValueError,'fingerprint'):validate_checkpoint_signature(plan,'vision',4,saved,root)

if __name__=='__main__':unittest.main()
