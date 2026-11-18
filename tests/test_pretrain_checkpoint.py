import tempfile
from pathlib import Path
import unittest
from trainer.pretrain_checkpoint import recover_checkpoint
class PretrainCheckpointTest(unittest.TestCase):
    def snapshot(self,path):
        path.mkdir(parents=True)
        for name in ['config.json','trainer_state.pt','model.safetensors']:(path/name).write_bytes(b'snapshot')
    def test_interrupted_switch_restores_committed_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.snapshot(root/'.previous-checkpoint')
            partial=root/'.checkpoint-interrupted';partial.mkdir();(partial/'model.safetensors').write_bytes(b'partial')
            self.assertEqual(recover_checkpoint(root),root/'checkpoint')
            self.assertFalse((root/'.previous-checkpoint').exists());self.assertTrue(partial.exists())
    def test_current_checkpoint_is_preferred_to_older_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.snapshot(root/'checkpoint');self.snapshot(root/'.previous-checkpoint')
            self.assertEqual(recover_checkpoint(root),root/'checkpoint')
            self.assertTrue((root/'.previous-checkpoint').exists())
    def test_partial_checkpoint_is_not_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.previous-checkpoint').mkdir()
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):recover_checkpoint(root)
            self.assertFalse((root/'checkpoint').exists())
if __name__=='__main__':unittest.main()
