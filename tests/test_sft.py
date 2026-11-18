import json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_pipeline'))
from sft_common import empty,rendered_messages
from prepare_sft import fusion_view
from dataset.sft_dataset import SFTDataset,MediaRows
from dataset.voice_conditioning import choose_voice_mode,voice_pool

class SFTContractTest(unittest.TestCase):
 def test_legacy_t2a_is_separate_from_tts_condition_pool(self):
  row={'sample_id':'old','source':'old_t2a','task_type':'audio_response','spk_emb':[],'ref_audios':[]}
  self.assertEqual(voice_pool(row),'legacy_unconditioned_t2a')
  self.assertEqual(choose_voice_mode(row,42,0)[1],'none')
 def test_tts_condition_sampling_is_reproducible_and_epoch_aware(self):
  row={'sample_id':'tts-1','source':'libritts_r','task_type':'tts','spk_emb':[0.0]*192,'ref_audios':[1]*800}
  first=choose_voice_mode(row,42,0);self.assertEqual(first,choose_voice_mode(row,42,0))
  self.assertTrue(all(choose_voice_mode(row,42,e)[1] in {'both','speaker','reference','none'} for e in range(20)))
 def test_audio_instruction_and_all_images_are_preserved(self):
  row=empty();row.update(conversations=json.dumps([{'role':'user','content':'<image><image> Compare the spoken request.'},{'role':'assistant','content':'They differ.'}]),image_paths=['a','b'],audio_frames_upper_bound=33)
  messages=rendered_messages(row);self.assertEqual(messages[0]['content'].count('<|image_pad|>'),128);self.assertEqual(messages[0]['content'].count('<|audio_pad|>'),33);self.assertIn('Compare the spoken request.',messages[0]['content'])
 def test_wrong_image_count_is_rejected(self):
  row=empty();row['conversations']=json.dumps([{'role':'user','content':'<image> hi'},{'role':'assistant','content':'a'}])
  with self.assertRaises(ValueError):rendered_messages(row)
 def test_view_and_voice_allocation_deterministic(self):
  views=[fusion_view(str(i)) for i in range(10000)];self.assertEqual(set(views),{0,1,2,3});self.assertTrue(all(2000<views.count(i)<3000 for i in range(4)))
 def test_reader_keeps_image_and_spoken_instruction(self):
  ds=SFTDataset.__new__(SFTDataset);ds.training=True;ds.augment_enabled=False;ds.use_modality_boundaries=True;ds.audio_start_token='<|audio_start|>';ds.audio_end_token='<|audio_end|>';ds.audio_token='<|audio_pad|>'
  class Tok:
   def apply_chat_template(self,messages,**kw):return json.dumps(messages)
  ds.tokenizer=Tok();messages=[{'role':'user','content':'<image> Answer spoken question.'},{'role':'assistant','content':'Answer'}];out=ds.create_chat_prompt(messages,9,'image_audio_response');self.assertIn('<image>',out);self.assertIn('Answer spoken question.',out);self.assertEqual(out.count('<|audio_pad|>'),9)
 def test_media_paths_preserve_last_user_audio_alignment(self):
  import dataset.sft_dataset as mod
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'q.wav').write_bytes(b'wave');(root/'i.jpg').write_bytes(b'image')
   row={'conversations':json.dumps([{'role':'user','content':'old'},{'role':'assistant','content':'old'},{'role':'user','content':'<image> Listen'},{'role':'assistant','content':'a'}]),'image_paths':['i.jpg'],'question_audio_paths':['q.wav'],'question_audios':[]}
   with patch.object(mod,'ROOT',root):got=MediaRows([row])[0]
   self.assertEqual(got['image_bytes'],[b'image']);self.assertEqual(got['question_audios'],[b'',b'wave'])

class SFTWriterRegressionTest(unittest.TestCase):
 def test_literal_image_marker_excludes_only_affected_text_row(self):
  import pyarrow.parquet as pq
  from unittest.mock import patch
  import prepare_sft as build

  class Tokenizer:
   def encode(self, text, **kwargs):
    return [10] if text.startswith('<|im_start|>') else [11]
   def apply_chat_template(self, messages, **kwargs):
    return json.dumps(messages)
   def __call__(self, prompts, **kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(input_ids=[[1] * 20 for prompt in prompts])

  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   valid=empty()
   valid.update(sample_id='valid',source='text',split='train',task_type='text',conversations=json.dumps([{'role':'user','content':'Question'},{'role':'assistant','content':'Answer'}]))
   invalid={**valid,'sample_id':'invalid','conversations':json.dumps([{'role':'user','content':'An <image> literal'},{'role':'assistant','content':'Answer'}])}
   with patch.object(build,'OUT',root/'sft'),patch.object(build,'META',root/'meta'),patch.object(build,'R',root):
    writer=build.Writer('text',Tokenizer())
    writer.add([invalid,valid])
    report=writer.finish()
   self.assertEqual(report['counts'],{'main_2048/train':1})
   rejected=pq.read_table(root/'meta/excluded/text.parquet').to_pylist()
   self.assertEqual(rejected[0]['sample_id'],'invalid')

class SFTManifestEntryTest(unittest.TestCase):
 def test_manifest_resolves_tiers_and_capacity(self):
  from dataset.sft_dataset import load_manifest_paths
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'manifest.json'
   path.write_text(json.dumps({'tiers':{
    'main_2048':{'max_length':2048,'data_files':{'train':['dataset/sft/main_2048/train/x.parquet']}},
    'long_text_3072':{'max_length':3072,'data_files':{'train':['dataset/sft/long_text_3072/train/y.parquet']}}
   }}))
   paths,capacity=load_manifest_paths(path)
   self.assertEqual(capacity,3072)
   self.assertEqual(len(paths.split(',')),2)
   paths,capacity=load_manifest_paths(path,'main_2048')
   self.assertEqual(capacity,2048)
   self.assertEqual(len(paths.split(',')),1)
