import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_pipeline'))
import reorganize_pretrain_data as prep

class ReorganizationTests(unittest.TestCase):
 def test_shared_librivox_speaker_partition(self):
  for i in range(1000):self.assertEqual(prep.partition('librispeech',str(i)),prep.partition('libritts_r',str(i)))
  n=sum(prep.partition('librispeech',str(i))=='pretrain' for i in range(1000))
  self.assertTrue(620<n<780,n)
 def test_unified_schema_retains_asr_instruction_and_media(self):
  row={'sample_id':'asr:one','speaker_id':'1','task_type':'asr','conversations':json.dumps([{'role':'user','content':'Transcribe exactly.'},{'role':'assistant','content':'hello'}]),'question_audios':[b'original-waveform']}
  got=prep.normalized(row,'librispeech','train','raw.parquet',3,prep.partition('librispeech','1'))
  self.assertEqual(got['question_audios'],row['question_audios']);self.assertEqual(got['conversations'],row['conversations']);self.assertEqual(got['raw_source_row'],3);self.assertEqual(got['answer_audios'],[])
 def test_tts_reference_cannot_cross_phase(self):
  row={'sample_id':'tts:one','speaker_id':'1','reference_speaker_id':'1','spk_emb':[0.1]*192,'answer_audios':[[1]*8],'codec_frames':1,'conversations':'[]'}
  phase=prep.partition('libritts_r','1');wrong='sft_reserved' if phase=='pretrain' else 'pretrain'
  with self.assertRaisesRegex(ValueError,'reference_cross_phase'):prep.normalized(row,'libritts_r','train','raw.parquet',0,wrong)
  got=prep.normalized(row,'libritts_r','train','raw.parquet',0,phase);self.assertEqual(got['answer_audios'],row['answer_audios'])
 def test_reject_partial_codec_without_truncation(self):
  row={'sample_id':'x','answer_audios':[[1]*9],'codec_frames':1,'conversations':'[]'}
  with self.assertRaisesRegex(ValueError,'incomplete_target_codec'):prep.normalized(row,'old_t2a','train','raw',0,'pretrain')
 def test_legacy_aia_requires_real_question_audio(self):
  with self.assertRaisesRegex(ValueError,'without_audio'):prep.normalized({'sample_id':'x'},'old_a2a_input','train','raw',0,'pretrain')


 def test_columnar_payload_and_normalization_equivalence(self):
  import pyarrow as pa
  from pretrain_columnar import convert_table
  row={'sample_id':'tts:one','task_type':'tts','conversations':'[]','answer_audios':[[12]*8],'codec_frames':1}
  table=pa.Table.from_pylist([row]);meta=[{**row,'_row':0,'_group':row['sample_id']}]
  got=convert_table(table,meta,prep.SCHEMA,'old_t2a','train','raw.parquet','pretrain').to_pylist()[0]
  expected=prep.normalized(row,'old_t2a','train','raw.parquet',0,'pretrain')
  self.assertEqual(got,expected)
 def test_columnar_rejects_partial_target(self):
  import pyarrow as pa
  from pretrain_columnar import convert_table
  row={'sample_id':'tts:bad','answer_audios':[[12]*9],'codec_frames':1}
  with self.assertRaisesRegex(ValueError,'incomplete_target_codec'):
   convert_table(pa.Table.from_pylist([row]),[{**row,'_row':0,'_group':'bad'}],prep.SCHEMA,'old_t2a','train','raw','pretrain')

if __name__=='__main__':unittest.main()
