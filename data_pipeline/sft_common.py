"""Shared SFT preparation contracts; importing this does not initialize training."""
import collections,hashlib,json,math,re,unicodedata,tempfile,os
from pathlib import Path
import pyarrow as pa
R=Path(__file__).resolve().parents[1];OUT=R/'dataset/sft';META=R/'dataset/raw/manifests/sft';W=R/'.runtime/sft-data'
SCHEMA=pa.schema([
 ('sample_id',pa.string()),('source',pa.string()),('source_row',pa.int64()),('split',pa.string()),('task_type',pa.string()),('conversations',pa.string()),
 ('question_audios',pa.list_(pa.binary())),('question_audio_paths',pa.list_(pa.string())),('answer_audios',pa.list_(pa.list_(pa.int64()))),
 ('image_paths',pa.list_(pa.string())),('ref_audios',pa.list_(pa.int64())),('spk_emb',pa.list_(pa.float32())),('codec_frames',pa.int32()),
 ('duration',pa.float64()),('sample_rate',pa.int32()),('language',pa.string()),('speaker_id',pa.string()),
 ('reference_sample_id',pa.string()),('reference_speaker_id',pa.string()),('voice_self_fallback',pa.bool_()),
 ('group_id',pa.string()),('length_tier',pa.string()),('required_length',pa.int32()),('text_length',pa.int32()),('audio_frames_upper_bound',pa.int32()),
 ('voice_conditioning',pa.string()),('supervision_policy',pa.string()),('raw_source_parquet',pa.string()),('raw_source_row',pa.int64()),('metadata_json',pa.string())])
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=p.parent,prefix=p.name+'.',suffix='.tmp',delete=False) as f:
  json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');name=f.name
 os.replace(name,p)
def digest_text(x):return hashlib.sha256(x.encode()).hexdigest()
def norm(x):return ''.join(c for c in unicodedata.normalize('NFKC',str(x)).casefold() if c.isalnum())
def resolve_media(x):
 for old,new in [('dataset/sft_v1/raw/fusion/','dataset/raw/external/fusion/'),('dataset/sft_v1/media/m4/','dataset/raw/external/m4/')]:
  if x.startswith(old):return new+x[len(old):]
 return x
def empty():
 d={f.name:None for f in SCHEMA}
 for k in ['question_audios','question_audio_paths','answer_audios','image_paths','ref_audios','spk_emb']:d[k]=[]
 d.update(codec_frames=0,voice_self_fallback=False,voice_conditioning='available_both',supervision_policy='all_assistant_text',metadata_json='{}',audio_frames_upper_bound=0);return d

def rendered_messages(row):
 messages=json.loads(row['conversations']);audio=row.get('audio_frames_upper_bound') or 0
 if audio:
  last=max(i for i,m in enumerate(messages) if m['role']=='user');messages[last]={**messages[last],'content':messages[last]['content']+'\n<|audio_start|>'+'<|audio_pad|>'*audio+'<|audio_end|>'}
 image_count=sum(m['content'].count('<image>') for m in messages)
 if image_count!=len(row.get('image_paths') or []):raise ValueError('image marker/path mismatch')
 return [{**m,'content':m['content'].replace('<image>','<|vision_start|>'+'<|image_pad|>'*64+'<|vision_end|>')} for m in messages]

def length_batch(rows,tokenizer):
 assistant=tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False);think=tokenizer.encode('</think>\n\n',add_special_tokens=False)
 prompts=[tokenizer.apply_chat_template(rendered_messages(r),tokenize=False,add_generation_prompt=False) for r in rows]
 ids_batch=tokenizer(prompts,add_special_tokens=False).input_ids
 for row,ids in zip(rows,ids_batch):
  frames=row['codec_frames'];prefix=160 if frames and (row['spk_emb'] or row['ref_audios']) else 0;required=len(ids)
  if frames:
   starts=[i+len(assistant) for i in range(len(ids)-len(assistant)+1) if ids[i:i+len(assistant)]==assistant]
   if not starts:raise ValueError('assistant marker missing')
   start=starts[-1]
   for i in range(start,len(ids)-len(think)+1):
    if ids[i:i+len(think)]==think:start=i+len(think);break
   required=max(required,start+frames+2)
  row.update(text_length=len(ids),required_length=required+prefix)
  if row['required_length']<=2048:row['length_tier']='main_2048'
  elif row['task_type']=='text' and row['required_length']<=3072:row['length_tier']='long_text_3072'
  else:row['length_tier']='excluded_overlength'
 return rows
