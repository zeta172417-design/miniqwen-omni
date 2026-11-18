#!/usr/bin/env python3
"""Sample full untruncated SFT conversations and measure current-model length."""
import collections,io,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'data_pipeline'))
from trainer.trainer_utils import load_omni_tokenizer
def required_length(input_ids, assistant_tokens, think_tokens, audio_frames=0, conditioned=False):
    # Conservative upper bound used for SFT pool sizing: tokenizer length plus
    # target Mimi frames and optional reference-condition prefix.
    return len(input_ids) + int(audio_frames) * 8 + (80 if conditioned and audio_frames else 0)
W=R/'.runtime/sft-length-sample';W.mkdir(exist_ok=True)
def save(name,value):(W/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sample(paths,n,columns=None,seed=42,positions=None):
 files=[(Path(p),pq.ParquetFile(p)) for p in paths];total=sum(f.metadata.num_rows for _,f in files);want=sorted(positions if positions is not None else random.Random(seed).sample(range(total),min(n,total)));wi=0;offset=0
 for path,pf in files:
  cols=[c for c in columns if c in pf.schema_arrow.names] if columns else None
  for batch in pf.iter_batches(batch_size=256,columns=cols,use_threads=False):
   end=offset+len(batch)
   while wi<len(want) and want[wi]<end:
    idx=want[wi];yield str(path),idx-offset+0,batch.slice(idx-offset,1).to_pylist()[0];wi+=1
   offset=end
   if wi==len(want):return

def main():
 os.nice(10);tok=load_omni_tokenizer(R/'model/Qwen3-0.6B');assistant=tok.encode('<|im_start|>assistant\n',add_special_tokens=False);think=tok.encode('</think>\n\n',add_special_tokens=False);records=[];errors=[]
 def measure(source,sid,messages,images=0,input_seconds=None,frames=0,conditioned=False,mode='text',estimated=False):
  messages=[dict(m) for m in messages];last=max(i for i,m in enumerate(messages) if m['role']=='user')
  if input_seconds is not None:
   audio='<|audio_start|>'+'<|audio_pad|>'*(math.ceil(input_seconds/.06)+2)+'<|audio_end|>'
   if mode=='asr':messages[last]['content']+='\n'+audio
   elif images:
    # Proposed fusion SFT must preserve images/instruction and omit spoken-Q transcript.
    messages[last]['content']='<image>\nAnswer the spoken question.\n'+audio
   else:messages[last]['content']=audio
  if mode.endswith('speech_output'):
   messages[last]['content']+='\nRespond with speech.'
  for m in messages:m['content']=m['content'].replace('<image>','<|vision_start|>'+'<|image_pad|>'*64+'<|vision_end|>')
  text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False);ids=tok(text,add_special_tokens=False).input_ids
  length=required_length(ids,assistant,think,frames,conditioned)
  records.append({'source':source,'sample_id':sid,'mode':mode,'required_length':length,'chat_tokens':len(ids),'image_count':images,'turns':sum(m['role']=='assistant' for m in messages),'codec_frames':frames,'condition_prefix':80 if conditioned and frames else 0,'input_seconds':input_seconds,'audio_length_estimated':input_seconds is not None or estimated})
 for name in ['ultrachat','coig','llava15_single','m4','fusion']:
  path=R/'dataset/sft/external'/name/'train.parquet'
  for _,_,entry in sample([path],3000,columns=['record_json']):
   r=json.loads(entry['record_json']);messages=r['conversations'];images=len(r.get('image_paths',[]));sid=r['sample_id']
   if name=='fusion':
    frames=math.ceil(r['target_seconds']*12.5)
    for mode,audio,target in [('image_text',None,0),('image_audio',r['input_seconds'],0),('image_text_speech_output',None,frames),('image_audio_speech_output',r['input_seconds'],frames)]:measure('fusion/'+mode,sid,messages,images,audio,target,False,mode,True)
   else:measure(name,sid,messages,images)
  print('MEASURED',name,flush=True);save('status.json',{'status':'SAMPLING','completed_source':name})
 for name in ['librispeech','aishell1','libritts_r','aishell3']:
  index=pq.read_table(R/f'dataset/raw/manifests/pretrain/partitions/{name}.parquet').to_pylist();selected=[r for r in index if r['partition']=='sft_reserved'];chosen=random.Random(42).sample(selected,min(1000,len(selected)));byfile=collections.defaultdict(list)
  for r in chosen:byfile[r['source_parquet']].append(r['source_row'])
  for path,positions in byfile.items():
   for _,_,r in sample([R/path],0,columns=['sample_id','task_type','conversations','duration','codec_frames','reference_sample_id'],positions=positions):
    messages=json.loads(r['conversations']);asr=name in ['librispeech','aishell1'];measure(name,r['sample_id'],messages,0,r['duration'] if asr else None,0 if asr else r['codec_frames'],not asr,'asr' if asr else 'tts_conditioned')
  print('MEASURED',name,len(chosen),flush=True)
 # Full old A2A uniform sample: retain original multi-turn history, only final question audio is injected.
 path=R/'dataset/raw/legacy/sft_a2a.parquet'
 for _,_,r in sample([path],768,columns=['conversations','question_audios','answer_audios','ref_audios','spk_emb']):
  try:
   messages=json.loads(r['conversations']);nu=sum(m['role']=='user' for m in messages);na=sum(m['role']=='assistant' for m in messages);audios=r.get('question_audios') or [];answers=r.get('answer_audios') or []
   q=audios[nu-1] if len(audios)>=nu else None
   if not q:continue
   duration=sf.info(io.BytesIO(q)).duration;codes=answers[na-1] if len(answers)>=na else []
   if len(codes)%8:raise ValueError('Partial codec frame')
   measure('old_a2a',f'old_a2a:sample{len(records)}',messages,0,duration,len(codes)//8,bool(r.get('ref_audios') or r.get('spk_emb')),'audio_speech_output')
  except Exception as exc:errors.append(repr(exc))
 print('MEASURED old_a2a',flush=True)
 with (W/'sample-lengths.jsonl').open('w') as f:
  for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 groups=collections.defaultdict(list)
 for r in records:groups[r['source']].append(r)
 result={}
 for name,rows in groups.items():
  xs=np.array([r['required_length'] for r in rows]);result[name]={'samples':len(rows),'p50':int(np.percentile(xs,50,method='higher')),'p90':int(np.percentile(xs,90,method='higher')),'p95':int(np.percentile(xs,95,method='higher')),'p99':int(np.percentile(xs,99,method='higher')),'max':int(xs.max()),'coverage':{str(n):round(float((xs<=n).mean()*100),3) for n in [768,1024,1536,2048,3072,4096,8192]},'mean_turns':sum(r['turns'] for r in rows)/len(rows),'audio_length_estimated':any(r['audio_length_estimated'] for r in rows)}
 payload={'status':'COMPLETE','seed':42,'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sources':result,'sample_rows_including_fusion_views':len(records),'errors':errors,'method':['Uniform row sampling without truncating conversations. External sources3000each train, voice reserved1000each, legacyA2A768source rows before no-audio exclusion.','Actual current tokenizer/chat template,64tokens/image+boundaries,80condition slots for TTS/clone. Full history, final target audio start as in required_length.','Input audio uses conservative ceil(seconds/0.06)+2 length bound, not frontend execution; no GPU/model training.','Fusion4 hypothetical task views reported separately; targetframes estimatedceil(seconds*12.5), not encoded exactcodec. Fusionviews are not fourfold independent training examples.','OldA2A original speaker/reference independence, dev/benchmark overlap not certified; finalpool audit pending.','Not length distribution of dev nor exact all-source finalmixedtraining; proposed fusion task wording can change a fewtokens.']}
 (R/'reports/SFT_LENGTH_SAMPLE.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');save('status.json',{'status':'COMPLETE','report':'reports/SFT_LENGTH_SAMPLE.json'});print(json.dumps(payload,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:save('status.json',{'status':'FAILED','error':repr(exc)});raise
