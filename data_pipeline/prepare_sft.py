#!/usr/bin/env python3
"""Build accepted final SFT main/long-text Parquet pools, resumable per source."""
import collections,hashlib,io,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from sft_common import *
from prepare_sft_groups import external_rows
from model.voice_conditioning import TRAINING_REFERENCE_MAX_FRAMES, LEGACY_REFERENCE_MAX_FRAMES

def fusion_view(sid):return int(digest_text('sft-v5-view:42:'+sid)[:16],16)%4
def crop_codes(refs, limit):
 if not refs:return []
 if len(refs)%8:raise ValueError('incomplete_reference_codec')
 frames=len(refs)//8
 if frames<=limit:return refs
 start=(frames-limit)//2
 return refs[start*8:(start+limit)*8]
class Writer:
 def __init__(self,source,tok):
  save(W/'build-status.json',{'status':'BUILDING','source':source})
  self.source=source;self.tok=tok;self.handles={};self.paths=collections.defaultdict(list);self.counts=collections.Counter();self.sizes=collections.Counter();self.stats=collections.defaultdict(collections.Counter);self.exclusions=[];self.seen=set();self.excluded_handle=None;self.row_buffers=collections.defaultdict(list)
 def exclude(self,sid,reason,row=None):
  self.exclusions.append({'sample_id':str(sid),'reason':reason,'raw_source_parquet':(row or {}).get('raw_source_parquet'),'raw_source_row':(row or {}).get('raw_source_row'),'required_length':(row or {}).get('required_length')})
  if len(self.exclusions)>=512:self.flush_exclusions()
 def flush_exclusions(self):
  if not self.exclusions:return
  schema=pa.schema([('sample_id',pa.string()),('reason',pa.string()),('raw_source_parquet',pa.string()),('raw_source_row',pa.int64()),('required_length',pa.int32())]);p=META/'excluded'/f'{self.source}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
  if self.excluded_handle is None:self.excluded_handle=pq.ParquetWriter(p,schema,compression='zstd')
  self.excluded_handle.write_table(pa.Table.from_pylist(self.exclusions,schema=schema));self.exclusions=[]
 def add(self,rows):
  if not rows:return
  valid_rows = []
  for row in rows:
   try:
    rendered_messages(row)
   except (ValueError, KeyError, TypeError) as exc:
    self.exclude(row['sample_id'], 'invalid_task_format:' + str(exc), row)
    continue
   valid_rows.append(row)
  if not valid_rows:
   return
  for row in length_batch(valid_rows,self.tok):
   sid=row['sample_id'];reason=None
   if sid in self.seen:raise ValueError('Repeated sample ID')
   self.seen.add(sid)
   if row['length_tier']=='excluded_overlength':self.exclude(sid,'complete_sequence_over_budget',row);continue
   if row['codec_frames']:
    codes=row['answer_audios'][-1]
    if len(codes)!=8*row['codec_frames'] or min(codes)<0 or max(codes)>=2048:raise ValueError('Invalid target codec')
   tier=row['length_tier'];split=row['split'];key=(tier,split)
   if key not in self.handles or self.sizes[key]>=8192:
    if key in self.handles:self.close(key)
    p=OUT/tier/split/f'{self.source}-{len(self.paths[key]):03d}.parquet';p.parent.mkdir(parents=True,exist_ok=True);self.paths[key].append(p)
    self.handles[key]=pq.ParquetWriter(p.with_suffix('.parquet.partial'),SCHEMA,compression='zstd',compression_level=3);self.sizes[key]=0
   # Bounded metadata rows become one small row-group batch; media remain inline only where needed.
   self.row_buffers[key].append(row)
   if len(self.row_buffers[key])>=128:
    self.handles[key].write_table(pa.Table.from_pylist(self.row_buffers[key],schema=SCHEMA),row_group_size=128);self.row_buffers[key]=[]
   self.sizes[key]+=1;self.counts[key]+=1
   if sum(self.counts.values()) % 8192 == 0:
    print('SOURCE_PROGRESS',self.source,sum(self.counts.values()),flush=True)
    save(W/'build-status.json',{'status':'BUILDING','source':self.source,'accepted_rows':sum(self.counts.values())})
   st=self.stats[key];st['rows']+=1;st['max_required_length']=max(st['max_required_length'],row['required_length']);st['sum_required_length']+=row['required_length'];st['target_codec_frames']+=row['codec_frames'];st['images']+=len(row['image_paths']);st['input_audio_seconds']+=row['duration'] or 0 if row['audio_frames_upper_bound'] else 0
 def close(self,key):
  if self.row_buffers[key]:
   self.handles[key].write_table(pa.Table.from_pylist(self.row_buffers[key],schema=SCHEMA),row_group_size=128);self.row_buffers[key]=[]
  self.handles.pop(key).close();p=self.paths[key][-1];p.with_suffix('.parquet.partial').replace(p)
 def finish(self):
  for key in list(self.handles):self.close(key)
  self.flush_exclusions()
  if self.excluded_handle:self.excluded_handle.close()
  report={'status':'COMPLETE','source':self.source,'data_files':{'/'.join(k):[str(p.relative_to(R)) for p in ps] for k,ps in self.paths.items()},'counts':{'/'.join(k):v for k,v in self.counts.items()},'statistics':{'/'.join(k):dict(v) for k,v in self.stats.items()},'seen_sample_ids':len(self.seen)}
  save(META/'sources'/f'{self.source}.json',report);print('SOURCE_READY',self.source,json.dumps(report['counts']),flush=True);return report

def old_audio_groups(path):
 from prepare_sft_groups import DSU
 checkpoint=META/'old-a2a-groups.parquet'
 if checkpoint.exists():return {r['source_row']:(r['group_id'],r['split']) for r in pq.read_table(checkpoint).to_pylist()}
 graph=DSU();dev=set();rows=[];offset=0
 for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['conversations','question_audios'],use_threads=False):
  for raw in batch.to_pylist():
   idx=offset;offset+=1
   try:
    messages=json.loads(raw['conversations']);nu=sum(m['role']=='user' for m in messages);qs=raw.get('question_audios') or [];q=qs[nu-1] if len(qs)>=nu else None
    if not q:continue
    target=norm(messages[-1]['content']);text_hash=hashlib.sha256(target.encode()).hexdigest();audio_hash=hashlib.sha256(q).hexdigest();node='target:'+text_hash;graph.union(node,'question:'+audio_hash);rows.append((idx,node))
    if int.from_bytes(bytes.fromhex(text_hash)[:8],'big')%100==0:dev.add(node)
   except (ValueError,TypeError,KeyError,IndexError):continue
  if offset%32768==0:print('OLD_GROUP_INDEX',offset,flush=True)
 dev_roots={graph.find(h) for h in dev};output=[{'source_row':idx,'group_id':graph.find(node),'split':'dev' if graph.find(node) in dev_roots else 'train'} for idx,node in rows]
 pq.write_table(pa.Table.from_pylist(output),checkpoint,compression='zstd');counts=collections.Counter(r['split'] for r in output);save(META/'old-a2a-group-audit.json',{'status':'PASS','counts':dict(counts),'policy':'Union identical final input-audio bytes and normalized target-text groups; preserve previous normalized-target dev membership; whole component dev if any existing dev target. No input recording or exact target group crosses SFT train/dev.'})
 return {r['source_row']:(r['group_id'],r['split']) for r in output}

def main():
 os.nice(15)
 while not (META/'group-audit.json').exists():time.sleep(5)
 from trainer.trainer_utils import load_omni_tokenizer
 tok=load_omni_tokenizer(R/'model/Qwen3-0.6B');media={r['path']:r for r in pq.read_table(META/'media-groups.parquet').to_pylist()};texts={r['group_id']:r['split'] for r in pq.read_table(META/'text-groups.parquet').to_pylist()};blocked=json.loads((META/'blocked-evaluation.json').read_text());blocked_q=set(blocked['question_normalized']);blocked_t=set(blocked['target_normalized']);text_dev=set(blocked['text_conversation_dev_digests']);seen_text=set()
 # Official development target text exclusion is cross-corpus, including cross-ASR/TTS shared Chinese text.
 official_dev_targets=set()
 for source in ['librispeech','aishell1','libritts_r','aishell3']:
  for p in (R/'dataset/raw/corpora'/source).glob('dev*.parquet'):
   for b in pq.ParquetFile(p).iter_batches(columns=['text_target']):
    official_dev_targets.update(norm(x) for x in b.column(0).to_pylist() if len(norm(x))>=16)
 blocked_t|=official_dev_targets
 def blocked_messages(messages):
  return any(len(norm(m['content']))>=16 and norm(m['content']) in (blocked_q if m['role']=='user' else blocked_t) for m in messages)
 def process_external(name):
  if (META/'sources'/f'{name}.json').exists():return
  writer=Writer(name,tok);buffer=[];codec={}
  if name=='fusion':
   while not (R/'dataset/raw/corpora/sft_fusion_codec/manifest.json').exists():time.sleep(10)
   for p in (R/'dataset/raw/corpora/sft_fusion_codec').glob('part-*.parquet'):
    for r in pq.read_table(p).to_pylist():codec[r['sample_id']]=r
  for original_split,r in external_rows(name):
   sid=r['sample_id'];messages=r['conversations'];row=empty();row.update(sample_id=sid,source=name,task_type='text',supervision_policy='all_assistant_text',source_row=r.get('source_row'),conversations=json.dumps(messages,ensure_ascii=False),raw_source_parquet=f'dataset/sft/external/{name}/{original_split}.parquet',raw_source_row=None)
   if blocked_messages(messages):writer.exclude(sid,'fixed_eval_or_official_dev_exact_text',row);continue
   if name in ('ultrachat','coig'):
    first=next(m['content'] for m in messages if m['role']=='user');group=digest_text(norm(first));split=texts[group];fingerprint=digest_text(json.dumps(messages,ensure_ascii=False,sort_keys=True))
    if original_split=='train' and fingerprint in text_dev:writer.exclude(sid,'duplicate_dev_conversation',row);continue
    if fingerprint in seen_text:writer.exclude(sid,'cross_pool_exact_conversation_duplicate',row);continue
    seen_text.add(fingerprint);row.update(split=split,group_id=group)
   else:
    paths=[resolve_media(x) for x in r['image_paths']];assets=[media[p] for p in paths];groups={a['group_id'] for a in assets};assert len(groups)==1
    if any(a['blocked_benchmark_image'] for a in assets):writer.exclude(sid,'fixed_evaluation_image_component',row);continue
    row.update(split=assets[0]['split'],group_id=assets[0]['group_id'],image_paths=paths,task_type='image_instruction',supervision_policy='all_assistant_text')
    if name=='fusion':
     view=fusion_view(sid);messages=[dict(m) for m in messages];last=max(i for i,m in enumerate(messages) if m['role']=='user');spoken=view in (1,3);speech=view in (2,3)
     if spoken:messages[last]['content']='<image>\nAnswer the spoken question.';row['question_audio_paths']=[resolve_media(r['question_audio_path'])];row['duration']=r['input_seconds'];row['audio_frames_upper_bound']=math.ceil(r['input_seconds']/.06)+2
     if speech:
      messages[last]['content']+='\nRespond with speech.';c=codec[sid];row['codec_frames']=c['codec_frames'];row['answer_audios']=[c['codes']]
     row['task_type']='image_audio_response' if spoken else 'image_instruction';row['supervision_policy']='all_text_last_audio' if speech else 'all_assistant_text';row['conversations']=json.dumps(messages,ensure_ascii=False);row['metadata_json']=json.dumps({'fusion_view':view,'source_image_group':r['group_id'],'input_transcript_not_provided':spoken,'source_output_seconds':r['target_seconds']})
   buffer.append(row)
   if len(buffer)>=128:writer.add(buffer);buffer=[]
  writer.add(buffer);writer.finish()
 for name in ['ultrachat','coig','llava15_single','m4']:process_external(name)
 print('EXTERNAL_NONFUSION_DONE',flush=True)
 # Local corpus data retain their existing 70/30 speaker partition; no new target audio encoding.
 for source in ['librispeech','aishell1','libritts_r','aishell3']:
  if (META/'sources'/f'{source}.json').exists():continue
  writer=Writer(source,tok);index=pq.read_table(META.parent/'pretrain/partitions'/f'{source}.parquet').to_pylist();reserved={r['sample_id'] for r in index if r['partition']=='sft_reserved'};reserved_array=pa.array(sorted(reserved));asr=source in ('librispeech','aishell1');buffer=[]
  for path in sorted((R/'dataset/raw/corpora'/source).glob('*.parquet')):
   offset=0
   for batch in pq.ParquetFile(path).iter_batches(batch_size=128,use_threads=False):
    # Filter in Arrow before converting codec integers or media bytes into Python.
    import pyarrow.compute as pc
    selected=batch.filter(pc.is_in(batch.column(batch.schema.get_field_index('sample_id')),value_set=reserved_array)) if path.name.startswith('train') else batch
    # Original row offset comes from a small sample_id lookup for this batch.
    positions={x:i+offset for i,x in enumerate(batch.column(batch.schema.get_field_index('sample_id')).to_pylist())}
    for raw in selected.to_pylist():
     sid=raw['sample_id'];split='train' if path.name.startswith('train') else 'dev';row=empty();row.update(sample_id='sft:'+sid,source=source,split=split,source_row=positions[sid],raw_source_parquet=str(path.relative_to(R)),raw_source_row=positions[sid],group_id=('librivox' if source in ('librispeech','libritts_r') else source)+':'+str(raw['speaker_id']),speaker_id=str(raw['speaker_id']),language=raw.get('language'),duration=raw.get('duration'),sample_rate=raw.get('sample_rate'),task_type='asr' if asr else 'tts',supervision_policy='all_assistant_text' if asr else 'all_text_last_audio',conversations=raw['conversations'])
     messages=json.loads(raw['conversations']);target=norm(raw.get('text_target') or messages[-1]['content'])
     if split=='train' and target in blocked_t:writer.exclude(sid,'fixed_eval_or_official_dev_exact_target',row);continue
     if asr:
      row['question_audios']=raw['question_audios'];row['audio_frames_upper_bound']=math.ceil(raw['duration']/.06)+2
     else:
      mode='task_mixed';row['codec_frames']=raw['codec_frames'];row['answer_audios']=raw['answer_audios'];valid_ref=not raw.get('voice_self_fallback') and raw.get('reference_sample_id')!=sid and str(raw.get('reference_speaker_id'))==str(raw['speaker_id'])
      row['spk_emb']=raw.get('spk_emb') or [];row['ref_audios']=crop_codes(raw.get('ref_audios') or [], TRAINING_REFERENCE_MAX_FRAMES) if valid_ref else [];row['reference_sample_id']=raw.get('reference_sample_id');row['reference_speaker_id']=raw.get('reference_speaker_id');row['voice_self_fallback']=bool(raw.get('voice_self_fallback'))
      row['task_type']='tts_clone' if row['ref_audios'] else 'tts_speaker'
      row['voice_conditioning']='task_mixed' if split=='train' else 'available_both';row['metadata_json']=json.dumps({'voice_mode':mode,'reference_independent':valid_ref,'original_reference_split':raw.get('reference_split')})
     buffer.append(row)
     if len(buffer)>=64:writer.add(buffer);buffer=[]
    offset+=len(batch)
   print('LOCAL_FILE',source,path.name,flush=True)
  writer.add(buffer);writer.finish()
 # Legacy conversational audio: keep final audio question, full text history, complete last target codes.
 source='old_a2a'
 if not (META/'sources'/f'{source}.json').exists():
  writer=Writer(source,tok);path=R/'dataset/raw/legacy/sft_a2a.parquet';buffer=[];offset=0;seen=set();old_groups=old_audio_groups(path)
  for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['conversations','question_audios','answer_audios','spk_emb','ref_audios'],use_threads=False):
   for raw in batch.to_pylist():
    idx=offset;offset+=1;sid=f'sft:old_a2a:{idx}';row=empty();row.update(sample_id=sid,source=source,source_row=idx,raw_source_parquet=str(path.relative_to(R)),raw_source_row=idx,task_type='audio_response',supervision_policy='all_text_last_audio')
    try:
     messages=json.loads(raw['conversations']);users=sum(m['role']=='user' for m in messages);assistants=sum(m['role']=='assistant' for m in messages);questions=raw.get('question_audios') or [];answers=raw.get('answer_audios') or []
     q=questions[users-1] if len(questions)>=users else None
     if not q:writer.exclude(sid,'no_final_question_audio_not_a2a',row);continue
     if not messages or messages[-1]['role']!='assistant' or not messages[-1]['content'].strip():raise ValueError('invalid conversation')
     if blocked_messages(messages):writer.exclude(sid,'fixed_eval_or_official_dev_exact_text',row);continue
     codes=answers[assistants-1] if len(answers)>=assistants else []
     if not codes or len(codes)%8:raise ValueError('missing_or_partial_codec')
     info=sf.info(io.BytesIO(q));duration=info.duration
     if not info.frames:raise ValueError('empty audio')
     target=norm(messages[-1]['content']);group,split=old_groups[idx]
     content_digest=hashlib.sha256(q+np.asarray(codes,dtype='<i8').tobytes()+raw['conversations'].encode()).hexdigest()
     if content_digest in seen:writer.exclude(sid,'exact_qa_media_duplicate',row);continue
     seen.add(content_digest);last=max(i for i,m in enumerate(messages) if m['role']=='user');messages[last]={**messages[last],'content':'Listen to the spoken question and respond with speech.'}
     row.update(split=split,group_id=group,question_audios=[b'']*(users-1)+[q],answer_audios=[[]]*(assistants-1)+[codes],codec_frames=len(codes)//8,duration=duration,sample_rate=info.samplerate,audio_frames_upper_bound=math.ceil(duration/.06)+2,conversations=json.dumps(messages,ensure_ascii=False),metadata_json=json.dumps({'original_final_question_text_preserved_in_raw':True,'legacy_reference_independence_verified':False}))
     speaker=raw.get('spk_emb') or []
     if len(speaker)==192 and all(math.isfinite(x) for x in speaker):row['spk_emb']=speaker;row['voice_conditioning']='speaker'
     # Legacy A2A references are retained when present, even though their
     # independent-reference provenance is unknown. They are capped at the
     # legacy 12-second budget and marked accordingly for evaluation policy.
     legacy_ref=crop_codes(raw.get('ref_audios') or [], LEGACY_REFERENCE_MAX_FRAMES)
     if legacy_ref:
      row['ref_audios']=legacy_ref;row['voice_conditioning']='both' if row['spk_emb'] else 'reference'
      row['metadata_json']=json.dumps({'legacy_reference_provenance':'unknown','reference_max_seconds':12.0})
     buffer.append(row)
     if len(buffer)>=64:writer.add(buffer);buffer=[]
    except (ValueError,TypeError,KeyError,IndexError) as exc:writer.exclude(sid,'invalid:'+str(exc),row)
   if offset%8192==0:print('OLD_A2A',offset,flush=True)
  writer.add(buffer);writer.finish()
 process_external('fusion')
 save(W/'build-status.json',{'status':'BUILT_AWAITING_VALIDATION'});print('SFT_BUILD_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:save(W/'build-status.json',{'status':'FAILED','error':repr(exc)});raise
