#!/usr/bin/env python3
"""Prepare agreed legacy+supplement alignment pools, with explicit storage migration.

No SFT construction or model training. Metadata and rollback maps precede moves.
"""
import argparse,collections,fcntl,hashlib,json,os,shutil,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1];D=R/'dataset';W=R/'.runtime/pretrain-data';META=D/'raw/manifests/pretrain';STAGE=W/'build'
sys.path.insert(0,str(R))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def rel(p):return str(Path(p).relative_to(R))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def status(state,**kw):save(W/'status.json',{'status':state,'pid':os.getpid(),'updated_unix':time.time(),**kw})
def family(source):return 'librivox' if source in ['librispeech','libritts_r'] else source
def partition(source,speaker):
 key=f'{family(source)}:{speaker}'
 return 'pretrain' if int(hashlib.sha256(('miniqwen-partition-v4:42:'+key).encode()).hexdigest()[:16],16)%100<70 else 'sft_reserved'

SCHEMA=pa.schema([
 ('sample_id',pa.string()),('source',pa.string()),('source_row',pa.int64()),('split',pa.string()),
 ('task_type',pa.string()),('conversations',pa.string()),('question_audios',pa.list_(pa.binary())),
 ('answer_audios',pa.list_(pa.list_(pa.int64()))),('ref_audios',pa.list_(pa.int64())),('spk_emb',pa.list_(pa.float32())),
 ('image_bytes',pa.binary()),('codec_frames',pa.int32()),('reference_original_frames',pa.int32()),('reference_crop_start_frame',pa.int32()),
 ('language',pa.string()),('speaker_id',pa.string()),('text_original',pa.string()),('text_target',pa.string()),
 ('duration',pa.float64()),('sample_rate',pa.int32()),('codec_padding_samples',pa.int32()),
 ('reference_sample_id',pa.string()),('reference_speaker_id',pa.string()),('reference_split',pa.string()),
 ('reference_duration',pa.float64()),('voice_self_fallback',pa.bool_()),('training_partition',pa.string()),
 ('group_id',pa.string()),('supervision_policy',pa.string()),('raw_source_parquet',pa.string()),('raw_source_row',pa.int64())])

def inventory():
 path=W/'inventory.json'
 if path.exists():return json.loads(path.read_text())
 old=json.loads((D/'pretrain_legacy_v1/manifest.json').read_text());new=json.loads((D/'pretrain_v2/manifest.json').read_text());voice=json.loads((D/'pretrain_v2/voice/manifest.json').read_text());plan=json.loads((D/'pretrain_legacy_v1/pretrain_plan_full_v3.json').read_text())
 original={p:json.loads((D/p).read_text()) for p in ['pretrain_v2/manifest.json','pretrain_v2/voice/manifest.json','pretrain_legacy_v1/manifest.json','pretrain_legacy_v1/pretrain_plan_full_v3.json']}
 for name,value in original.items():save(META/'historical'/name,value)
 sources={}
 for name,stage in [('old_a2a_input','AIA'),('old_i2t','IIA'),('old_t2a','AOA')]:
  sources[name]={'stage':stage,'supplement':False,'paths':old['sources'][name]['data_files'],'expected':{s:v['rows'] for s,v in old['sources'][name]['splits'].items()}}
 for name,stage in [('librispeech','AIA'),('aishell1','AIA'),('llava','IIA'),('libritts_r','AOA'),('aishell3','AOA')]:
  spec=voice['sources'][name] if stage=='AOA' else new['sources'][name]
  sources[name]={'stage':stage,'supplement':name!='llava','paths':spec['data_files'],'expected':spec.get('counts') or {s:v['rows'] for s,v in spec['splits'].items()}}
 for source,src in sources.items():
  src['raw_paths']={}
  if src['supplement'] or source=='llava':
   for split,files in src['paths'].items():src['raw_paths'][split]=[f'dataset/raw/corpora/{source}/{split}-{i:03d}.parquet' for i,_ in enumerate(files)]
  else:src['raw_paths']=src['paths']
  src['physical_paths']={s:[str((R/p).resolve()) for p in files] for s,files in src['paths'].items()}
  src['input_bytes']=sum((R/p).stat().st_size for files in src['paths'].values() for p in files)
 d={'sources':sources,'parent_plan':plan,'original_roots':[str(p) for p in D.iterdir() if p.is_dir()],'created_unix':time.time(),'layout':'raw + pretrain/{AIA,IIA,AOA} + sft','no_training':True};save(path,d);return d

def migrate_raw(inv):
 log=json.loads((W/'path-migration.json').read_text()) if (W/'path-migration.json').exists() else {}
 for source,src in inv['sources'].items():
  if not (src['supplement'] or source=='llava'):continue
  for split,paths in src['paths'].items():
   for i,old in enumerate(paths):
    dest=R/src['raw_paths'][split][i];physical=Path(src['physical_paths'][split][i]);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
     if not physical.exists():raise FileNotFoundError(physical)
     os.replace(physical,dest)
    log[old]=rel(dest);save(W/'path-migration.json',log)
 for p in list(D.glob('sft_*.parquet')):
  target=D/'raw/legacy'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():os.replace(p,target)
  log[rel(p)]=rel(target);save(W/'path-migration.json',log)
 save(META/'path-migration.json',log)
 # Canonical imported corpora preserve the original rows and speaker/reference fields.
 for name,src in inv['sources'].items():
  if src['supplement'] or name=='llava':save(D/'raw/corpora'/name/'manifest.json',{'source':name,'data_files':src['raw_paths'],'splits':src['expected'],'provenance_paths':src['paths'],'note':'Retained prepared-source Parquet; original ordinary TTS waveforms were removed in prior authorized cleanup. Existing codec and voice fields preserved; not newly downloaded waveform originals.'})

class Writer:
 def __init__(self,stage,source):
  self.stage=stage;self.source=source;self.handles={};self.files={'train':[],'dev':[]};self.count=collections.Counter();self.shardrows=collections.Counter();self.stats={s:collections.Counter() for s in ['train','dev']}
 def add(self,split,rows):
  if not rows:return
  if split not in self.handles or self.shardrows[split]>=16384:
   if split in self.handles:self.handles[split].close()
   path=STAGE/self.stage/self.source/f'{split}-{len(self.files[split]):03d}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
   self.files[split].append(path);self.handles[split]=pq.ParquetWriter(path,SCHEMA,compression='zstd',compression_level=3);self.shardrows[split]=0
  self.handles[split].write_table(pa.Table.from_pylist(rows,schema=SCHEMA),row_group_size=128)
  self.shardrows[split]+=len(rows);self.count[split]+=len(rows)
  st=self.stats[split];st['rows']+=len(rows);st['codec_frames']+=sum(r['codec_frames'] for r in rows);st['audio_seconds']+=sum(r['duration'] or 0 for r in rows)
 def finish(self):
  for h in self.handles.values():h.close()
  return {'stage':self.stage,'source':self.source,'data_files':{s:[f'dataset/pretrain/{self.stage}/{self.source}/{p.name}' for p in files] for s,files in self.files.items()},'splits':{s:dict(self.stats[s]) for s in self.files},'file_sha256':{f'dataset/pretrain/{self.stage}/{self.source}/{p.name}':digest(p) for files in self.files.values() for p in files}}

def normalized(row,source,split,rawpath,index,selected):
 out={f.name:None for f in SCHEMA};out.update({k:v for k,v in row.items() if k in out})
 out.update(source=source,split=split,training_partition=selected,raw_source_parquet=rawpath,raw_source_row=index)
 if out['source_row'] is None:out['source_row']=index
 for key in ['question_audios','answer_audios','ref_audios','spk_emb']:out[key]=out[key] or []
 for key in ['codec_frames','reference_original_frames','reference_crop_start_frame','codec_padding_samples']:out[key]=out[key] or 0
 out['voice_self_fallback']=bool(out['voice_self_fallback'])
 out['supervision_policy']='all_text_last_audio' if out['codec_frames'] or out['task_type'] in {'audio_response','asr','tts'} else 'all_assistant_text'
 speaker=str(out['speaker_id'] or '');out['speaker_id']=speaker
 out['group_id']=f'{family(source)}:{speaker}' if speaker else str(out['sample_id'])
 if source in ['libritts_r','aishell3']:
  if out['reference_speaker_id']!=speaker:raise ValueError('reference_speaker_mismatch')
  if split=='train' and partition(source,out['reference_speaker_id'])!=selected:raise ValueError('reference_cross_phase')
  if len(out['spk_emb'])!=192:raise ValueError('invalid_speaker_dimension')
 if source=='old_a2a_input' and not any(out['question_audios']):raise ValueError('legacy_input_without_audio')
 if source in ['old_i2t','llava']:
  if not out['image_bytes'] or '<image>' not in out['conversations']:raise ValueError('missing_image_or_marker')
 if source in ['old_t2a','libritts_r','aishell3']:
  if not out['answer_audios'] or not out['answer_audios'][-1] or len(out['answer_audios'][-1])!=out['codec_frames']*8:raise ValueError('incomplete_target_codec')
 return out

def build_source(source,src):
 checkpoint=W/'source-results'/f'{source}.json'
 if checkpoint.exists():return json.loads(checkpoint.read_text())
 writer=Writer(src['stage'],source);reserved=collections.Counter();speaker_sets=collections.defaultdict(set);index_rows=[];rawcount=collections.Counter()
 indexschema=pa.schema([('sample_id',pa.string()),('source',pa.string()),('speaker_id',pa.string()),('official_split',pa.string()),('partition',pa.string()),('source_parquet',pa.string()),('source_row',pa.int64())])
 idxpath=META/'partitions'/f'{source}.parquet';idxpath.parent.mkdir(parents=True,exist_ok=True)
 with pq.ParquetWriter(idxpath,indexschema,compression='zstd') as indices:
  for split,paths in src['raw_paths'].items():
   for rawpath in paths:
    index=0
    for batch in pq.ParquetFile(R/rawpath).iter_batches(batch_size=128,use_threads=False):
     accepted=[];indices_batch=[]
     for row in batch.to_pylist():
      rawcount[split]+=1;speaker=str(row.get('speaker_id') or '')
      selected=partition(source,speaker) if src['supplement'] and split=='train' else 'pretrain' if split=='train' else 'dev'
      if src['supplement'] and not speaker:raise ValueError('speaker_required_for_partition')
      if selected=='sft_reserved':reserved['rows']+=1;reserved['audio_seconds']+=row.get('duration') or 0;speaker_sets['sft_reserved'].add(speaker)
      else:
       accepted.append(normalized(row,source,split,rawpath,index,selected));speaker_sets[selected].add(speaker)
      if src['supplement']:indices_batch.append({'sample_id':row['sample_id'],'source':source,'speaker_id':speaker,'official_split':split,'partition':selected,'source_parquet':rawpath,'source_row':index})
      index+=1
     writer.add(split,accepted)
     if indices_batch:indices.write_table(pa.Table.from_pylist(indices_batch,schema=indexschema))
     if rawcount[split]%16384==0:print('BUILD',source,split,rawcount[split],dict(writer.count),dict(reserved),flush=True)
  for split,n in src['expected'].items():
   if rawcount[split]!=n:raise ValueError((source,split,rawcount[split],n))
 if speaker_sets['pretrain']&speaker_sets['sft_reserved']:raise ValueError('speaker_phase_overlap')
 if (speaker_sets['pretrain']|speaker_sets['sft_reserved'])&speaker_sets['dev']- {''}:raise ValueError('speaker_dev_overlap')
 result=writer.finish();result.update(input_rows=dict(rawcount),sft_reserved=dict(reserved),speaker_counts={k:len(v-{''}) for k,v in speaker_sets.items()},partition_index=rel(idxpath) if src['supplement'] else None)
 if not src['supplement']:idxpath.unlink(missing_ok=True)
 save(checkpoint,result);print('SOURCE_COMPLETE',source,json.dumps({k:v for k,v in result.items() if k in ['splits','sft_reserved','speaker_counts']},ensure_ascii=False),flush=True)
 return result

def commit(inv):
 status('COMMITTING_LAYOUT')
 archive=D/'raw/_migration_archive';archive.mkdir(parents=True,exist_ok=True)
 # Retain immutable historical records while making only three dataset directories.
 for name in ['pretrain','pretrain_v2','pretrain_clean360','pretrain_legacy_v1','pretrain_audit']:
  old=D/name;dest=archive/name
  if old.exists() and not dest.exists():os.replace(old,dest)
 if not (D/'pretrain').exists():os.replace(STAGE,D/'pretrain')
 for name in ['arch_eval','arch_eval_holdout','eval_omni']:
  old=D/name;dest=D/'raw/evaluation_assets'/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if old.exists() and not dest.exists():os.replace(old,dest)
 sft=D/'sft';sft.mkdir(exist_ok=True)
 old=D/'sft_v1';dest=D/'raw/external/sft_v1';dest.parent.mkdir(parents=True,exist_ok=True)
 if old.exists() and not dest.exists():os.replace(old,dest)
 processed=dest/'processed'
 if processed.exists():
  for p in list(processed.iterdir()):
   target=sft/'external'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():os.replace(p,target)
  if not list(processed.iterdir()):processed.rmdir()
 remap={'dataset/sft_v1/processed/':'dataset/sft/external/','dataset/sft_v1/':'dataset/raw/external/sft_v1/','dataset/arch_eval/':'dataset/raw/evaluation_assets/arch_eval/','dataset/arch_eval_holdout/':'dataset/raw/evaluation_assets/arch_eval_holdout/','dataset/eval_omni/':'dataset/raw/evaluation_assets/eval_omni/'}
 save(META/'sft-path-remap.json',{'policy':'Existing SFT Parquet payload preserved byte-for-byte; use longest-prefix path aliases for relocated media. No new SFT semantic processing or codec encoding.','prefix_map':remap})
 for p in (sft/'external').glob('*/manifest.json'):
  m=json.loads(p.read_text())
  if 'data_files' in m:m['data_files']={k:v.replace('dataset/sft_v1/processed/','dataset/sft/external/') for k,v in m['data_files'].items()}
  m['media_path_remap']=rel(META/'sft-path-remap.json');save(p,m)
 # Retired metadata may contain old paths; canonical raw index and this migration map are authoritative.
 log=json.loads((W/'path-migration.json').read_text())
 log.update({f'dataset/{n}/':f'dataset/raw/_migration_archive/{n}/' for n in ['pretrain','pretrain_v2','pretrain_clean360','pretrain_legacy_v1','pretrain_audit']});log.update(remap)
 save(META/'path-migration.json',{'files':log,'use':'Exact paths first, longest prefix second. Historical plans are retired, not resume plans.'})
 save(sft/'README.json',{'status':'EXISTING_POOLS_MIGRATED_NEW_SFT_NOT_PREPARED','existing_external_pools':'dataset/sft/external','reserved_supplements':'dataset/raw/manifests/pretrain/partitions','old_a2a_source':'dataset/raw/legacy/sft_a2a.parquet','media_path_remap':rel(META/'sft-path-remap.json')})


def write_catalog(inv,results):
 sources={}
 for name,result in results.items():
  manifest=META/'sources'/f'{name}.json';spec={k:result[k] for k in ['data_files','splits','file_sha256']};spec['source_manifest']=rel(manifest);sources[name]=spec;save(manifest,result)
 catalog={'version':4,'sources':sources,'policy':{'base':'All previously qualified legacy views; do not include old_a2a_output in AOA.','supplements':'~70% speaker groups pretrain,~30% reserved; librispeech/libritts share librivox speaker namespace; official dev retained.','epochs':1,'mix':'All qualified samples once, no old/new target quota or infinite repeats.','schema':'Unified embedded-media Parquet, original conversations/codec retained.','max_seq_len':768,'validation':'Inherited complete-example length/media audits plus new schema/partition/full row accounting and reader acceptance.'}}
 save(META/'manifest.json',catalog)
 aoa={n:{'data_files':r['data_files'],'sha256':r['file_sha256']} for n,r in results.items() if r['stage']=='AOA'}
 save(META/'voice/manifest.json',{'sources':aoa,'policy':'Original independent same-speaker reference conditions for LibriTTS-R/AISHELL-3; legacy T2A may be unconditioned. No A2A output pretrain.'})
 plan=inv['parent_plan'];plan.pop('resume_parent_plan',None);plan.pop('resume_parent_plan_sha256',None);plan['version']=6;plan['data_root']=rel(META);plan['output_root']='out/miniqwen_pretrain';plan['data_catalog_sha256']=digest(META/'manifest.json');plan['voice_manifest_sha256']=digest(META/'voice/manifest.json')
 mapping={'audio':('AIA',['old_a2a_input','librispeech','aishell1']),'vision':('IIA',['old_i2t','llava']),'tts':('AOA',['old_t2a','libritts_r','aishell3'])}
 for phase in plan['phases']:
  stage,names=mapping[phase['name']];phase['alignment_stage']=stage;phase['sources']={n:results[n]['data_files'] for n in names};phase['epochs']=1;phase['max_seq_len']=768
 plan['tracking']={'project':'MiniQwen-Omni-Pretrain','name_prefix':'mixed-v4-'};plan['comparison_notes']=['Fresh run only; no migration from old checkpoint. Data preparation does not start training.','Legacy A2A input-only in AIA; full A2A output reserved for future SFT.','No fixed old/new quota; supplements speaker-partitioned approx70/30; all selected train examples once.']
 plan['formal_training_started']=False
 plan['code_sha256']={p:digest(R/p) for p in plan['code_sha256']};plan['code_sha256'][rel(R/'data_pipeline/reorganize_pretrain_data.py')]=digest(__file__)
 plan['code_sha256']['data_pipeline/pretrain_columnar.py']=digest(R/'data_pipeline/pretrain_columnar.py')
 save(META/'plan.json',plan)


def main():
 W.mkdir(parents=True,exist_ok=True);lock=(W/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inv=inventory();status('MIGRATING_RAW');migrate_raw(inv);results={}
 from pretrain_columnar import build_source_columnar
 for source,src in inv['sources'].items():
  status('BUILDING',source=source)
  results[source]=build_source_columnar(source,src,(R,W,META,STAGE,SCHEMA,Writer,partition,family,save,rel,digest))
 commit(inv);write_catalog(inv,results);status('BUILT_AWAITING_VALIDATION',catalog=rel(META/'manifest.json'));print('PRETRAIN_DATA_BUILD_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:status('FAILED',error=repr(exc));raise
