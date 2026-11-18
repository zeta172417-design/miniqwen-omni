#!/usr/bin/env python3
"""Publish manifests from the bounded, resumable training-data rebuild audit."""
import json,hashlib,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter,defaultdict
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'.runtime/data-finalize'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 temp=p.with_name(p.name+'.publish');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def main():
 records=json.loads((W/'result.json').read_text())['files'];by_path={r['path']:r for r in records}
 assert len(records)==len(list((ROOT/'dataset/pretrain').rglob('*.parquet')))+len([p for p in (ROOT/'dataset/sft').rglob('*.parquet') if 'external' not in p.parts])
 frames=Counter();statistics={}
 def process(pair):
  rel,r=pair
  cache=W/'statistics'/(hashlib.sha256(rel.encode()).hexdigest()+'.json')
  if cache.exists():
   hit=json.loads(cache.read_text());statistics[rel]=hit['statistics'];r.update(hit['record']);return
  if r['kind']=='sft' and r['source'] in {'libritts_r','aishell3','old_a2a'}:
   path=ROOT/rel;tmp=path.with_name(path.name+'.labels');writer=None;pf=pq.ParquetFile(path)
   for b in pf.iter_batches(batch_size=512,use_threads=False):
    t=pa.Table.from_batches([b]);nr=pc.list_value_length(t['ref_audios']).to_pylist()
    if r['source']!='old_a2a':
     names=['tts_clone' if n else 'tts_speaker' for n in nr]
     t=t.set_column(t.column_names.index('task_type'),'task_type',pa.array(names))
     meta=[dict(json.loads(x),voice_mode='runtime_categorical',conditions_stored='all_available') for x in t['metadata_json'].to_pylist()]
    else:
     meta=[]
     for x in t['metadata_json'].to_pylist():
      v=json.loads(x);v.pop('old_reference_codes_not_used_for_clone_supervision',None);v.update(legacy_reference_independence_verified=False,reference_max_seconds=12);meta.append(v)
    t=t.set_column(t.column_names.index('metadata_json'),'metadata_json',pa.array([json.dumps(x,ensure_ascii=False) for x in meta]))
    if writer is None:writer=pq.ParquetWriter(tmp,t.schema,compression='zstd',compression_level=3)
    writer.write_table(t,row_group_size=128)
   if writer:writer.close()
   pf.close();tmp.replace(path);r['sha256']=sha(path);r['bytes']=path.stat().st_size
  pf=pq.ParquetFile(ROOT/rel);assert pf.metadata.num_rows==r['rows'];assert 'supervision_policy' in pf.schema_arrow.names
  n=pf.metadata.num_rows;maxlen=0;frame=0;sumlen=0;tasks_one=Counter();voices_one=Counter()
  colindex=pf.schema.names.index('codec_frames')
  rgstats=[pf.metadata.row_group(i).column(colindex).statistics for i in range(pf.num_row_groups)]
  zero=all(st is not None and st.has_min_max and st.min==0 and st.max==0 for st in rgstats)
  columns=[] if zero else ['codec_frames']
  if 'required_length' in pf.schema_arrow.names:columns+=['required_length','task_type','voice_conditioning']
  if columns:
   for batch in pf.iter_batches(columns=columns,batch_size=65536,use_threads=True):
    tb=pa.Table.from_batches([batch])
    if 'codec_frames' in columns:frame+=pc.sum(tb['codec_frames']).as_py() or 0
    if 'required_length' in columns:
     maxlen=max(maxlen,pc.max(tb['required_length']).as_py() or 0);sumlen+=pc.sum(tb['required_length']).as_py() or 0
     tasks_one.update(tb['task_type'].to_pylist());voices_one.update(tb['voice_conditioning'].to_pylist())
  statistics[rel]={'rows':n,'codec_frames':frame,'max_required_length':maxlen,'sum_required_length':sumlen,'tasks':dict(tasks_one),'voice_modes':dict(voices_one)}
  cache.parent.mkdir(parents=True,exist_ok=True);save(cache,{'record':{'sha256':r['sha256'],'bytes':r['bytes']},'statistics':statistics[rel]})
  print('MANIFEST_STATS',rel,flush=True)
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(process,by_path.items()))
 for rel,r in by_path.items():frames[r['kind']]+=statistics[rel]['codec_frames']
 base=ROOT/'dataset/raw/manifests/pretrain';cat=json.loads((base/'manifest.json').read_text())
 for source,value in cat['sources'].items():
  src_path=ROOT/value['source_manifest'];src=json.loads(src_path.read_text())
  files=[r for r in records if r['kind']=='pretrain' and r['source']==source]
  hashes={r['path']:r['sha256'] for r in files};src['file_sha256']=hashes;value['file_sha256']=hashes
  for split in ['train','dev']:
   subset=[r for r in files if Path(r['path']).name.startswith(split+'-')]
   row=sum(r['rows'] for r in subset);fr=sum(statistics[r['path']]['codec_frames'] for r in subset)
   for spec in [src,value]:spec.setdefault('splits',{}).setdefault(split,{}).update(rows=row,codec_frames=fr)
  src['supervision']='all_assistant_text / all_text_last_audio; final audio only';save(src_path,src)
 cat['status']='PREPARED_PENDING_READER';cat['policy']['supervision']='All assistant text; last assistant audio only'
 save(base/'manifest.json',cat)
 voice=json.loads((base/'voice/manifest.json').read_text())
 for source,spec in voice['sources'].items():
  hashes={r['path']:r['sha256'] for r in records if r['kind']=='pretrain' and r['source']==source}
  spec['sha256']=hashes
 voice['policy']='Complete existing 4-second TTS conditions retained; runtime categorical dropout; old T2A independently unconditioned.';save(base/'voice/manifest.json',voice)
 plan=json.loads((base/'plan.json').read_text());plan['data_catalog_sha256']=sha(base/'manifest.json');plan['voice_manifest_sha256']=sha(base/'voice/manifest.json')
 for phase in plan['phases']:
  phase['condition_prefix_length']=160
  if phase['name']=='tts':phase.update(voice_conditioning='task_mixed',eval_voice_conditioning='available_both',speaker_warmup_fraction=0)
 plan['length_audit']={source:value['splits'] for source,value in cat['sources'].items()}
 plan['code_sha256']={p:sha(ROOT/p) for p in plan.get('code_sha256',{}) if (ROOT/p).is_file()}
 for p in ['dataset/voice_conditioning.py','dataset/sft_dataset.py','trainer/training_common.py','trainer/pretrain_sampler.py','model/voice_conditioning.py','data_pipeline/finalize_training_data.py']:
  plan['code_sha256'][p]=sha(ROOT/p)
 save(base/'plan.json',plan)
 p=ROOT/'dataset/sft/manifest.json';m=json.loads(p.read_text());m['file_sha256']={r['path']:r['sha256'] for r in records if r['kind']=='sft'}
 source_counts=defaultdict(Counter);policies=Counter();tasks=defaultdict(Counter);voices=defaultdict(Counter)
 for tier,v in m['tiers'].items():
  for split,paths in v['data_files'].items():
   v['counts'][split]=sum(by_path[rel]['rows'] for rel in paths)
   v['max_observed_length'][split]=max((statistics[rel]['max_required_length'] for rel in paths),default=0)
   for rel in paths:
    r=by_path[rel];source_counts[r['source']][tier+'/'+split]+=r['rows'];policies.update(r['policies'])
    tasks[tier+'/'+split].update(statistics[rel]['tasks']);voices[r['source']].update(statistics[rel]['voice_modes'])
 for source,spec in m['sources'].items():
  spec['counts']=dict(source_counts[source]);sp=ROOT/spec['manifest'];sm=json.loads(sp.read_text());sm['counts']=dict(source_counts[source]);sm['file_sha256']={r['path']:r['sha256'] for r in records if r['kind']=='sft' and r['source']==source};sm['rebuild_exclusions']=[x for r in records if r['kind']=='sft' and r['source']==source for x in r['excluded']]
  for key,paths in sm['data_files'].items():
   st=sm.setdefault('statistics',{}).setdefault(key,{})
   st.update(rows=sum(statistics[rel]['rows'] for rel in paths),target_codec_frames=sum(statistics[rel]['codec_frames'] for rel in paths),max_required_length=max((statistics[rel]['max_required_length'] for rel in paths),default=0),sum_required_length=sum(statistics[rel]['sum_required_length'] for rel in paths))
  if sm['rebuild_exclusions']:
   ep=ROOT/'dataset/raw/manifests/sft/excluded'/f'{source}.parquet'
   schema=pa.schema([('sample_id',pa.string()),('reason',pa.string()),('raw_source_parquet',pa.string()),('raw_source_row',pa.int64()),('required_length',pa.int32())])
   prior=pq.read_table(ep) if ep.exists() else pa.Table.from_pylist([],schema=schema)
   seen=set(prior['sample_id'].to_pylist());new=[x for x in sm['rebuild_exclusions'] if x['sample_id'] not in seen]
   if new:
    temp=ep.with_suffix('.publish');pq.write_table(pa.concat_tables([prior,pa.Table.from_pylist(new,schema=schema)]),temp,compression='zstd');temp.replace(ep)
  save(sp,sm)
 sr=[r for r in records if r['kind']=='sft'];m['statistics'].update(stored_rows=sum(r['rows'] for r in sr),target_codec_frames=frames['sft'],target_audio_equivalent_hours=frames['sft']/12.5/3600,parquet_GiB=sum(r['bytes'] for r in sr)/2**30,task_counts={k:dict(v) for k,v in tasks.items()},voice_modes={k:dict(v) for k,v in voices.items()},supervision_counts=dict(policies))
 m['validation']={'rebuilt_rows':'PASS','raw_voice_restoration':'PASS','codec_reference_caps':'PASS','reader':'PENDING','GPU_forward':'PENDING','media':'Unchanged raw media; prior source acceptance reused, no repeated full stat scan'}
 m['policies']['voice_views']='Store complete available raw conditions; train runtime both60/speaker15/reference10/none15 categorical draw; separate legacy T2A pool; no row quota.'
 m['policies']['supervision']='All assistant text labels; final assistant audio only; no random turn truncation.'
 m['limitations']=['Legacy A2A reference independence unverified; original condition fields retained as requested.','No new semantic/perceptual decontamination.','GPU forward acceptance pending; an untrained multimodal model can verify interface compatibility.']
 m['status']='PREPARED_PENDING_READER';m['created_at']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(p,m)
 summary={}
 for kind in ['sft','pretrain']:
  rr=[r for r in records if r['kind']==kind];summary[kind]={'rows':sum(r['rows'] for r in rr),'input_rows':sum(r['input_rows'] for r in rr),'excluded_rows':sum(len(r['excluded']) for r in rr),'files':len(rr),'GiB':sum(r['bytes'] for r in rr)/2**30,'sources':{}}
  for source in sorted({r['source'] for r in rr}):
   subset=[r for r in rr if r['source']==source]
   summary[kind]['sources'][source]={'rows':sum(r['rows'] for r in subset),'reference_rows':sum(r['reference_rows'] for r in subset),'speaker_rows':sum(r['speaker_rows'] for r in subset),'max_reference_frames':max(r['max_reference_frames'] for r in subset),'multi_assistant_rows':sum(r['multi_assistant_rows'] for r in subset)}
 save(W/'result.json',{'status':'FILES_COMPLETE','files':records});save(W/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
