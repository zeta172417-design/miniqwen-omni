#!/usr/bin/env python3
"""Validate, reader-check, then retire redundant data. Never starts training."""
import collections,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'data_pipeline'))
from reorganize_pretrain_data import SCHEMA,partition,save,digest
D=R/'dataset';W=R/'.runtime/pretrain-data';M=D/'raw/manifests/pretrain'
def status(s,**kw):save(W/'finish-status.json',{'status':s,'updated_unix':time.time(),**kw})
def metadata_checks():
 catalog=json.loads((M/'manifest.json').read_text());results={};samplepaths={};splitsets={}
 for source,spec in catalog['sources'].items():
  meta=json.loads((M/'sources'/f'{source}.json').read_text());actual={}
  for split,paths in spec['data_files'].items():
   n=0;selected=[]
   for path in paths:
    pf=pq.ParquetFile(R/path)
    if pf.schema_arrow!=SCHEMA:raise ValueError('Parquet schema mismatch:'+path)
    n+=pf.metadata.num_rows
    # Only sample payloads here; build checked every row and inherited full length/media audits.
   if n!=spec['splits'][split]['rows']:raise ValueError((source,split,'row count mismatch'))
   actual[split]=n
   # Global first/middle/last rows, without loading all binary media into memory.
   want={0,n//2,n-1};offset=0
   for path in paths:
    pf=pq.ParquetFile(R/path)
    for i in range(pf.num_row_groups):
     nr=pf.metadata.row_group(i).num_rows;positions=sorted(x-offset for x in want if offset<=x<offset+nr)
     if positions:selected.extend(pf.read_row_group(i).take(pa.array(positions)).to_pylist())
     offset+=nr
   target=W/'reader-inputs'/f'{source}-{split}.parquet';target.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(selected,schema=SCHEMA),target,compression='zstd')
   samplepaths[source+'/'+split]=str(target)
  if meta.get('partition_index'):
   index=pq.read_table(R/meta['partition_index']).to_pylist();counts=collections.Counter();speakers=collections.defaultdict(set)
   for row in index:
    counts[(row['official_split'],row['partition'])]+=1;speakers[row['partition']].add(row['speaker_id'])
    if row['official_split']=='train' and row['partition']!=partition(source,row['speaker_id']):raise ValueError('unstable source partition')
   assert not speakers['pretrain']&speakers['sft_reserved'];assert not (speakers['pretrain']|speakers['sft_reserved'])&speakers['dev']
   assert actual['train']==counts[('train','pretrain')];assert actual['dev']==counts[('dev','dev')]
   assert actual['train']+counts[('train','sft_reserved')]==meta['input_rows']['train']
   splitsets[source]=speakers
  results[source]={'actual_rows':actual,'sft_reserved':meta['sft_reserved'],'speaker_counts':meta['speaker_counts'],'stage':meta['stage']}
 # Cross-corpus English speaker partition consistency.
 if {'librispeech','libritts_r'}<=set(splitsets):
  assert not splitsets['librispeech']['pretrain']&splitsets['libritts_r']['sft_reserved']
  assert not splitsets['libritts_r']['pretrain']&splitsets['librispeech']['sft_reserved']
 # Original corpora conserved, including the complete old A2A kept for future SFT.
 for name,n in [('a2a',414024),('i2t',2904511),('t2a',1248923)]:assert pq.ParquetFile(D/'raw/legacy'/f'sft_{name}.parquet').metadata.num_rows==n
 assert set(p.name for p in (D/'pretrain').iterdir())=={'AIA','IIA','AOA'}
 assert not any('a2a_output' in n for n in catalog['sources'])
 # Existing SFT pools are only relocated; verify their counts and alias resolution.
 sft_counts={};remap={'dataset/sft_v1/raw/fusion/':'dataset/raw/external/fusion/','dataset/sft_v1/media/m4/':'dataset/raw/external/m4/'}
 for name in ['ultrachat','coig','fusion','m4']:
  spec=json.loads((D/'sft/external'/name/'manifest.json').read_text());sft_counts[name]={}
  for split,path in spec['data_files'].items():
   pf=pq.ParquetFile(R/path);assert pf.metadata.num_rows==spec['counts'][split];sft_counts[name][split]=pf.metadata.num_rows
   if name in ['fusion','m4']:
    row=json.loads(next(pf.iter_batches(batch_size=1,columns=['record_json'])).column(0)[0].as_py())
    assets=row.get('image_paths',[])+[row[k] for k in ['question_audio_path','answer_audio_path'] if row.get(k)]
    for asset in assets:
     current=asset
     for before,after in sorted(remap.items(),key=lambda item:len(item[0]),reverse=True):
      if current.startswith(before):current=after+current[len(before):];break
     assert (R/current).is_file(),current
 assert sum(sum(v.values()) for v in sft_counts.values())==533335
 save(W/'sft-relocation-validation.json',{'status':'PASS','rows':sft_counts,'payloads_unchanged':True,'alias_resolution':'PASS'})
 protocol=json.loads((R/'.runtime/legacy-pretrain-full-v3-eval/protocol.json').read_text())
 assert digest(protocol['main_manifest'])==protocol['main_manifest_sha256']
 assert digest(protocol['clone_manifest'])==protocol['clone_manifest_sha256']
 save(W/'metadata-validation.json',{'status':'PASS','sources':results,'sample_paths':samplepaths,'english_cross_corpus_speaker_partition':'PASS','original_legacy_corpora_preserved':True,'sft_new_processing':False})
 return results,samplepaths

def reader_checks(samples):
 from transformers import SiglipImageProcessor
 from dataset.omni_dataset import OmniDataset
 from model.model_omni import MiniQwenOmni
 from trainer.trainer_utils import load_omni_tokenizer
 tok=load_omni_tokenizer(R/'model/Qwen3-0.6B');encoder,audio=MiniQwenOmni.load_sensevoice(R/'model/SenseVoiceSmall');del encoder
 vision=SiglipImageProcessor.from_pretrained(R/'model/siglip2-base-p32-256-ve',local_files_only=True)
 catalog=json.loads((M/'manifest.json').read_text());checks={}
 for name,path in samples.items():
  source,split=name.split('/');stage=json.loads((M/'sources'/f'{source}.json').read_text())['stage'];training=split=='train'
  ds=OmniDataset(path,tok,audio,vision,max_length=768,training=training,augment=False,scheduled_sampling=0,strict_length=True,condition_prefix_length=80 if stage=='AOA' else 0,voice_conditioning='available_mixed' if training else 'available_both',use_modality_boundaries=True,use_talker_ref_boundaries=True)
  details=[]
  for i in range(len(ds)):
   ids,mask,text,codes,features,nframes,pixels,speaker=ds[i];row=ds.dataset[i]
   assert text.ne(-100).any(),row['sample_id']
   if stage=='AIA':
    assert nframes>0 and ids[8].eq(ds.audio_token_id).any()
    if row['task_type']=='asr':assert json.loads(row['conversations'])[0]['content'] in tok.decode(ids[8].tolist(),skip_special_tokens=False)
    assert not codes.ne(-100).any()
   if stage=='IIA':assert pixels is not None and pixels.any() and int(ids[8].eq(ds.image_token_id).sum())==64
   if stage=='AOA':
    assert int(codes[0].ne(-100).sum())==row['codec_frames']+1
    assert int(codes[1].ne(-100).sum())==row['codec_frames']
    assert int(codes[0].eq(ds.audio_stop_token).sum())==1
    assert nframes==0
   details.append({'id':row['sample_id'],'text_tokens':int(text.ne(-100).sum()),'codec_frames':row['codec_frames'],'input_audio_frames':int(nframes)})
  checks[name]=details;print('READER_PASS',name,len(details),flush=True)
 # Real mixed-source batches must also share one Arrow schema.
 for stage in ['AIA','IIA','AOA']:
  selected=[pq.read_table(p) for n,p in samples.items() if n.endswith('/train') and json.loads((M/'sources'/f'{n.split("/")[0]}.json').read_text())['stage']==stage]
  path=W/'reader-inputs'/f'{stage}-mixed.parquet';pq.write_table(pa.concat_tables(selected),path,compression='zstd')
  ds=OmniDataset(str(path),tok,audio,vision,max_length=768,training=True,augment=False,scheduled_sampling=0,strict_length=True,condition_prefix_length=80 if stage=='AOA' else 0,voice_conditioning='available_mixed',use_modality_boundaries=True,use_talker_ref_boundaries=True)
  for i in range(len(ds)):assert ds[i][2].ne(-100).any()
  print('MIXED_READER_PASS',stage,len(ds),flush=True)
 save(W/'reader-validation.json',{'status':'PASS','checks':checks,'mixed_stages':['AIA','IIA','AOA'],'scope':'Representative first/middle/last per source/split; no model updates or training.'})



def write_report(results,removed):
 catalog=json.loads((M/'manifest.json').read_text());stages=collections.defaultdict(lambda:collections.Counter())
 for name,m in results.items():
  for split,n in m['actual_rows'].items():stages[m['stage']][split]+=n
 payload={'status':'DATA_READY_NO_TRAINING_STARTED','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'layout':{'raw':'dataset/raw','pretrain':['dataset/pretrain/AIA','dataset/pretrain/IIA','dataset/pretrain/AOA'],'sft':'dataset/sft'},'sources':results,'stages':{s:dict(v) for s,v in stages.items()},'plan':'dataset/raw/manifests/pretrain/plan.json','cleanup_logical_GiB':removed/2**30,'sft_processing':'Only previous external Parquet pools migrated byte-for-byte; no new mixed SFT views or Mimi encoding.30% supplement reservations are selector indices, not SFT training views.','validation':['Full row conservation and identical unified schema','Speaker/phase and LibriSpeech-LibriTTS cross-corpus partition checks','Source reference speaker/phase checks during full build','Representative real OmniDataset single/mixed-source reads','Formal launcher preflight hashes/row counts; no --run'],'limitations':['No new full perceptual/semantic benchmark decontamination certification.','Original length budget768 and previously qualified legacy views retained, originals/exclusion histories preserved.','Raw corpora include retained prepared imports; ordinary raw TTS waveforms removed in older authorized cleanup were not recreated.','Historical references may be invalid by user preference; final SFT paths/schema will be built later. No compatibility directory tree retained.']}
 save(R/'reports/PRETRAIN_DATA_V4_READY.json',payload)
 lines=['# Pretrain 混合数据 V4 整理与验收','',f'状态：数据就绪，未启动训练。完成：{payload["created_at"]}。','', '| 阶段 | 训练样本 | 开发样本 | 来源 |','|---|---:|---:|---|']
 for stage in ['AIA','IIA','AOA']:
  names=[n for n,v in results.items() if v['stage']==stage];lines.append(f'| {stage} | {stages[stage]["train"]:,} | {stages[stage]["dev"]:,} | '+', '.join(names)+' |')
 lines+=['','## 补充数据划分','', '| 来源 | pretrain train | SFT预留 | dev |','|---|---:|---:|---:|']
 for name in ['librispeech','aishell1','libritts_r','aishell3']:
  x=results[name];lines.append(f'| {name} | {x["actual_rows"]["train"]:,} | {x["sft_reserved"].get("rows",0):,} | {x["actual_rows"]["dev"]:,} |')
 lines+=['','按说话人稳定哈希约70%/30%划分，因此实际行数/小时比不会恰好70/30；LibriSpeech/LibriTTS-R共享Librivox说话人命名空间。没有旧/新占比或重复采样配额。AIA含旧A2A音频→文字与显式ASR；IIA含旧I2T与LLaVA-Pretrain；AOA含旧T2A与真人TTS/独立音色条件，不含旧A2A语音输出。','', '## 目录与归档','', '`dataset/raw/legacy/`保留三份完整旧原件；`raw/corpora/`保存现存补充语料源Parquet；`raw/manifests/pretrain/`保存划分索引、训练计划、来源及清理记录。受保护的异常原件和完整编码整理到raw/audit，评测资料整理到raw/evaluation_assets；旧版本工作目录删除，不修复历史失效引用。既有SFT池迁至`dataset/sft/external/`，本轮没有新处理SFT。','', 'pretrain根目录仅AIA、IIA、AOA；各目录按来源和train/dev分片，均为统一schema、内嵌媒体/codec的Parquet。保留完整目标，不裁断语音。此前普通TTS原波形已清理，本次保留的源是现存编码/条件Parquet，没有声称恢复原波形。','',f'验收通过后清理旧派生视图、mini子集和废弃数据缓存约{removed/2**30:.2f}GiB（逻辑文件大小），模型、checkpoint、三份完整旧原件及评测数据保留。历史plan和旧版本目录已清理，旧路径不再支持续训；新计划为`{payload["plan"]}`。','', '原长度预算768和合格旧视图沿用，先前排除的超长/无图文本等仍保留在原件，不把它们悄悄塞入对齐训练。所有阶段最多1epoch，当前只做数据和reader验收，未进行权重更新。','', 'SFT旧Parquet的record_json未重写，内部旧媒体路径留待SFT阶段处理；现有媒体集中在raw/external/{fusion,m4}，不保留废弃版本目录或兼容映射树。未新增全量跨库感知去污染认证。','', '验收证据：`.runtime/pretrain-data/metadata-validation.json`、`reader-validation.json`、`preflight.log`；清理证据：`dataset/raw/manifests/pretrain/cleanup.json`。']
 (R/'reports/PRETRAIN_DATA_V4_READY.md').write_text('\n'.join(lines)+'\n')
 save(W/'status.json',{'status':'DATA_READY_NO_TRAINING_STARTED','finished_unix':time.time(),'report':'reports/PRETRAIN_DATA_V4_READY.md','plan':payload['plan']})


def refresh_active_layout_and_plan():
 from simplify_dataset_storage import simplify_external
 simplify_external(R,M,save)
 archive=D/'raw/_migration_archive';planpath=M/'plan.json';plan=json.loads(planpath.read_text());prior=json.loads((archive/'pretrain_v2/pretrain_plan.json').read_text());inv=json.loads((W/'inventory.json').read_text());audit={}
 for source,spec in json.loads((M/'manifest.json').read_text())['sources'].items():
  original=(inv['parent_plan']['length_audit'] if source.startswith('old_') else prior['length_audit'])[source]
  audit[source]={split:{'rows':stats['rows'],'max_required_upper_bound':original[split]['max_required'],'basis':'Inherited verified budget; only schema metadata and phase selection changed.'} for split,stats in spec['splits'].items()}
 plan['length_audit']=audit
 plan['code_sha256']={name:digest(R/name) for name in plan['code_sha256']}
 save(planpath,plan)


def main():
 status('WAITING_FOR_BUILD')
 while True:
  p=W/'status.json'
  if p.exists():
   s=json.loads(p.read_text())
   if s['status']=='DATA_READY_NO_TRAINING_STARTED':
    status('COMPLETE',report='reports/PRETRAIN_DATA_V4_READY.md');print('PRETRAIN_DATA_V4_ALREADY_READY_NO_TRAINING',flush=True);return
   if s['status']=='FAILED':raise RuntimeError('Builder failed:'+s.get('error',''))
   if s['status']=='BUILT_AWAITING_VALIDATION':break
  time.sleep(10)
 refresh_active_layout_and_plan()
 status('VALIDATING_METADATA');results,samples=metadata_checks()
 status('VALIDATING_READERS');reader_checks(samples)
 status('PREFLIGHT_NO_TRAINING')
 with (W/'preflight.log').open('w') as f:subprocess.run([sys.executable,str(R/'data_pipeline/validate_pretrain_assets.py'),'--plan',str(M/'plan.json')],cwd=R,env=os.environ,stdout=f,stderr=subprocess.STDOUT,check=True)
 status('PRESERVING_REQUIRED_TEST_AND_EXCEPTION_ASSETS')
 from simplify_dataset_storage import preserve_required,cleanup_simple
 preserve_required(R,M,save,digest)
 status('CLEANING_VERIFIED_DUPLICATES');removed=cleanup_simple(R,M,W,save,digest);write_report(results,removed);status('COMPLETE',report='reports/PRETRAIN_DATA_V4_READY.md');print('PRETRAIN_DATA_V4_READY_NO_TRAINING',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:status('FAILED',error=repr(e));raise
