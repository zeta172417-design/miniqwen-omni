#!/usr/bin/env python3
"""Continue selected external SFT preparation; no model training or speech generation."""
import collections,concurrent.futures,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from sft_data_utils import ROOT,BASE,load_hub,save,sha
from prepare_sft_media import extract_m4,normalize_m4,normalize_fusion
from prepare_sft_external import text_source
W=ROOT/'.runtime/sft-data-v1'
LOCK=(W/'processing.lock').open('w');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)

def status(state,**kw):save(W/'status.json',{'status':state,'pid':os.getpid(),'updated_unix':time.time(),**kw})
def archive_ready(path):
 receipt=path.with_name(path.name+'.download.json')
 return path.exists() and receipt.exists() and json.loads(receipt.read_text()).get('upstream_sha256_verified')
def cleanup_text_raw(name):
 out=BASE/'processed'/name;m=json.loads((out/'manifest.json').read_text());counts=m['counts']
 if counts.get('train',0)+counts.get('dev',0)+sum(m['excluded'].values())!=counts['raw_rows']:raise ValueError('Text row accounting mismatch')
 # Rejections preserve their original records. Normalized records retain source fields.
 for line in (out/'excluded.jsonl').open():
  if 'original' not in json.loads(line):raise ValueError('Cannot remove source without original rejected rows')
 deletions=[]
 for f in load_hub(name)['siblings']:
  relative=f['rfilename']
  selected=relative.startswith('data/train_sft-') if name=='ultrachat' else relative=='COIG-CQIA-full.jsonl'
  if not selected:continue
  p=BASE/'raw'/name/relative
  if not p.exists():continue
  actual=sha(p)
  if f.get('lfs',{}).get('sha256')!=actual:raise ValueError('Raw cleanup hash mismatch')
  deletions.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':actual,'reason':'All source rows preserved in normalized or excluded records'})
  p.unlink()
 save(out/'raw-cleanup.json',{'removed':deletions})

def report(state):
 sources={};deletions=[]
 for name in ['ultrachat','coig','fusion','m4']:
  p=BASE/'processed'/name/'manifest.json'
  if p.exists():sources[name]=json.loads(p.read_text())
  p=BASE/'processed'/name/'raw-cleanup.json'
  if p.exists():deletions+=json.loads(p.read_text())['removed']
 for p in (BASE/'processed/m4/media-manifests').glob('*.json'):
  deletions+=json.loads(p.read_text()).get('removed_archives',[])
 summary={'status':state,'updated_unix':time.time(),'sources':sources,'deleted_duplicate_bytes':sum(x['bytes'] for x in deletions),'deletions':deletions,'training_ready':False,'remaining_after_quantity_preparation':['Select token/codec length budget and task mix after user reviews counts','Mimi encode validated external answer waveforms after quantity choices','Full benchmark media/perceptual near-duplicate audit and SFT reader/device acceptance'],'pretrain_unchanged':True}
 save(BASE/'manifest.json',summary);save(ROOT/'reports/SFT_EXTERNAL_DATA_STATUS.json',summary)
 lines=['# SFT 外部数据准备实际统计','',f'状态：{state}。当前pretrain不改动；本次不合成语音、不启动SFT。','', '| 数据源 | 原始/候选 | train | dev | 排除 | 状态 |','|---|---:|---:|---:|---:|---|']
 for name,m in sources.items():
  c=m['counts'];raw=c.get('raw_rows',c.get('candidate_rows',0));excluded=m.get('excluded_rows',sum(m.get('excluded',{}).values()))
  lines.append(f'| {name} | {raw:,} | {c.get("train",0):,} | {c.get("dev",0):,} | {excluded:,} | {m["status"]} |')
 lines+=['',f'已清理重复文件：{summary["deleted_duplicate_bytes"]/2**30:.2f} GiB；明细见JSON。压缩包在上游哈希、ZIP CRC、所需图片解码以及媒体清单提交后删除。纯文本源在所有行已保存到规范化/隔离记录后删除。','', '计数尚未应用模型token/codec长度预算，也未证明完成跨数据集语义/感知去污染。文本按首个用户问题分组；M4按共享图片连通组；融合数据按图片划分。train/dev来自各自选定训练池，未使用官方test作为训练池。','', 'M4主注释615,814条，默认候选272,810条：只取明确train、2–4静态图且角色/标记合法的记录；full/val、视频/3D和超图片数另列。全部候选媒体核验后才有最终可用数。不同图片任务分布和实际保留数应在配比选择时共同考虑。','', '融合源采用8900条完整主清单，按记录引用选择媒体（可能分布在两个原始音频目录），不导入另一份缺失较多的10000条清单，不把四类任务视图计成四倍独立语料。回答音频是外部已有波形，本轮仅解码统计，Mimi编码待后续长度/配额选择。','', '规范化JSONL保留原文、来源及媒体路径，便于本轮做数量取舍；尚非现有OmniDataset可直接训练的最终Parquet。不会以本报告声称SFT训练就绪。']
 (ROOT/'reports/SFT_EXTERNAL_DATA_STATUS.md').write_text('\n'.join(lines)+'\n')
 return summary

def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False

try:
 status('PROCESSING_TEXT')
 for name in ['ultrachat','coig']:
  # These downloads were already complete before this coordinator started.
  if not (BASE/'processed'/name/'raw-cleanup.json').exists():text_source(name);cleanup_text_raw(name)
 report('DOWNLOADING_AND_PROCESSING_MEDIA')
 needed=json.loads((BASE/'source_metadata/m4-media-selection.json').read_text())['prefixes']
 pending=set(needed);fusion_done=False;last_report=0;download_watch_start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={}
  while pending or futures or not fusion_done:
   for prefix in sorted(pending):
    manifest=BASE/'processed/m4/media-manifests'/f'{prefix}.json'
    if manifest.exists():pending.remove(prefix);continue
    paths=[BASE/'raw/m4'/f'{prefix}.zip'] if prefix!='dreamsim' else [BASE/'raw/m4/dreamsim_split.z01',BASE/'raw/m4/dreamsim_split.zip']
    if len(futures)<2 and all(archive_ready(p) for p in paths):
     futures[pool.submit(extract_m4,prefix,paths)]=prefix;pending.remove(prefix)
   for future in list(futures):
    if future.done():
     future.result();del futures[future]
   receipt=BASE/'source_metadata/fusion-downloads.json'
   if not fusion_done and receipt.exists():
    d=json.loads(receipt.read_text())
    if d.get('errors'):
     # Retry only failed files, preserving the full download accounting.
     from sft_data_utils import download_many
     hub=load_hub('fusion');failed={x['path'] for x in d['errors']}
     download_many('fusion',[x for x in hub['siblings'] if x['rfilename'] in failed],workers=16)
     d.update(completed=d['selected_files'],errors=[]);save(receipt,d)
    if not d.get('errors') and d.get('completed')==d.get('selected_files'):
     status('VALIDATING_FUSION',pending_m4=sorted(pending));normalize_fusion();fusion_done=True;report('DOWNLOADING_AND_PROCESSING_MEDIA')
   if time.time()-last_report>30:
    status('DOWNLOADING_AND_PROCESSING_MEDIA',pending_m4=sorted(pending),extracting=list(futures.values()),fusion_processed=fusion_done)
    report('DOWNLOADING_AND_PROCESSING_MEDIA');last_report=time.time()
   if pending or futures or not fusion_done:time.sleep(10)
 normalize_m4()
 # Delete duplicate fusion annotations only after validated row outputs exist.
 root=BASE/'raw/fusion';cleanup=[]
 for name in ['llava_instruct_9k.json','metadata_9k.jsonl']:
  p=root/name
  if p.exists():cleanup.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)});p.unlink()
 save(BASE/'processed/fusion/raw-cleanup.json',{'removed':cleanup})
 # Video annotations are outside this selected still-image scope; keep their hash only.
 p=BASE/'raw/m4/m4_instruct_video.json'
 if p.exists():save(BASE/'source_metadata/unused-video-annotation.json',{'bytes':p.stat().st_size,'sha256':sha(p),'reason':'outside static image scope'});p.unlink()
 summary=report('QUANTITY_PREPARATION_COMPLETE');status('QUANTITY_PREPARATION_COMPLETE',report='reports/SFT_EXTERNAL_DATA_STATUS.md')
 print('SFT_QUANTITY_PREPARATION_COMPLETE',json.dumps({k:v['counts'] for k,v in summary['sources'].items()},ensure_ascii=False),flush=True)
except BaseException as e:
 status('FAILED',error=repr(e));report('FAILED');traceback.print_exc();raise
