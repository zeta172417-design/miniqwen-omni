#!/usr/bin/env python3
"""Download pinned original-image mirrors, retain selected media, emit SFT Parquet."""
import collections,hashlib,io,json,os,sys,time,zipfile,tarfile,zlib,concurrent.futures
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'data_pipeline'))
from prepare_llava15_single_sft import RAW,META,W,fetch,normalize
from sft_data_utils import save,sha
from sft_range_download import fetch_ranges

def bucket(path):
 p=path.split('/')
 if p[0]=='vg':return p[1]
 return p[0]

def inspect_image(data):
 with Image.open(io.BytesIO(data)) as im:
  im.load()
  if min(im.size)<=0:raise ValueError('Empty image')
  return {'width':im.width,'height':im.height,'mode':im.mode}

def extract(spec,needed):
 name=spec['Name'];report=META/'media-audits'/(name+'.parquet');summary=report.with_suffix('.json')
 if summary.exists() and json.loads(summary.read_text()).get('status')=='PASS':return json.loads(summary.read_text())
 url=f'https://modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision={spec["Revision"]}&FilePath={name}'
 path=RAW/'downloads'/name
 receipt=path.with_name(path.name+'.download.json')
 committed=json.loads(receipt.read_text()) if receipt.exists() else {}
 if not (path.exists() and committed.get('verified') and committed.get('sha256')==spec['Sha256'] and path.stat().st_size==spec['Size'] and path.stat().st_mtime<=receipt.stat().st_mtime):
  path=fetch_ranges(url,path,spec['Size'],spec['Sha256'],workers=int(os.environ.get('LLAVA_RANGE_WORKERS','3')))
 print('ARCHIVE_VERIFIED',name,flush=True)
 bybase=collections.defaultdict(list)
 for key in needed:bybase[Path(key).name].append(key)
 records=[];failures=[];missing=set(needed);seen=set()
 parents={(RAW/'images'/key).parent for key in needed}
 existing_targets=set()
 for parent in parents:
  parent.mkdir(parents=True,exist_ok=True)
  existing_targets.update(str((parent/n).relative_to(RAW/'images')) for n in os.listdir(parent))
 fusion_dir=R/'dataset/raw/external/fusion/images'
 fusion_names=set(os.listdir(fusion_dir)) if fusion_dir.exists() else set()
 def consume(member,data,crc=None):
  basename=Path(member).name
  if basename not in bybase:return
  try:shape=inspect_image(data)
  except Exception as exc:
   for key in bybase[basename]:
    quarantine=R/'dataset/raw/audit/llava15'/key;quarantine.parent.mkdir(parents=True,exist_ok=True);quarantine.write_bytes(data)
    failures.append({'original_image':key,'archive_member':member,'reason':str(exc),'sha256':hashlib.sha256(data).hexdigest(),'quarantine_path':str(quarantine.relative_to(R))});missing.discard(key)
   return
  h=hashlib.sha256(data).hexdigest()
  for key in bybase[basename]:
   target=RAW/'images'/key
   # COCO fusion photos may already exist; reuse only if their bytes are identical.
   existing=R/'dataset/raw/external/fusion/images'/basename
   if key.startswith('coco/') and basename in fusion_names and existing.is_file() and existing.stat().st_size==len(data) and sha(existing)==h:
    actual=existing;reused=True
   else:
    if key not in existing_targets or target.stat().st_size!=len(data) or sha(target)!=h:
     tmp=target.with_name(target.name+'.partial');tmp.write_bytes(data);tmp.replace(target)
    actual=target;reused=False
   records.append({'original_image':key,'path':str(actual.relative_to(R)),'bytes':len(data),'sha256':h,'archive_member':member,'crc32':crc,'reused':reused,**shape});missing.discard(key)
 with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('LLAVA_EXTRACT_WORKERS','4'))) as pool:
  pending=set();queued=0
  def dispatch(member,data,crc):
   nonlocal pending,queued
   basename=Path(member).name
   if basename in seen:raise ValueError('Ambiguous repeated image basename:'+member)
   seen.add(basename);queued+=1
   pending.add(pool.submit(consume,member,data,crc))
   if len(pending)>=16:
    done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
    for task in done:task.result()
   if queued%5000==0:print('MEDIA_PROGRESS',name,len(records),'/',len(needed),flush=True)
  if name.endswith('.zip'):
   with path.open('rb',buffering=8*1024**2) as archive_stream, zipfile.ZipFile(archive_stream) as z:
    for item in z.infolist():
     if item.is_dir() or Path(item.filename).name not in bybase:continue
     if ((item.external_attr>>16)&0o170000)==0o120000:raise ValueError('Image symlink in archive')
     dispatch(item.filename,z.read(item),item.CRC)
  else:
   with path.open('rb',buffering=8*1024**2) as archive_stream, tarfile.open(fileobj=archive_stream,mode='r|*',bufsize=1024**2) as tar:
    for item in tar:
     if not item.isfile() or Path(item.name).name not in bybase:continue
     with tar.extractfile(item) as f:data=f.read()
     dispatch(item.name,data,zlib.crc32(data))
  for task in concurrent.futures.as_completed(pending):task.result()
 if missing:
  failures.extend({'original_image':key,'reason':'Selected image absent from verified archive'} for key in sorted(missing))
 save(META/'media-audits'/(name+'.failed.json'),{'failed_images':failures})
 report.parent.mkdir(parents=True,exist_ok=True);tmp=report.with_suffix('.parquet.partial');pq.write_table(pa.Table.from_pylist(records),tmp,compression='zstd');assert pq.ParquetFile(tmp).metadata.num_rows+len(failures)==len(needed);tmp.replace(report)
 result={'status':'PASS','archive':name,'url':url,'size':spec['Size'],'sha256':spec['Sha256'],'mirror_revision':spec['Revision'],'needed':len(needed),'decoded':len(records),'failed_images':failures,'reused':sum(r['reused'] for r in records),'retained_bytes':sum(r['bytes'] for r in records),'media_audit':str(report.relative_to(R)),'validation':'Pinned upstream archive SHA256, ZIP CRC (or TAR archive digest), every selected image audited; undecodable originals quarantined and excluded; media hashes recorded.'}
 save(summary,result)
 # Keep receipt and selected images, delete redundant original archive only after acceptance.
 path.unlink();result['archive_deleted']=True;save(summary,result)
 print('MEDIA_READY',name,len(records),'reused',result['reused'],'archive_deleted',flush=True);return result

def accepted_entries(media,quality,target=307000):
 import heapq
 from sft_data_utils import json_array
 selected=set();seen=set();rejected=[];replacements=[];count=0
 for batch in pq.ParquetFile(RAW/'selected-307k.parquet').iter_batches(batch_size=256):
  for entry in batch.to_pylist():
   selected.add(entry['source_row']);raw=json.loads(entry['record_json']);image,messages=normalize(raw)
   if image not in media:
    rejected.append({'source_row':entry['source_row'],'sample_id':raw.get('id'),'image':image,'reason':'Image failed media audit'});continue
   digest=hashlib.sha256(json.dumps([image,messages],ensure_ascii=False,sort_keys=True).encode()).hexdigest();seen.add(digest);count+=1;yield entry
 deficit=target-count
 if deficit:
  def spares():
   for index,raw in enumerate(json_array(RAW/'llava_v1_5_mix665k.json')):
    if index in selected:continue
    try:image,messages=normalize(raw)
    except ValueError:continue
    if image not in media:continue
    digest=hashlib.sha256(json.dumps([image,messages],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if digest in seen:continue
    seen.add(digest);rank=hashlib.sha256(('llava15-single:42:'+digest).encode()).digest()
    yield rank,index,json.dumps(raw,ensure_ascii=False),image
  candidates=heapq.nsmallest(deficit,spares())
  if len(candidates)!=deficit:raise ValueError('Insufficient valid existing-media replacements')
  for _,index,record,image in candidates:
   replacements.append({'source_row':index,'image':image});count+=1;yield {'source_row':index,'record_json':record,'quality_replacement':True}
 assert count==target
 quality.update(rejected_rows=rejected,replacement_rows=replacements,policy='Bad media are quarantined, never repaired or accepted with truncated-image loading. Fill only from original structurally valid, nonduplicate conversations whose images already passed media audit; same seeded hash ranking, no synthetic content or extra image downloads.')
 save(META/'quality-adjustment.json',quality)

def finalize():
 media={}
 for p in (META/'media-audits').glob('*.parquet'):
  for row in pq.read_table(p).to_pylist():
   if row['original_image'] in media:raise ValueError('Repeated media mapping')
   media[row['original_image']]=row
 selection=json.loads((META/'selection.json').read_text());assert set(media)<=set(selection['needed_images'])
 quality={}
 # Keep image groups intact, including exact-byte duplicate aliases within this pool.
 out=R/'dataset/sft/external/llava15_single';out.mkdir(parents=True,exist_ok=True);writers={};buffers={'train':[],'dev':[]};counts=collections.Counter();sources=collections.Counter();turns=collections.Counter()
 schema=pa.schema([('sample_id',pa.string()),('source',pa.string()),('task_type',pa.string()),('split',pa.string()),('group_id',pa.string()),('record_json',pa.string())])
 def write(split):
  if not buffers[split]:return
  if split not in writers:writers[split]=pq.ParquetWriter(out/(split+'.parquet.partial'),schema,compression='zstd')
  writers[split].write_table(pa.Table.from_pylist(buffers[split],schema=schema));buffers[split]=[]
 revision=json.loads((META/'official-hub.json').read_text())['sha']
 try:
  for entry in accepted_entries(media,quality):
   raw=json.loads(entry['record_json']);image,messages=normalize(raw);asset=media[image];group=asset['sha256'];split='dev' if int(hashlib.sha256(('llava15-image:'+group).encode()).hexdigest()[:12],16)%100==0 else 'train'
   sid=f'llava15:{raw.get("id",entry["source_row"])}:{entry["source_row"]}'
   record={'sample_id':sid,'source':'liuhaotian/LLaVA-Instruct-150K:llava_v1_5_mix665k','source_revision':revision,'source_row':entry['source_row'],'task_type':'single_image_qa','split':split,'group_id':group,'conversations':messages,'image_paths':[asset['path']],'metadata':{'official_image':image,'original_id':raw.get('id'),'image_sha256':group,'image_width':asset['width'],'image_height':asset['height'],'selection_seed':42,'m4_recommended_single_image_complement':True,'quality_replacement':entry.get('quality_replacement',False)}}
   buffers[split].append({'sample_id':sid,'source':'llava15_single','task_type':'single_image_qa','split':split,'group_id':group,'record_json':json.dumps(record,ensure_ascii=False)})
   counts[split]+=1;sources[image.split('/')[0]]+=1;turns[len(messages)//2]+=1
   if len(buffers[split])==256:write(split)
  for split in buffers:write(split)
 finally:
  for writer in writers.values():writer.close()
 for split in buffers:
  p=out/(split+'.parquet.partial');assert pq.ParquetFile(p).metadata.num_rows==counts[split];p.replace(out/(split+'.parquet'))
 assert sum(counts.values())==307000
 # Validate split isolation and path validity for every unique image, not every duplicate row.
 train=set(pq.read_table(out/'train.parquet',columns=['group_id']).column(0).to_pylist());dev=set(pq.read_table(out/'dev.parquet',columns=['group_id']).column(0).to_pylist());assert not train&dev
 paths=list({v['path'] for v in media.values()})
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for offset in range(0,len(paths),512):
   assert all(pool.map(lambda path:(R/path).is_file(),paths[offset:offset+512]))
 manifest={'status':'NORMALIZED_POOL_MEDIA_AUDITED_LENGTH_AND_FINAL_MIX_PENDING','counts':dict(counts),'selected_rows':307000,'quality_adjustment':quality,'unique_original_image_paths':len(media),'unique_image_byte_groups':len(train|dev),'image_source_rows':dict(sources),'conversation_turns':dict(turns),'source_revision':revision,'data_files':{s:str((out/(s+'.parquet')).relative_to(R)) for s in buffers},'raw_media_root':str((RAW/'images').relative_to(R)),'policy':'M4-recommended LLaVA1.5 stage2 single-image complement; exact307K deterministic subset after structural validation/dedup; complete conversations preserved, no row expansion or synthetic data; image-byte SHA256 groups split about1% dev. Initial selection retained in raw selected Parquet; any quality replacements logged with original full-annotation source rows. No training token-length budget yet; cross-pool image/dev alignment and benchmark decontamination must precede final SFT mixture.'}
 save(out/'manifest.json',manifest)
 save(W/'status.json',{'status':'COMPLETE_NORMALIZED_POOL_ONLY','manifest':str((out/'manifest.json').relative_to(R)),'counts':dict(counts),'no_training_started':True})
 print('LLAVA15_SINGLE_POOL_READY',json.dumps(manifest,ensure_ascii=False),flush=True)
 return manifest

def main():
 os.nice(10)
 while not (META/'selection.json').exists():
  time.sleep(5)
 selection=json.loads((META/'selection.json').read_text());needed=collections.defaultdict(set)
 for p in selection['needed_images']:needed[bucket(p)].add(p)
 tree=json.loads((META/'media-modelscope-tree.json').read_text());files={x['Name']:x for x in tree['Data']['Files']}
 save(W/'media-status.json',{'status':'DOWNLOADING_AND_EXTRACTING','pid':os.getpid(),'needed':{k:len(v) for k,v in needed.items()}})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  tasks={pool.submit(extract,files[k+('.tar' if k=='ocr_vqa' else '.zip')],v):k for k,v in needed.items()}
  for future in concurrent.futures.as_completed(tasks):
   result=future.result();print('MEDIA_SOURCE_PASS',result['archive'],flush=True)
 finalize()
 save(W/'media-status.json',{'status':'COMPLETE','manifest':'dataset/sft/external/llava15_single/manifest.json'})
if __name__=='__main__':
 try:main()
 except BaseException as exc:save(W/'media-status.json',{'status':'FAILED','error':repr(exc)});raise
