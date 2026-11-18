#!/usr/bin/env python3
"""Validate/normalize existing SFT media, extract selected ZIP members and clean duplicates."""
import collections,concurrent.futures,hashlib,json,math,os,sys,time,zipfile
from pathlib import Path,PurePosixPath
from sft_data_utils import ROOT,BASE,save,sha
from sft_splitzip import SplitZipFile
from prepare_sft_external import normalize_messages,group_split

def safe_target(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe_media_path')
 out=root/str(p);out.parent.mkdir(parents=True,exist_ok=True)
 if not out.parent.resolve().is_relative_to(root.resolve()) or out.is_symlink():raise ValueError('unsafe_media_symlink')
 return out

def needed_m4():
 needed=collections.defaultdict(set)
 for line in (BASE/'source_metadata/m4-eligible-annotations.jsonl').open():
  for p in json.loads(line)['image']:needed[p.split('/')[0]].add(p)
 return needed

def validate_image(path):
 from PIL import Image
 with Image.open(path) as im:
  im.load()
  if min(im.size)<=0:raise ValueError('empty_image')
  return {'width':im.width,'height':im.height,'mode':im.mode}

def cleanup_archive(path,manifest):
 # Caller invokes this only after selected media have been fully read, CRC checked,
 # decoded and all outcomes written in a committed manifest.
 receipt=path.with_name(path.name+'.download.json')
 if not receipt.exists():raise ValueError('missing_verified_download_receipt')
 r=json.loads(receipt.read_text())
 if not r.get('upstream_sha256_verified') or r['size']!=path.stat().st_size:raise ValueError('unverified_archive')
 size=path.stat().st_size;path.unlink()
 return {'path':str(path.relative_to(ROOT)),'bytes':size,'receipt':str(receipt.relative_to(ROOT)),'media_manifest':str(manifest.relative_to(ROOT))}

def extract_m4(prefix,paths):
 needed=needed_m4()[prefix];out=BASE/'media/m4';manifest=BASE/'processed/m4/media-manifests'/f'{prefix}.json'
 if manifest.exists():return json.loads(manifest.read_text())
 zipobj=SplitZipFile(paths) if len(paths)>1 else zipfile.ZipFile(paths[0]);found={};failed=[];missing=[];total=0
 def extract(name,info,data):
  target=safe_target(out,name)
  if ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('symlink_in_archive')
  # ZIP bytes are read serially by the archive owner; decode/write is parallel.
  h=hashlib.sha256(data).hexdigest();tmp=target.with_name(target.name+'.partial');tmp.write_bytes(data);tmp.replace(target)
  try:shape=validate_image(target)
  except Exception as e:
   target.unlink();return name,None,{'path':name,'reason':type(e).__name__+':'+str(e)}
  return name,{'bytes':len(data),'sha256':h,**shape},None
 try:
  members={x.filename.removeprefix('./'):x for x in zipobj.infolist() if not x.is_dir()}
  tasks=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   # Bounded futures to avoid retaining decoded images in a large pending queue.
   pending=set()
   for name in sorted(needed):
    info=members.get(name)
    if info is None:missing.append(name);continue
    data=zipobj.read(info) # CRC verified before submitting independent bytes.
    pending.add(pool.submit(extract,name,info,data))
    if len(pending)>=12:
     done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
     for f in done:
      n,result,error=f.result()
      if error:failed.append(error)
      else:found[n]=result
   for f in concurrent.futures.as_completed(pending):
    n,result,error=f.result()
    if error:failed.append(error)
    else:found[n]=result
 finally:zipobj.close()
 report={'prefix':prefix,'needed':len(needed),'decoded':len(found),'files':found,'missing':missing,'failed':failed,'status':'MEDIA_AUDITED','input_archives':[str(p.relative_to(ROOT)) for p in paths]}
 save(manifest,report)
 deletions=[cleanup_archive(p,manifest) for p in paths]
 report['removed_archives']=deletions;save(manifest,report)
 print('EXTRACTED',prefix,'decoded',len(found),'missing',len(missing),'failed',len(failed),'deleted_GiB',round(sum(x['bytes'] for x in deletions)/2**30,2),flush=True)
 return report

class Groups:
 def __init__(self):self.parents={}
 def find(self,x):
  p=self.parents.setdefault(x,x)
  while p!=self.parents[p]:p=self.parents[p]
  while x!=p:n=self.parents[x];self.parents[x]=p;x=n
  return p
 def union(self,a,b):
  x,y=self.find(a),self.find(b)
  if x!=y:self.parents[max(x,y)]=min(x,y)

def normalize_m4():
 out=BASE/'processed/m4';out.mkdir(parents=True,exist_ok=True);groups=Groups();valid=set();bad=set();subsets=collections.Counter();counts=collections.Counter();excluded=collections.Counter();seen=set()
 for p in (out/'media-manifests').glob('*.json'):
  m=json.loads(p.read_text());valid.update(m['files']);bad.update(m['missing']);bad.update(x['path'] for x in m['failed'])
 # Union all shared-image records before assigning dev to prevent image leakage.
 for line in (BASE/'source_metadata/m4-eligible-annotations.jsonl').open():
  row=json.loads(line);images=row['image']
  if not all(p in valid for p in images):continue
  for p in images[1:]:groups.union(images[0],p)
 handles={s:(out/(s+'.jsonl')).open('w') for s in ['train','dev']}
 try:
  for idx,line in enumerate((BASE/'source_metadata/m4-eligible-annotations.jsonl').open()):
   row=json.loads(line);counts['candidate_rows']+=1;images=row['image']
   if not all(p in valid for p in images):excluded['missing_or_invalid_image']+=1;continue
   messages=normalize_messages(row['conversations']);sub=row['metadata']['dataset']
   digest=hashlib.sha256(json.dumps([images,messages],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   if digest in seen:excluded['exact_sample_duplicate']+=1;continue
   seen.add(digest);group=groups.find(images[0]);split=group_split(group);counts[split]+=1;counts[f'{len(images)}_images']+=1;subsets[sub]+=1
   record={'sample_id':f'm4:{sub}:{row.get("sample_id",idx)}:{idx}','source':'m4','source_subset':sub,'task_type':'multi_image_qa','split':split,'group_id':group,'conversations':messages,'image_paths':[str((BASE/'media/m4'/p).relative_to(ROOT)) for p in images],'metadata':row.get('metadata',{})}
   handles[split].write(json.dumps(record,ensure_ascii=False)+'\n')
 finally:
  for f in handles.values():f.close()
 save(out/'manifest.json',{'status':'MEDIA_VALIDATED_NORMALIZED_CODEC_NOT_NEEDED','counts':dict(counts),'excluded':dict(excluded),'subsets':dict(subsets),'unique_valid_images':len(valid),'policy':'Source train,2-4 still images; image connected components shared by train/dev. No target/media truncation; no token budget yet. Full semantic/perceptual benchmark contamination audit not claimed.'})


def normalize_fusion():
 import numpy as np
 import soundfile as sf
 out=BASE/'processed/fusion';out.mkdir(parents=True,exist_ok=True);root=BASE/'raw/fusion';rows=[json.loads(s) for s in (root/'metadata.jsonl').read_text().splitlines() if s.strip()]
 counts=collections.Counter();failures=[];seen=set();media={};valid=[];durs=[]
 for r in rows:
  counts['raw_rows']+=1
  try:
   if not r['user_prompt'].strip() or not r['target_text'].strip():raise ValueError('empty_text')
   for k in ['image_path','input_audio_path','target_audio_path']:
    name=r[k];p=root/name
    if name in media:continue
    if k=='image_path':media[name]={'bytes':p.stat().st_size,**validate_image(p)}
    else:
     audio,sr=sf.read(p,dtype='float32',always_2d=True)
     if not len(audio) or not np.isfinite(audio).all() or sr<=0:raise ValueError('empty_or_nonfinite_audio')
     media[name]={'bytes':p.stat().st_size,'sample_rate':sr,'samples':len(audio),'channels':audio.shape[1],'seconds':len(audio)/sr}
   digest=hashlib.sha256(json.dumps([r['image_path'],r['user_prompt'],r['target_text']],ensure_ascii=False).encode()).hexdigest()
   if digest in seen:raise ValueError('exact_sample_duplicate')
   seen.add(digest);split=group_split(r['image_path']);counts[split]+=1
   counts['input_seconds']+=media[r['input_audio_path']]['seconds'];counts['target_seconds']+=media[r['target_audio_path']]['seconds']
   durs.append(media[r['target_audio_path']]['seconds'])
   valid.append({'sample_id':'fusion:'+r['id'],'source':'dreyn74/llava_instruct_9k','task_type':'image_audio_qa_pool','split':split,'group_id':r['image_path'],'conversations':[{'role':'user','content':'<image>\n'+r['user_prompt']},{'role':'assistant','content':r['target_text']}],'image_paths':[str((root/r['image_path']).relative_to(ROOT))],'question_audio_path':str((root/r['input_audio_path']).relative_to(ROOT)),'answer_audio_path':str((root/r['target_audio_path']).relative_to(ROOT)),'input_seconds':media[r['input_audio_path']]['seconds'],'target_seconds':media[r['target_audio_path']]['seconds'],'available_views':['I+Q->R','I+Aq->R','I+Q->R+Ar','I+Aq->R+Ar']})
  except Exception as e:failures.append({'sample_id':r.get('id'),'reason':type(e).__name__+':'+str(e),'original':r})
 handles={s:(out/(s+'.jsonl')).open('w') for s in ['train','dev']}
 for row in valid:handles[row['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
 for f in handles.values():f.close()
 save(out/'media-audit.json',{'files':media,'failures':failures})
 durs.sort();save(out/'manifest.json',{'status':'RAW_AUDIO_VALIDATED_NORMALIZED_MIMI_PENDING','counts':dict(counts),'excluded_rows':len(failures),'unique_media_files':len(media),'target_duration':{'max':max(durs,default=0),'p50':durs[len(durs)//2] if durs else 0},'canonical_annotation':'metadata.jsonl','policy':'8900-row complete mapping only; group split by original image; no audio synthesis, target truncation or duplication across4views. Future Mimi encoding needed before training. All selected waveforms fully decoded and finite.'})
 print('FUSION_PROCESSED',dict(counts),'excluded',len(failures),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('operation',choices=['fusion','m4']);a=p.parse_args()
 normalize_fusion() if a.operation=='fusion' else normalize_m4()
