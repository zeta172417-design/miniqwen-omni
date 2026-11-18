#!/usr/bin/env python3
"""Prepare the M4-recommended LLaVA-1.5 single-image instruction pool."""
import collections,concurrent.futures,hashlib,io,json,os,random,sys,time,zipfile,tarfile
from pathlib import Path,PurePosixPath
import requests
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'data_pipeline'))
from sft_data_utils import json_array,save,sha
RAW=R/'dataset/raw/external/llava15';META=R/'dataset/raw/manifests/llava15';W=R/'.runtime/sft-llava15'
for p in (RAW,META,W):p.mkdir(parents=True,exist_ok=True)
def client():
 s=requests.Session();s.trust_env=False;return s

def fetch(url,path,size,expected):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size==size and sha(path)==expected:return path
 partial=path.with_name(path.name+'.partial')
 for attempt in range(8):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   if offset>size:partial.unlink();offset=0
   headers={'Accept-Encoding':'identity'}
   if offset:headers['Range']=f'bytes={offset}-'
   if offset<size:
    with client().get(url,headers=headers,stream=True,timeout=(15,90)) as res:
     res.raise_for_status()
     if offset and res.status_code!=206:offset=0
     if offset and not res.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Bad resume range')
     with partial.open('ab' if offset else 'wb') as f:
      for chunk in res.iter_content(1024**2):
       if chunk:f.write(chunk)
   if partial.stat().st_size!=size:raise ValueError('Incomplete download')
   actual=sha(partial)
   if actual!=expected:partial.unlink();raise ValueError('SHA256 mismatch')
   partial.replace(path);save(path.with_name(path.name+'.download.json'),{'url':url,'size':size,'sha256':actual,'verified':True});return path
  except Exception as exc:
   print('DOWNLOAD_RETRY',path.name,attempt,repr(exc),flush=True)
   if attempt==7:raise
   time.sleep(min(2**attempt,20))

def annotation():
 tree=json.loads((META/'official-tree.json').read_text());official=next(x for x in tree if x['path']=='llava_v1_5_mix665k.json')
 mirror=json.loads((META/'modelscope-tree.json').read_text());m=next(x for x in mirror['Data']['Files'] if x['Name']==official['path'])
 assert m['Size']==official['size'] and m['Sha256']==official['lfs']['oid']
 url=f'https://modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Instruct-150K/repo?Revision={m["Revision"]}&FilePath={m["Path"]}'
 path=fetch(url,RAW/official['path'],official['size'],official['lfs']['oid'])
 save(META/'annotation-source.json',{'official_repo':'liuhaotian/LLaVA-Instruct-150K','official_revision':json.loads((META/'official-hub.json').read_text())['sha'],'file':official,'mirror_revision':m['Revision'],'mirror_url':url})
 print('ANNOTATION_VERIFIED',path,flush=True);return path

def normalize(row):
 image=row.get('image')
 if not isinstance(image,str) or not image.strip():raise ValueError('no_single_image')
 p=PurePosixPath(image)
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe_image_path')
 messages=[]
 for m in row.get('conversations',[]):
  role={'human':'user','gpt':'assistant'}.get(m.get('from',m.get('role')),m.get('role'))
  text=m.get('value',m.get('content'))
  if role not in ('system','user','assistant') or not isinstance(text,str) or not text.strip():raise ValueError('invalid_conversation')
  if messages and messages[-1]['role']==role:raise ValueError('invalid_roles')
  messages.append({'role':role,'content':text})
 if not messages or messages[-1]['role']!='assistant' or next((m['role'] for m in messages if m['role']!='system'),None)!='user':raise ValueError('invalid_endpoints')
 if sum(m['content'].count('<image>') for m in messages)!=1:raise ValueError('image_marker_count')
 if not any(m['role']=='user' and '<image>' in m['content'] for m in messages):raise ValueError('image_marker_not_in_user')
 return image,messages

def select(path,n=307000):
 import pyarrow as pa
 import pyarrow.parquet as pq
 counts=collections.Counter();reasons=collections.Counter();seen=set();candidates=[]
 # Store only compact candidate metadata in memory; write full rows in second pass.
 for index,row in enumerate(json_array(path)):
  counts['raw_rows']+=1
  try:
   image,messages=normalize(row);digest=hashlib.sha256(json.dumps([image,messages],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   if digest in seen:raise ValueError('exact_duplicate')
   seen.add(digest);rank=hashlib.sha256(('llava15-single:42:'+digest).encode()).digest();candidates.append((rank,index,image));counts['eligible']+=1
  except ValueError as exc:reasons[str(exc)]+=1
 assert len(candidates)>=n
 chosen=sorted(candidates)[:n];selected={i for _,i,_ in chosen};images=sorted({p for _,_,p in chosen});by_source=collections.Counter(p.split('/')[0] for _,_,p in chosen)
 out=RAW/'selected-307k.parquet';tmp=out.with_suffix('.parquet.partial');writer=None;buffer=[]
 schema=pa.schema([('source_row',pa.int64()),('record_json',pa.string())])
 try:
  for index,row in enumerate(json_array(path)):
   if index not in selected:continue
   buffer.append({'source_row':index,'record_json':json.dumps(row,ensure_ascii=False)})
   if len(buffer)==512:
    table=pa.Table.from_pylist(buffer,schema=schema)
    if writer is None:writer=pq.ParquetWriter(tmp,schema,compression='zstd')
    writer.write_table(table);buffer=[]
  if buffer:
   if writer is None:writer=pq.ParquetWriter(tmp,schema,compression='zstd')
   writer.write_table(pa.Table.from_pylist(buffer,schema=schema))
 finally:
  if writer:writer.close()
 assert pq.ParquetFile(tmp).metadata.num_rows==n;tmp.replace(out)
 save(META/'selection.json',{'status':'SELECTED_MEDIA_PENDING','seed':42,'sample_rows':n,'raw_counts':dict(counts),'excluded':dict(reasons),'selected_rows_by_image_source':dict(by_source),'unique_images':len(images),'needed_images':images,'source_annotation':str(path.relative_to(R)),'selected_annotation':str(out.relative_to(R)),'policy':'Deterministic hash-ranked uniform subset of valid unique single-image conversations. Full conversations retained; no text rewriting, no single-turn expansion, no length filtering yet. Selection307K before train/dev image-group split and media validation.'})
 print('SELECTED',n,'images',len(images),'sources',dict(by_source),flush=True)
 return images

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--annotation-only',action='store_true');p.add_argument('--select',action='store_true');args=p.parse_args()
 save(W/'status.json',{'status':'DOWNLOADING_ANNOTATION','pid':os.getpid()})
 path=annotation()
 if args.select:
  save(W/'status.json',{'status':'SELECTING_SINGLE_IMAGE_RECORDS','pid':os.getpid()});select(path)
 save(W/'status.json',{'status':'ANNOTATION_SELECTED_MEDIA_PENDING' if args.select else 'ANNOTATION_DOWNLOADED','pid':os.getpid()})
if __name__=='__main__':main()
