"""Pinned SFT downloads and bounded JSON readers; no training imports."""
import concurrent.futures,hashlib,json,os,threading,time
from pathlib import Path
import requests
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/sft_v1'
_local=threading.local()
def session():
 if not hasattr(_local,'session'):_local.session=requests.Session()
 return _local.session
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
@lru_cache(maxsize=8)
def load_hub(name):return json.loads((BASE/'source_metadata'/f'{name}-hub.json').read_text())
def sha(path,kind='sha256'):
 h=hashlib.new(kind)
 if kind=='sha1':h.update(f'blob {path.stat().st_size}\0'.encode())
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def download(name,file):
 hub=load_hub(name);path=BASE/'raw'/name/file['rfilename'];path.parent.mkdir(parents=True,exist_ok=True)
 expected=file.get('lfs',{}).get('sha256');git=file.get('blobId');size=file.get('size')
 if path.exists() and path.stat().st_size==size:
  if expected and sha(path)==expected:return path
  if not expected and git and sha(path,'sha1')==git:return path
  if not expected and not git:return path
 url=f'https://huggingface.co/datasets/{hub["id"]}/resolve/{hub["sha"]}/{file["rfilename"]}'
 if name=='fusion':url=url.replace('https://huggingface.co/','https://hf-mirror.com/')
 temp=path.with_name(path.name+'.partial')
 for attempt in range(8):
  try:
   offset=temp.stat().st_size if temp.exists() else 0
   if offset>size:raise ValueError('Oversized partial download')
   if offset<size:
    headers={'Accept-Encoding':'identity'}
    if offset:headers['Range']=f'bytes={offset}-'
    client=session()
    if name=='fusion':client.trust_env=False
    with client.get(url,headers=headers,stream=True,timeout=(20,90)) as r:
     r.raise_for_status()
     if offset and r.status_code!=206:offset=0
     elif offset and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Wrong range')
     with temp.open('ab' if offset else 'wb') as f:
      for chunk in r.iter_content(1024*1024):f.write(chunk)
   if temp.stat().st_size!=size:raise ValueError('Incomplete download')
   actual=sha(temp)
   if expected and actual!=expected:
    temp.unlink();raise ValueError('SHA256 mismatch')
   if not expected and git and sha(temp,'sha1')!=git:
    temp.unlink();raise ValueError('Git blob hash mismatch')
   temp.replace(path)
   return path
  except Exception:
   if attempt==7:raise
   time.sleep(min(2**attempt,20))
def download_many(name,files,workers=12):
 start=time.monotonic();last=start;done=0;errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(download,name,f):f for f in files}
  for future in concurrent.futures.as_completed(futures):
   f=futures[future]
   try:future.result()
   except Exception as exc:errors.append({'path':f['rfilename'],'error':repr(exc)})
   done+=1
   if time.monotonic()-last>25 or done==len(files):
    print('DOWNLOAD',name,done,'/',len(files),'errors',len(errors),'seconds',round(time.monotonic()-start),flush=True);last=time.monotonic()
 save(BASE/'source_metadata'/f'{name}-downloads.json',{'revision':load_hub(name)['sha'],'selected_files':len(files),'completed':done-len(errors),'errors':errors})
 if errors:raise RuntimeError(f'{name}: {len(errors)} download failures, see receipt')
def json_array(path):
 """Iterate a top-level JSON array without loading a gigabyte annotation list."""
 decoder=json.JSONDecoder();buf='';pos=0;eof=False;started=False
 with Path(path).open() as f:
  while True:
   if pos>=len(buf) or len(buf)-pos<1024 and not eof:
    buf=buf[pos:]+f.read(1024*1024);pos=0
    if not buf:break
   while pos<len(buf) and buf[pos].isspace():pos+=1
   if not started:
    if pos==len(buf):continue
    if buf[pos]!='[':raise ValueError('Expected JSON array')
    pos+=1;started=True;continue
   while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
   if pos<len(buf) and buf[pos]==']':return
   try:item,end=decoder.raw_decode(buf,pos)
   except json.JSONDecodeError:
    chunk=f.read(1024*1024)
    if not chunk:raise
    buf=buf[pos:]+chunk;pos=0;continue
   yield item;pos=end
  if started:raise ValueError('Incomplete JSON array')
