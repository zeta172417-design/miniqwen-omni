"""Resumable bounded HTTP range downloads, verified against a pinned SHA256."""
import concurrent.futures,json,os,time
from pathlib import Path
import requests
from sft_data_utils import save,sha

def fetch_ranges(url,path,size,expected,workers=3,chunk_size=32*1024**2):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size==size and sha(path)==expected:return path
 partial=path.with_name(path.name+'.partial');progress=path.with_name(path.name+'.ranges.json')
 count=(size+chunk_size-1)//chunk_size
 if progress.exists():
  state=json.loads(progress.read_text())
  if (state['size'],state['sha256'],state['chunk_size'])!=(size,expected,chunk_size):raise ValueError('Range download identity mismatch')
  if not partial.exists():raise ValueError('Missing partial range file')
  done=set(state['completed'])
 else:
  prefix=partial.stat().st_size if partial.exists() else 0
  if prefix>size:raise ValueError('Oversized partial')
  # Sequential partial files have a known committed prefix; retain complete chunks.
  done=set(range(prefix//chunk_size))
  state={'size':size,'sha256':expected,'chunk_size':chunk_size,'completed':sorted(done)}
  save(progress,state)
 fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,size)
 def download(index):
  start=index*chunk_size;end=min(size,start+chunk_size)-1
  for attempt in range(6):
   try:
    session=requests.Session();session.trust_env=False
    with session.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(15,75)) as r:
     r.raise_for_status()
     if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Incorrect HTTP range response')
     position=start
     for data in r.iter_content(1024**2):
      if not data:continue
      if position+len(data)>end+1:raise ValueError('Oversized HTTP range')
      view=memoryview(data)
      while view:
       written=os.pwrite(fd,view,position)
       if written<=0:raise OSError('Range write failed')
       view=view[written:];position+=written
     if position!=end+1:raise ValueError('Truncated HTTP range')
    session.close();return index
   except Exception:
    if attempt==5:raise
    time.sleep(min(2**attempt,15))
 last=time.monotonic()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
   futures=[pool.submit(download,i) for i in range(count) if i not in done]
   for f in concurrent.futures.as_completed(futures):
    done.add(f.result());state['completed']=sorted(done);save(progress,state)
    if time.monotonic()-last>25:
     print('RANGE_DOWNLOAD',path.name,len(done),'/',count,flush=True);last=time.monotonic()
  os.fsync(fd)
 finally:os.close(fd)
 if len(done)!=count:raise ValueError('Unfinished ranges')
 actual=sha(partial)
 if actual!=expected:raise ValueError('Full archive SHA256 mismatch; keep partial for inspection')
 partial.replace(path);save(path.with_name(path.name+'.download.json'),{'url':url,'size':size,'sha256':actual,'verified':True,'download_method':'bounded_ranges','range_workers':workers});progress.unlink(missing_ok=True)
 return path
