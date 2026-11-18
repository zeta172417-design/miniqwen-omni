#!/usr/bin/env python3
"""Share byte-identical media storage while preserving every source path."""
import collections,concurrent.futures,hashlib,json,os,sys
from pathlib import Path
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'data_pipeline'))
from sft_data_utils import sha,save
M=R/'dataset/raw/manifests/llava15';RAW=R/'dataset/raw/external/llava15/images'
def main():
 os.nice(10);groups=collections.defaultdict(dict)
 for p in (M/'media-audits').glob('*.parquet'):
  for row in pq.read_table(p,columns=['path','sha256','bytes']).to_pylist():groups[row['sha256']][row['path']]=row['bytes']
 tasks=[]
 for digest,paths in groups.items():
  ordered=sorted(paths)
  for target in ordered[1:]:
   if (R/target).is_relative_to(RAW):tasks.append((digest,R/ordered[0],R/target,paths[target]))
 def link(task):
  digest,canonical,target,size=task
  a,b=canonical.stat(),target.stat()
  if (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino):return {'target':str(target.relative_to(R)),'already_shared':True,'bytes':0}
  if canonical.is_symlink() or target.is_symlink():raise ValueError('Unexpected media symlink')
  if a.st_size!=size or b.st_size!=size or sha(canonical)!=digest or sha(target)!=digest:raise ValueError('Media changed since acceptance')
  tmp=target.with_name(target.name+'.dedup-link.partial')
  try:
   os.link(canonical,tmp);os.replace(tmp,target)
  finally:
   if tmp.exists():tmp.unlink()
  assert canonical.stat().st_ino==target.stat().st_ino
  return {'target':str(target.relative_to(R)),'canonical':str(canonical.relative_to(R)),'bytes':size,'already_shared':False}
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for offset in range(0,len(tasks),256):
   records.extend(pool.map(link,tasks[offset:offset+256]))
   if offset%2048==0:print('DEDUP_PROGRESS',len(records),'/',len(tasks),flush=True)
 report={'status':'COMPLETE','unique_byte_groups':len(groups),'duplicate_paths_considered':len(tasks),'new_hardlinks':sum(not x['already_shared'] for x in records),'logical_bytes_deduplicated':sum(x['bytes'] for x in records),'policy':'Verified both files against their accepted SHA256 before atomically replacing duplicate path with a hard link. Every source path, byte content and SFT row remains available.','files':records}
 save(M/'storage-deduplication.json',report);print('MEDIA_STORAGE_DEDUP_COMPLETE',report['new_hardlinks'],'GiB',report['logical_bytes_deduplicated']/2**30,flush=True)
if __name__=='__main__':main()
