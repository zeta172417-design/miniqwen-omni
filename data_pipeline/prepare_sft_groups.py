#!/usr/bin/env python3
"""Join cross-source image groups; preserve existing dev assignments conservatively."""
import collections,hashlib,json,os,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sft_common import *

def external_rows(name):
 for split in ('train','dev'):
  p=OUT/'external'/name/(split+'.parquet')
  for batch in pq.ParquetFile(p).iter_batches(batch_size=256,columns=['record_json'],use_threads=False):
   for item in batch.column(0).to_pylist():yield split,json.loads(item)
class DSU:
 def __init__(self):self.parent={}
 def find(self,x):
  p=self.parent.setdefault(x,x)
  while p!=self.parent[p]:p=self.parent[p]
  while x!=p:n=self.parent[x];self.parent[x]=p;x=n
  return p
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)
def main():
 os.nice(15);META.mkdir(parents=True,exist_ok=True);W.mkdir(parents=True,exist_ok=True);save(W/'groups-status.json',{'status':'BUILDING_GROUPS','pid':os.getpid()})
 media={}
 for p in (OUT/'external/m4/media-manifests').glob('*.json'):
  for path,info in json.loads(p.read_text())['files'].items():media['dataset/raw/external/m4/'+path]=info['sha256']
 for p in (R/'dataset/raw/manifests/llava15/media-audits').glob('*.parquet'):
  for row in pq.read_table(p,columns=['path','sha256']).to_pylist():media[row['path']]=row['sha256']
 # Fusion source had complete decode audit but no per-file digests; hash its selected pictures only.
 fusion_paths=set()
 for _,row in external_rows('fusion'):fusion_paths.update(resolve_media(x) for x in row['image_paths'])
 for i,path in enumerate(sorted(fusion_paths)):
  if path not in media:media[path]=hashlib.sha256((R/path).read_bytes()).hexdigest()
  if i%1000==0:print('FUSION_IMAGE_HASH',i,'/',len(fusion_paths),flush=True)
 graph=DSU();dev_images=set();rows_by_source=collections.Counter()
 for name in ('m4','llava15_single','fusion'):
  for split,row in external_rows(name):
   hashes=[media[resolve_media(p)] for p in row['image_paths']]
   for h in hashes:graph.find(h)
   for h in hashes[1:]:graph.union(hashes[0],h)
   if split=='dev':dev_images.update(hashes)
   rows_by_source[name]+=1
  print('GROUPED',name,rows_by_source[name],flush=True)
 dev_roots={graph.find(h) for h in dev_images}
 # Fixed evaluation manifests are read only; neither test text nor media enter training.
 blocked_hashes=set();blocked_questions=set();blocked_targets=set();protocol=json.loads((R/'.runtime/legacy-pretrain-full-v3-eval/protocol.json').read_text());benchfiles=set()
 for key in ('main_manifest','clone_manifest'):
  for line in Path(protocol[key]).read_text().splitlines():
   row=json.loads(line)
   for k in ('prompt','target_text','reference','ref_text'):
    value=row.get(k)
    if isinstance(value,str) and len(norm(value))>=16:
     if k=='prompt':blocked_questions.add(norm(value))
     else:blocked_targets.add(norm(value))
   for path in [row.get('image'),*(row.get('images') or [])]:
    if path:benchfiles.add(path)
 for path in benchfiles:blocked_hashes.add(hashlib.sha256(Path(path).read_bytes()).hexdigest())
 blocked_roots={graph.find(h) for h in blocked_hashes if h in graph.parent}
 media_rows=[{'path':path,'sha256':h,'group_id':graph.find(h),'split':'dev' if graph.find(h) in dev_roots else 'train','blocked_benchmark_image':graph.find(h) in blocked_roots} for path,h in media.items()]
 pq.write_table(pa.Table.from_pylist(media_rows),META/'media-groups.parquet',compression='zstd',row_group_size=4096)
 text_groups={};exact_text_dev=set()
 for name in ('ultrachat','coig'):
  for split,row in external_rows(name):
   first=next(m['content'] for m in row['conversations'] if m['role']=='user');key=digest_text(norm(first));text_groups[key]='dev' if split=='dev' or text_groups.get(key)=='dev' else 'train'
   if split=='dev':exact_text_dev.add(digest_text(json.dumps(row['conversations'],ensure_ascii=False,sort_keys=True)))
 pq.write_table(pa.Table.from_pylist([{'group_id':k,'split':v} for k,v in text_groups.items()]),META/'text-groups.parquet',compression='zstd')
 save(META/'blocked-evaluation.json',{'image_sha256':sorted(blocked_hashes),'question_normalized':sorted(blocked_questions),'target_normalized':sorted(blocked_targets),'image_group_roots':sorted(blocked_roots),'method':'Exact SHA256 of fixed eval image bytes; normalized exact questions/targets with minimum16 alphanumeric characters. Not perceptual or semantic near-duplicate certification.','text_conversation_dev_digests':sorted(exact_text_dev)})
 # Quantify changes before constructing any training view.
 changed=collections.Counter();components=collections.Counter()
 for name in ('m4','llava15_single','fusion'):
  for split,row in external_rows(name):
   root=graph.find(media[resolve_media(row['image_paths'][0])]);new='dev' if root in dev_roots else 'train';changed[(name,split,new)]+=1;components[root]+=1
 report={'status':'PASS','media_paths':len(media),'unique_image_bytes':len({r['sha256'] for r in media_rows}),'connected_components':len(components),'largest_component_records':max(components.values()),'existing_dev_wins':True,'split_counts':{'/'.join(k):v for k,v in changed.items()},'blocked_fixed_eval_image_components':len(blocked_roots),'policy':'Shared images across all3sources remain in one split. Any existing source dev membership makes the complete connected component dev; no dev example is moved into train.'}
 save(META/'group-audit.json',report);save(W/'groups-status.json',{'status':'COMPLETE'});print('GROUPS_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
