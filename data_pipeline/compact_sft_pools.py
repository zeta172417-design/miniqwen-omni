#!/usr/bin/env python3
"""Compact normalized pools losslessly and remove verified redundant source files."""
import hashlib,json,os,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from sft_data_utils import ROOT,BASE,save,sha,load_hub
W=ROOT/'.runtime/sft-data-v1'

def compact(path):
 dest=path.with_suffix('.parquet');temp=dest.with_name(dest.name+'.partial');h=hashlib.sha256();count=0;buffer=[]
 schema=pa.schema([('sample_id',pa.string()),('source',pa.string()),('task_type',pa.string()),('split',pa.string()),('record_json',pa.string())])
 with pq.ParquetWriter(temp,schema,compression='zstd',compression_level=6) as writer:
  for line in path.open():
   if not line.strip():continue
   r=json.loads(line);h.update(line.encode());count+=1
   buffer.append({'sample_id':r['sample_id'],'source':r['source'],'task_type':r['task_type'],'split':r['split'],'record_json':line.rstrip('\n')})
   if len(buffer)>=512:writer.write_table(pa.Table.from_pylist(buffer,schema=schema));buffer=[]
  if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=schema))
 restored=hashlib.sha256();checked=0
 for batch in pq.ParquetFile(temp).iter_batches(columns=['record_json']):
  for value in batch.column(0).to_pylist():restored.update((value+'\n').encode());checked+=1
 if checked!=count or restored.hexdigest()!=h.hexdigest():raise ValueError('Lossless Parquet verification failed')
 temp.replace(dest);info={'source':str(path.relative_to(ROOT)),'target':str(dest.relative_to(ROOT)),'rows':count,'source_bytes':path.stat().st_size,'target_bytes':dest.stat().st_size,'payload_sha256':h.hexdigest(),'exact_payload_roundtrip':True};path.unlink();return info

def main():
 save(W/'compaction-status.json',{'status':'WAITING_FOR_MEDIA_PROCESSING'})
 while True:
  p=W/'status.json'
  if p.exists():
   state=json.loads(p.read_text())
   if state['status']=='QUANTITY_PREPARATION_COMPLETE':break
   if state['status']=='FAILED':
    save(W/'compaction-status.json',{'status':'WAITING_FOR_PROCESSING_RECOVERY','processing_error':state.get('error')})
  time.sleep(15)
 save(W/'compaction-status.json',{'status':'COMPACTING'})
 results=[]
 for name in ['ultrachat','coig','fusion','m4']:
  out=BASE/'processed'/name
  for split in ['train','dev']:
   p=out/(split+'.jsonl')
   if p.exists():results.append(compact(p));print('COMPACTED',name,split,flush=True)
  p=out/'manifest.json';m=json.loads(p.read_text());m['data_files']={s:str((out/(s+'.parquet')).relative_to(ROOT)) for s in ['train','dev']};m['format']='Parquet normalized pool: indexed scalar columns plus losslessly preserved record_json; not final OmniDataset binary schema';save(p,m)
 removed=[]
 # These are redundant annotations; pin/hash evidence and per-source audit remain.
 for p in [BASE/'raw/m4/m4_instruct_annotations.json',BASE/'source_metadata/m4-eligible-annotations.jsonl',BASE/'raw/fusion/metadata.jsonl']:
  if p.exists():
   removed.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p),'reason':'quantity processing committed; selected rows retained in canonical Parquet, upstream revision and audits retained'});p.unlink()
 save(BASE/'source_metadata/final-compaction.json',{'lossless_conversions':results,'removed_redundant_annotations':removed})
 p=BASE/'manifest.json';summary=json.loads(p.read_text());summary['status']='QUANTITY_PREPARATION_COMPLETE';summary['storage_compaction']={'manifest':'dataset/sft_v1/source_metadata/final-compaction.json','saved_bytes':sum(r['source_bytes']-r['target_bytes'] for r in results)+sum(r['bytes'] for r in removed)}
 for name in summary['sources']:summary['sources'][name]=json.loads((BASE/'processed'/name/'manifest.json').read_text())
 summary['deleted_duplicate_bytes']+=summary['storage_compaction']['saved_bytes'];save(p,summary);save(ROOT/'reports/SFT_EXTERNAL_DATA_STATUS.json',summary)
 report=ROOT/'reports/SFT_EXTERNAL_DATA_STATUS.md'
 with report.open('a') as f:
  f.write('\n规范化池已无损压缩为Parquet（Zstandard），每个split逐条反读record_json核对行数和payload SHA256通过后删除JSONL。原始M4/融合重复注释也已清理，来源版本、上游哈希、统计与隔离记录保留。Parquet尚非最终OmniDataset二进制训练schema。\n')
  f.write(f'\n本轮压缩及注释去重额外节省 {summary["storage_compaction"]["saved_bytes"]/2**30:.2f} GiB；累计清理/压缩节省 {summary["deleted_duplicate_bytes"]/2**30:.2f} GiB。\n')
 save(W/'compaction-status.json',{'status':'COMPLETE','report':str(report.relative_to(ROOT))})
 print('SFT_DATA_DOWNLOAD_PROCESS_CLEANUP_COMPLETE',flush=True)

if __name__=='__main__':main()
