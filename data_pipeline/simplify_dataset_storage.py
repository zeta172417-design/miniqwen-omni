"""Keep current data/assets; delete superseded dataset version directories."""
import collections,json,os,shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def move(src,dst):
 src,dst=Path(src),Path(dst)
 if src.exists() and not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);os.replace(src,dst)

def simplify_external(R,M,save):
 D=R/'dataset';old=D/'raw/external/sft_v1';meta=D/'raw/manifests/external'
 if old.exists():
  move(old/'media/m4',D/'raw/external/m4');move(old/'raw/fusion',D/'raw/external/fusion')
  move(old/'source_metadata',meta)
  for name in ['m4','coig','ultrachat']:move(old/'raw'/name,meta/'source-records'/name)
  move(old/'manifest.json',meta/'previous-pools-manifest.json')
  move(old/'processing-audits',meta/'audits')
  unexpected=[p for p in old.rglob('*') if p.is_file() and p.suffix.lower() in {'.parquet','.wav','.flac','.jpg','.jpeg','.png','.arrow','.zip','.z01'}]
  if unexpected:raise ValueError('Unmoved external assets:'+str(unexpected[:3]))
  shutil.rmtree(old)
 for folder in (D/'sft/external').iterdir():
  if not folder.is_dir():continue
  for p in list(folder.iterdir()):
   if p.is_file() and p.suffix!='.parquet' and p.name!='manifest.json':move(p,meta/'audits'/folder.name/p.name)
  path=folder/'manifest.json';spec=json.loads(path.read_text());spec.pop('media_path_remap',None)
  if folder.name in ['fusion','m4']:spec['raw_media_root']=f'dataset/raw/external/{folder.name}';spec['legacy_payload_media_paths_unmodified']=True
  save(path,spec)
 for name in ['sft-path-remap.json','path-migration.json']:(M/name).unlink(missing_ok=True)
 save(D/'sft/README.json',{'status':'EXISTING_PARQUET_POOLS_ONLY_NEW_SFT_NOT_PREPARED','external_pools':'dataset/sft/external','raw_media':'dataset/raw/external/{fusion,m4}','reserved_supplements':'dataset/raw/manifests/pretrain/partitions','note':'Old payload media paths are intentionally not repaired. Rebuild the final mixed SFT schema later; current Parquet files themselves remain byte-for-byte unchanged.'})

def preserve_required(R,M,save,digest):
 D=R/'dataset';a=D/'raw/_migration_archive';audit=D/'raw/audit';records=[]
 old=a/'pretrain_audit'
 move(old/'evaluation',D/'raw/evaluation_assets/official_sources')
 move(a/'pretrain/media/libritts_r/libritts_r_failed_speech_restoration_examples',audit/'metadata/libritts_r_official_failure_lists')
 originals=old/'originals'
 if originals.exists():
  for base,dirs,files in os.walk(originals,followlinks=False):
   for name in files:
    src=Path(base)/name
    if src.is_symlink():raise ValueError('Protected audit contains symlink; inspect before delete')
    source='aishell1' if name.startswith('BAC') else 'libritts_r';dst=audit/'originals'/source/name;dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
     if digest(src)!=digest(dst):dst=dst.with_name(digest(src)[:16]+'-'+name)
     else:src.unlink();continue
    os.replace(src,dst);records.append({'path':str(dst.relative_to(R)),'bytes':dst.stat().st_size})
 move(old/'manifest.json',audit/'previous-preservation-manifest.json')
 # Full dev waveforms and current reference crops remain useful source assets.
 move(a/'pretrain/media/libritts_r/LibriTTS_R/dev-clean',D/'raw/corpora/libritts_r/dev_waveforms')
 move(a/'pretrain_v2/voice/references',D/'raw/corpora/voice_references')
 cache=a/'pretrain_v2/voice/reference_cache.json'
 if cache.exists():
  data=json.loads(cache.read_text());index={}
  for sid,c in data.get('conditions',{}).items():
   index[sid]={k:v for k,v in c.items() if k not in ['spk_emb','ref_audios']}
   for k in ['path','audio_path','file']:
    if isinstance(index[sid].get(k),str):index[sid][k]=Path(index[sid][k]).name
  save(D/'raw/corpora/voice_references/index.json',{'references':index,'note':'Current reference crop files; numeric speaker/codec conditions already embedded in corpus Parquet.'})
 # Retain the explicitly protected official-failure encodings, not duplicate full codec trees.
 flagpaths=[a/'pretrain/processed/libritts_r/official_restoration_flags.json',a/'pretrain_v2/processed/libritts_r/official_clean360_flags.json']
 flags=set()
 for p in flagpaths:
  retained=audit/'metadata'/p.name
  if p.exists():move(p,retained)
  if retained.exists():
   ids=json.loads(retained.read_text())['sample_ids']
   if isinstance(ids,dict):
    for split_ids in ids.values():flags.update(split_ids)
   else:flags.update(ids)
 expected=json.loads((M/'historical/pretrain_v2/manifest.json').read_text())['sources']['libritts_r']['excluded_counts']
 output=audit/'libritts_r_flagged_full_codec.parquet';found=set();count=0
 if not output.exists() or pq.ParquetFile(output).metadata.num_rows!=sum(expected.values()):
  temporary=output.with_suffix('.parquet.partial')
  writer=None
  try:
   for root in [a/'pretrain/processed/libritts_r/codec',a/'pretrain_clean360/processed/libritts_r/codec']:
    for path in root.rglob('*.parquet'):
     for batch in pq.ParquetFile(path).iter_batches(batch_size=512,use_threads=False):
      table=pa.Table.from_batches([batch]);selected=table.filter(pc.is_in(table['sample_id'],value_set=pa.array(sorted(flags))))
      if not len(selected):continue
      ids=selected['sample_id'].to_pylist()
      if found.intersection(ids):raise ValueError('Duplicate protected codec IDs')
      found.update(ids);count+=len(ids)
      if writer is None:output.parent.mkdir(parents=True,exist_ok=True);writer=pq.ParquetWriter(temporary,selected.schema,compression='zstd')
      writer.write_table(selected)
  finally:
   if writer is not None:writer.close()
  if count!=sum(expected.values()):raise ValueError(('Protected codec count mismatch',count,expected))
  os.replace(temporary,output)
 else:count=pq.ParquetFile(output).metadata.num_rows
 if count!=sum(expected.values()):raise ValueError('Protected codec row count mismatch')
 records=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size} for p in (audit/'originals').rglob('*') if p.is_file()]
 save(audit/'manifest.json',{'original_files':records,'protected_full_codec_rows':count,'expected_excluded_counts':expected,'quality_flags':len(flags),'note':'Retained required exception originals/encodings and official heldout data; obsolete working versions removed.'})
 return count

def cleanup_simple(R,M,W,save,digest):
 D=R/'dataset';protected=[R/'model',R/'out',D/'raw/corpora',D/'raw/legacy/sft_a2a.parquet',D/'raw/legacy/sft_i2t.parquet',D/'raw/legacy/sft_t2a.parquet',D/'raw/audit',D/'raw/evaluation_assets',D/'raw/external',D/'sft']
 targets=[D/'raw/_migration_archive',M/'historical',D/'raw/legacy/sft_a2a_mini.parquet',D/'raw/legacy/sft_t2a_mini.parquet',D/'__pycache__']
 targets += [R/p for p in ['.runtime/legacy-pretrain-v1/dataset-cache','.runtime/legacy-pretrain-v1/smoke-cache','.runtime/legacy-pretrain-speed-v2/cache','.runtime/pretrain/validation-cache']]
 records=[]
 for p in targets:
  if not p.exists():continue
  if any(p.resolve()==x.resolve() or p.resolve().is_relative_to(x.resolve()) for x in protected):raise ValueError('Protected cleanup target')
  fs=[p] if p.is_file() else [x for x in p.rglob('*') if x.is_file() and not x.is_symlink()]
  row={'path':str(p.relative_to(R)),'bytes':sum(f.stat().st_size for f in fs),'files':len(fs),'reason':'Superseded working version, redundant derived data or cache. Required sources and protected assets were moved out first.'}
  if p.suffix=='.parquet':row.update(rows=pq.ParquetFile(p).metadata.num_rows,sha256=digest(p))
  records.append(row)
 save(W/'cleanup-plan.json',{'targets':records,'protected':[str(p.relative_to(R)) for p in protected]})
 for p in targets:
  if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
  elif p.exists() or p.is_symlink():p.unlink()
 save(M/'cleanup.json',{'status':'COMPLETE','targets':records,'logical_bytes':sum(x['bytes'] for x in records),'policy':'Delete superseded directories; do not maintain deprecated-path compatibility.'})
 return sum(x['bytes'] for x in records)
