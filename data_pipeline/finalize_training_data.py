#!/usr/bin/env python3
"""Repair prepared training views from immutable raw conditions and publish audits."""
import hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter,defaultdict
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trainer.trainer_utils import load_omni_tokenizer
W=ROOT/'.runtime/data-finalize';W.mkdir(exist_ok=True,parents=True)

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.new');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def setcol(t,n,v,typ=None):
 a=v if isinstance(v,(pa.Array,pa.ChunkedArray)) else pa.array(v,type=typ)
 return t.set_column(t.column_names.index(n),n,a) if n in t.column_names else t.append_column(n,a)
def crop(v,n):
 if len(v)%8:raise ValueError('partial reference frame')
 start=max(0,(len(v)//8-n)//2)*8
 return v[start:start+n*8]
def raw_table(source):
 if source=='old_a2a':
  return pq.read_table(ROOT/'dataset/raw/legacy/sft_a2a.parquet',columns=['spk_emb','ref_audios'])
 return pa.concat_tables([pq.read_table(p,columns=['sample_id','spk_emb','ref_audios']) for p in sorted((ROOT/'dataset/raw/corpora'/source).glob('*.parquet'))])

def main():
 tok=load_omni_tokenizer(ROOT/'model/Qwen3-0.6B')
 assistant=tok.encode('<|im_start|>assistant\n',add_special_tokens=False);think=tok.encode('</think>\n\n',add_special_tokens=False)
 raw={};lookup={}
 for source in ['old_a2a','libritts_r','aishell3']:
  raw[source]=raw_table(source)
  if source!='old_a2a':lookup[source]={x:i for i,x in enumerate(raw[source]['sample_id'].to_pylist())}
  print('RAW_CONDITIONS_LOADED',source,len(raw[source]),flush=True)
 results=[]
 files=sorted((ROOT/'dataset/pretrain').rglob('*.parquet'))+sorted(p for p in (ROOT/'dataset/sft').rglob('*.parquet') if 'external' not in p.parts)
 def process(pair):
  fi,p=pair
  rel=str(p.relative_to(ROOT));receipt=W/'files'/(hashlib.sha256(rel.encode()).hexdigest()+'.json')
  if receipt.exists():return json.loads(receipt.read_text())
  kind='sft' if 'sft' in p.parts else 'pretrain';pf=pq.ParquetFile(p)
  tmp=p.with_name(p.name+'.rebuild');writer=None;counts=Counter();policies=Counter();modes=Counter();maxref=0;excluded=[];refrows=0;spkrows=0;histmulti=0
  for b in pf.iter_batches(batch_size=256,use_threads=False):
   t=pa.Table.from_batches([b]);sources=t['source'].to_pylist();source=sources[0]
   oldr=pc.fill_null(pc.list_value_length(t['ref_audios']),0).to_pylist();olds=pc.fill_null(pc.list_value_length(t['spk_emb']),0).to_pylist()
   if kind=='sft' and source in raw:
    if source=='old_a2a':idx=t['raw_source_row']
    else:idx=pa.array([lookup[source][sid.removeprefix('sft:')] for sid in t['sample_id'].to_pylist()])
    restored=raw[source].take(idx)
    t=setcol(t,'spk_emb',restored['spk_emb'].cast(t.schema.field('spk_emb').type))
    t=setcol(t,'ref_audios',restored['ref_audios'].cast(t.schema.field('ref_audios').type))
   refs=t['ref_audios'].to_pylist();speakers=pc.fill_null(pc.list_value_length(t['spk_emb']),0).to_pylist()
   limit=150 if source=='old_a2a' else 50
   fallback=t['voice_self_fallback'].to_pylist() if 'voice_self_fallback' in t.column_names else [False]*len(t)
   refs=[[] if f else crop(r or [],limit) for r,f in zip(refs,fallback)]
   t=setcol(t,'ref_audios',refs,t.schema.field('ref_audios').type)
   conv=[json.loads(x) for x in t['conversations'].to_pylist()]
   frames=t['codec_frames'].to_pylist()
   policies_b=['all_text_last_audio' if f or (s not in {'old_i2t','llava','m4','llava15_single','coig','ultrachat'} and (s in {'old_a2a_input','librispeech','aishell1'})) else 'all_assistant_text' for s,f in zip(sources,frames)]
   t=setcol(t,'supervision_policy',policies_b,pa.string())
   if kind=='sft':
    split=t['split'].to_pylist();t=setcol(t,'voice_conditioning',['task_mixed' if x=='train' else 'available_both' for x in split],pa.string())
    required=t['required_length'].to_pylist()
    for i,(f,r,s,orr,oss) in enumerate(zip(frames,refs,speakers,oldr,olds)):
     if f:required[i]+= (160 if r or s else 0)-(80 if orr or oss else 0)
    # This pass is idempotent via per-file receipts; old files used an 80-position prefix.
    t=setcol(t,'required_length',required,t.schema.field('required_length').type)
    cap=3072 if 'long_text_3072' in p.parts else 2048
   elif source in {'libritts_r','aishell3'}:
    prompts=[tok.apply_chat_template(c,tokenize=False,add_generation_prompt=False) for c in conv]
    ids=tok(prompts,add_special_tokens=False).input_ids;required=[]
    for seq,f,r,s in zip(ids,frames,refs,speakers):
     starts=[i+len(assistant) for i in range(len(seq)-len(assistant)+1) if seq[i:i+len(assistant)]==assistant];start=starts[-1]
     for j in range(start,len(seq)-len(think)+1):
      if seq[j:j+len(think)]==think:start=j+len(think);break
     required.append(max(len(seq),start+f+2)+(160 if r or s else 0))
    cap=768
   else:required=[0]*len(t);cap=768
   keep=[n<=cap for n in required]
   for i,ok in enumerate(keep):
    if not ok:excluded.append({'sample_id':t['sample_id'][i].as_py(),'required_length':required[i],'reason':'complete_sequence_over_budget'})
   counts['input_rows']+=len(t)
   for i,ok in enumerate(keep):
    if ok:
     counts['rows']+=1;policies[policies_b[i]]+=1;maxref=max(maxref,len(refs[i])//8);refrows+=bool(refs[i]);spkrows+=bool(speakers[i]);histmulti+=sum(m['role']=='assistant' for m in conv[i])>1
   t=t.filter(pa.array(keep))
   if writer is None:writer=pq.ParquetWriter(tmp,t.schema,compression='zstd',compression_level=3)
   writer.write_table(t,row_group_size=128)
  if writer:writer.close()
  pf.close();tmp.replace(p)
  record={'path':rel,'kind':kind,'source':source,**counts,'excluded':excluded,'policies':dict(policies),'max_reference_frames':maxref,'reference_rows':refrows,'speaker_rows':spkrows,'multi_assistant_rows':histmulti,'sha256':sha(p),'bytes':p.stat().st_size}
  save(receipt,record)
  print('FINALIZED',fi,'/',len(files),rel,counts['rows'],flush=True)
  return record
 with ThreadPoolExecutor(max_workers=4) as pool:
  for value in pool.map(process,enumerate(files,1)):
   results.append(value)
 save(W/'result.json',{'status':'FILES_COMPLETE','files':results})
 print('FILES_COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
