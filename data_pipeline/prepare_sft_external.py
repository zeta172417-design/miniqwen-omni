#!/usr/bin/env python3
"""Normalize selected external SFT pools and report actual usable quantities.

No speech synthesis, codec encoding or training; media are referenced by path.
"""
import argparse,collections,hashlib,io,json,os,sys,unicodedata
from pathlib import Path
from sft_data_utils import ROOT,BASE,save,json_array

def key(s):return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
def group_split(s):return 'dev' if int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%100==0 else 'train'
def normalize_messages(items):
 out=[]
 for m in items:
  role={'human':'user','gpt':'assistant'}.get(m.get('role',m.get('from')),m.get('role',m.get('from')))
  text=m.get('content',m.get('value',''))
  if role not in ['system','user','assistant'] or not isinstance(text,str) or not text.strip():raise ValueError('invalid_role_or_empty_text')
  if out and out[-1]['role']==role:raise ValueError('consecutive_identical_roles')
  out.append({'role':role,'content':text})
 if not out or out[-1]['role']!='assistant' or next((x['role'] for x in out if x['role']!='system'),None)!='user':raise ValueError('invalid_conversation_endpoints')
 return out

def blocked_texts():
 out=set();p=Path('/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/omni-core-v2.3-standard/manifest.jsonl')
 if p.exists():
  for line in p.open():
   r=json.loads(line)
   for text in [r.get('prompt'),r.get('metadata',{}).get('transcript_prompt')]:
    if isinstance(text,str) and len(key(text))>=12:out.add(key(text))
 return out

def text_source(name):
 import pyarrow.parquet as pq
 out=BASE/'processed'/name;out.mkdir(parents=True,exist_ok=True)
 revision=json.loads((BASE/'source_metadata'/f'{name}-hub.json').read_text())['sha']
 blocked=blocked_texts();counts=collections.Counter();reasons=collections.Counter();seen=set();domains=collections.Counter();chars=[]
 if name=='ultrachat':
  rows=(r for p in sorted((BASE/'raw/ultrachat/data').glob('train_sft-*.parquet')) for b in pq.ParquetFile(p).iter_batches(batch_size=256) for r in b.to_pylist())
 else:rows=(json.loads(line) for line in (BASE/'raw/coig/COIG-CQIA-full.jsonl').open() if line.strip())
 handles={s:(out/(s+'.jsonl')).open('w') for s in ['train','dev']};reject=(out/'excluded.jsonl').open('w')
 try:
  for index,r in enumerate(rows):
   counts['raw_rows']+=1
   try:
    if name=='ultrachat':messages=normalize_messages(r['messages']);sid=str(r.get('prompt_id',index))
    else:
     prompt=r.get('instruction','');extra=r.get('input','');answer=r.get('output','')
     if not all(isinstance(x,str) for x in [prompt,extra,answer]):raise ValueError('non_string_field')
     messages=normalize_messages([{'role':'user','content':prompt+('\n'+extra if extra else '')},{'role':'assistant','content':answer}]);sid=str(index)
    userkeys=[key(x['content']) for x in messages if x['role']=='user']
    if any(k in blocked for k in userkeys):raise ValueError('exact_normalized_benchmark_question')
    digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if digest in seen:raise ValueError('exact_conversation_duplicate')
    seen.add(digest);split=group_split(userkeys[0]);counts[split]+=1
    record={'sample_id':name+':'+sid,'source':name,'source_revision':revision,'task_type':'text','split':split,'group_id':userkeys[0],'conversations':messages,'metadata':{k:v for k,v in r.items() if k not in ['messages','output']}}
    revision=record['source_revision']
    handles[split].write(json.dumps(record,ensure_ascii=False)+'\n');chars.append(sum(len(m['content']) for m in messages))
   except (ValueError,KeyError,TypeError) as e:
    reasons[str(e)]+=1;reject.write(json.dumps({'row':index,'reason':str(e),'original':r},ensure_ascii=False)+'\n')
 finally:
  for f in handles.values():f.close()
  reject.close()
 chars.sort();report={'status':'normalized_text_ready_length_budget_not_applied','source':name,'counts':dict(counts),'excluded':dict(reasons),'chars':{'max':max(chars,default=0),'p50':chars[len(chars)//2] if chars else 0,'p95':chars[int(len(chars)*.95)] if chars else 0},'policy':'Exact conversation dedup; split by normalized first user question; original content preserved. Not a full semantic near-duplicate audit. Token length selection deferred.'}
 save(out/'manifest.json',report);print('PROCESSED',name,json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--text',choices=['ultrachat','coig'],required=True);args=p.parse_args();text_source(args.text)
