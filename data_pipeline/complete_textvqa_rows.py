#!/usr/bin/env python3
import hashlib,json,concurrent.futures,requests,time
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'dataset/raw/external/textvqa'; OUT=ROOT/'dataset/sft/external/textvqa'
from sft_common import SCHEMA
def split(s): return 'dev' if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%10==0 else 'train'
def main():
    rows=[]
    for off in range(0,10000,100):
        u=f'https://datasets-server.huggingface.co/rows?dataset=lmms-lab-encoder/textvqa&config=default&split=train&offset={off}&length=100'
        for attempt in range(8):
            try:
                x=requests.get(u,timeout=120); x.raise_for_status(); rows.extend(z['row'] for z in x.json()['rows']); print('rows',len(rows),flush=True); break
            except Exception:
                if attempt==7: raise
                time.sleep(2+attempt*2)
    def get(r):
        target=RAW/'images'/f"{r['image_id']}.jpg"; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or target.stat().st_size<1000:
            b=requests.get(r['image']['src'],timeout=60); b.raise_for_status(); target.write_bytes(b.content)
        return r,target
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex: loaded=list(ex.map(get,rows))
    out={'train':[],'dev':[]}
    for i,(r,target) in enumerate(loaded):
        ans=[str(x).strip() for x in r.get('answers',[]) if str(x).strip()]
        if not ans: continue
        a=max(set(ans),key=ans.count); sid=f"textvqa:{r['image_id']}:{i}"; sp=split(sid)
        msg=[{'role':'user','content':'<image>\n'+r['question']+'\nAnswer using a short phrase only.'},{'role':'assistant','content':a}]
        out[sp].append({'sample_id':sid,'source':'textvqa','source_row':i,'split':sp,'task_type':'image_instruction','conversations':json.dumps(msg,ensure_ascii=False),'question_audios':[],'question_audio_paths':[],'answer_audios':[],'image_paths':[str(target.relative_to(ROOT))],'ref_audios':[],'spk_emb':[],'codec_frames':0,'duration':None,'sample_rate':None,'language':'en','speaker_id':None,'reference_sample_id':None,'reference_speaker_id':None,'voice_self_fallback':False,'group_id':r['image_id'],'length_tier':'main_2048','required_length':0,'text_length':0,'audio_frames_upper_bound':0,'voice_conditioning':'available_both','supervision_policy':'all_assistant_text','raw_source_parquet':'hf://lmms-lab-encoder/textvqa/train','raw_source_row':i,'metadata_json':json.dumps({'image_id':r['image_id'],'question_id':r['question_id'],'answers':ans,'ocr_tokens':r.get('ocr_tokens',[])},ensure_ascii=False)})
    for sp,rs in out.items(): pq.write_table(pa.Table.from_pylist(rs,schema=SCHEMA),OUT/f'{sp}.parquet',compression='zstd')
    m={'status':'READY','source':'lmms-lab-encoder/textvqa Dataset Server embedded images','selected_annotations':len(rows),'accepted_rows':sum(map(len,out.values())),'train_rows':len(out['train']),'dev_rows':len(out['dev']),'unique_images':len({r['group_id'] for rs in out.values() for r in rs})}
    (OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print(json.dumps(m,ensure_ascii=False))
if __name__=='__main__':main()
