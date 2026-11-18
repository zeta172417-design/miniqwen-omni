#!/usr/bin/env python3
"""Replace Flickr-dependent TextVQA media with the embedded HF mirror."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]; TMP=Path('/tmp/textvqa_hf'); RAW=ROOT/'dataset/raw/external/textvqa'; OUT=ROOT/'dataset/sft/external/textvqa'
from sft_common import SCHEMA
def split(s): return 'dev' if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%10==0 else 'train'
def main(target_count=10000):
    TMP.mkdir(exist_ok=True); (RAW/'images').mkdir(parents=True,exist_ok=True)
    selected=[]; shard=0
    while len(selected)<target_count and shard<20:
        dest=TMP/f'train-{shard:05d}.parquet'
        valid=False
        if dest.exists():
            try: pq.ParquetFile(dest); valid=True
            except Exception: dest.unlink()
        if not valid:
            url=f'https://huggingface.co/datasets/lmms-lab/textvqa/resolve/main/data/train-{shard:05d}-of-00020.parquet?download=true'
            if dest.exists(): dest.unlink()
            subprocess.run(['curl','--http1.1','-L','--fail','--retry','5','--silent','--show-error',url,'-o',str(dest)],check=True)
        for row in pq.ParquetFile(dest).iter_batches(batch_size=256):
            selected.extend(row.to_pylist())
            if len(selected)>=target_count: break
        shard+=1
    selected=selected[:target_count]; rows={'train':[],'dev':[]}; failed=0
    for i,item in enumerate(selected):
        image=item['image'] or {}; data=image.get('bytes')
        image_id=item['image_id']; target=RAW/'images'/f'{image_id}.jpg'
        if not data: failed+=1; continue
        if not target.exists() or target.stat().st_size<1000: target.write_bytes(data)
        answers=[str(x).strip() for x in (item.get('answers') or []) if str(x).strip()]; answer=max(set(answers),key=answers.count) if answers else ''
        if not answer: continue
        sid=f'textvqa:{image_id}:{i}'; sp=split(sid)
        msgs=[{'role':'user','content':'<image>\n'+item['question']+'\nAnswer using a short phrase only.'},{'role':'assistant','content':answer}]
        rows[sp].append({'sample_id':sid,'source':'textvqa','source_row':i,'split':sp,'task_type':'image_instruction','conversations':json.dumps(msgs,ensure_ascii=False),'question_audios':[],'question_audio_paths':[],'answer_audios':[],'image_paths':[str(target.relative_to(ROOT))],'ref_audios':[],'spk_emb':[],'codec_frames':0,'duration':None,'sample_rate':None,'language':'en','speaker_id':None,'reference_sample_id':None,'reference_speaker_id':None,'voice_self_fallback':False,'group_id':image_id,'length_tier':'main_2048','required_length':0,'text_length':0,'audio_frames_upper_bound':0,'voice_conditioning':'available_both','supervision_policy':'all_assistant_text','raw_source_parquet':'hf://lmms-lab/textvqa/data/train-00000-of-00020.parquet','raw_source_row':i,'metadata_json':json.dumps({'image_id':image_id,'question_id':item['question_id'],'answers':answers,'ocr_tokens':item.get('ocr_tokens',[])},ensure_ascii=False)})
    OUT.mkdir(parents=True,exist_ok=True)
    for sp,rs in rows.items(): pq.write_table(pa.Table.from_pylist(rs,schema=SCHEMA),OUT/f'{sp}.parquet',compression='zstd')
    summary={'status':'READY','source':'lmms-lab/textvqa embedded-image mirror','selected_annotations':len(selected),'accepted_rows':sum(map(len,rows.values())),'train_rows':len(rows['train']),'dev_rows':len(rows['dev']),'failed_rows':failed,'shards_read':shard,'unique_images':len({r['group_id'] for rs in rows.values() for r in rs})}
    (OUT/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary,ensure_ascii=False))
    for p in TMP.glob('train-*.parquet'): p.unlink()
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--target-count',type=int,default=10000)
    main(ap.parse_args().target_count)
