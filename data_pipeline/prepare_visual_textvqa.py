#!/usr/bin/env python3
"""Prepare a compact TextVQA train subset in the existing SFT Parquet schema."""
from __future__ import annotations
import concurrent.futures, hashlib, json, os, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset/raw/external/textvqa'
OUT = ROOT / 'dataset/sft/external/textvqa'
SCHEMA = pa.schema([
 ('sample_id',pa.string()),('source',pa.string()),('source_row',pa.int64()),('split',pa.string()),('task_type',pa.string()),('conversations',pa.string()),
 ('question_audios',pa.list_(pa.binary())),('question_audio_paths',pa.list_(pa.string())),('answer_audios',pa.list_(pa.list_(pa.int64()))),('image_paths',pa.list_(pa.string())),('ref_audios',pa.list_(pa.int64())),('spk_emb',pa.list_(pa.float32())),('codec_frames',pa.int32()),('duration',pa.float64()),('sample_rate',pa.int32()),('language',pa.string()),('speaker_id',pa.string()),('reference_sample_id',pa.string()),('reference_speaker_id',pa.string()),('voice_self_fallback',pa.bool_()),('group_id',pa.string()),('length_tier',pa.string()),('required_length',pa.int32()),('text_length',pa.int32()),('audio_frames_upper_bound',pa.int32()),('voice_conditioning',pa.string()),('supervision_policy',pa.string()),('raw_source_parquet',pa.string()),('raw_source_row',pa.int64()),('metadata_json',pa.string())])

def download(item):
    image_id, urls = item
    target = RAW / 'images' / f'{image_id}.jpg'; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1000: return image_id, True, None
    try:
        errors=[]
        for url in urls:
            try:
                req = Request(url, headers={'User-Agent':'MiniQwen-Omni dataset preparation'})
                with urlopen(req, timeout=20) as src, target.open('wb') as dst: dst.write(src.read())
                if target.stat().st_size > 1000: return image_id, True, None
            except Exception as exc: errors.append(repr(exc))
        return image_id, False, '; '.join(errors[-2:])
    except Exception as exc: return image_id, False, repr(exc)

def main():
    started = time.perf_counter()
    data = json.loads((RAW/'TextVQA_0.5.1_train.json').read_text())['data']
    selected = data[:10000]
    manifest = []
    for i, item in enumerate(selected):
        answers = [str(x).strip() for x in item.get('answers',[]) if str(x).strip()]
        if not answers: continue
        answer = max(set(answers), key=answers.count)
        split = 'dev' if int(hashlib.sha256(item['image_id'].encode()).hexdigest()[:8],16)%10 == 0 else 'train'
        urls=[u for u in [item.get('flickr_300k_url'), item.get('flickr_original_url')] if u]
        manifest.append((item['image_id'], urls, item, answer, split, i))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(download, [(x[0],x[1]) for x in manifest]))
    failed = {x[0] for x in results if not x[1]}
    rows={'train':[],'dev':[]}
    for image_id, _, item, answer, split, i in manifest:
        if image_id in failed: continue
        answers = [str(x).strip() for x in item.get('answers',[]) if str(x).strip()]
        messages=[{'role':'user','content':'<image>\n'+item['question']+'\nAnswer using a short phrase only.'},{'role':'assistant','content':answer}]
        rows[split].append({'sample_id':f'textvqa:{image_id}:{i}','source':'textvqa','source_row':i,'split':split,'task_type':'image_instruction','conversations':json.dumps(messages,ensure_ascii=False),'question_audios':[],'question_audio_paths':[],'answer_audios':[],'image_paths':[str((RAW/'images'/f'{image_id}.jpg').relative_to(ROOT))],'ref_audios':[],'spk_emb':[],'codec_frames':0,'duration':None,'sample_rate':None,'language':'en','speaker_id':None,'reference_sample_id':None,'reference_speaker_id':None,'voice_self_fallback':False,'group_id':image_id,'length_tier':'main_2048','required_length':0,'text_length':0,'audio_frames_upper_bound':0,'voice_conditioning':'available_both','supervision_policy':'all_assistant_text','raw_source_parquet':'dataset/raw/external/textvqa/TextVQA_0.5.1_train.json','raw_source_row':i,'metadata_json':json.dumps({'image_id':image_id,'question_id':item.get('question_id'),'answers':answers},ensure_ascii=False)})
    OUT.mkdir(parents=True,exist_ok=True)
    for split, rs in rows.items(): pq.write_table(pa.Table.from_pylist(rs,schema=SCHEMA),OUT/f'{split}.parquet',compression='zstd')
    (RAW/'selected_manifest.jsonl').write_text(''.join(json.dumps({'sample_id':r['sample_id'],'split':r['split'],'image':r['image_paths'][0]},ensure_ascii=False)+'\n' for rs in rows.values() for r in rs))
    elapsed = time.perf_counter() - started
    print(json.dumps({'selected':len(manifest),'accepted':sum(map(len,rows.values())),'train':len(rows['train']),'dev':len(rows['dev']),'failed_images':len(failed),'elapsed_seconds':round(elapsed,2),'download_workers':16},ensure_ascii=False))
if __name__=='__main__': main()
