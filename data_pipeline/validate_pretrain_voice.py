#!/usr/bin/env python3
"""Full metadata/code validation of materialized voice-conditioned views."""
import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=ROOT/os.environ.get('MINIQWEN_PRETRAIN_DATA','dataset/pretrain_v2')
    manifest=json.loads((base/'voice/manifest.json').read_text());summary={};fallback=[]
    for name,src in manifest['sources'].items():
        splits={};allrows={}
        for split,paths in src['data_files'].items():
            rows=[]
            for path in paths:
                pf=pq.ParquetFile(ROOT/path)
                cols=[c for c in pf.schema_arrow.names if c not in {'audios','images','conversations','answer_audios'}]
                for batch in pf.iter_batches(batch_size=512,columns=cols):rows.extend(batch.to_pylist())
            splits[split]=rows
            for row in rows:allrows[row['sample_id']]=row
        speakers={s:{r['speaker_id'] for r in rows} for s,rows in splits.items()}
        assert not speakers['train']&speakers['dev']
        for split,rows in splits.items():
            assert len(rows)==src['counts'][split]
            for r in rows:
                emb=np.asarray(r['spk_emb']);assert emb.shape==(192,) and np.isfinite(emb).all()
                codes=r['ref_audios'];assert len(codes)%8==0 and len(codes)//8+3<=160
                assert all(0<=c<2048 for c in codes)
                assert r['reference_speaker_id']==r['speaker_id'] and r['reference_split']==split
                if r['voice_self_fallback']:
                    assert not codes;fallback.append(r['sample_id'])
                else:
                    ref=allrows[r['reference_sample_id']]
                    assert ref['sample_id']!=r['sample_id'] and ref['text_target']!=r['text_target']
                    assert len(codes)>0
            summary[name+'/'+split]={'rows':len(rows),'speakers':len(speakers[split])}
    assert set(fallback)==set(manifest['fallback_sample_ids'])
    result={'status':'PASS','views':summary,'speaker_only_self_fallbacks':fallback}
    log=ROOT/os.environ.get('MINIQWEN_PRETRAIN_LOG','.runtime/pretrain-v2' if base.name=='pretrain_v2' else '.runtime/pretrain');log.mkdir(parents=True,exist_ok=True)
    out=log/'voice-validation.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
