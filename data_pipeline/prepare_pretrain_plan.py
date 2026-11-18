#!/usr/bin/env python3
"""Audit every sequence and write a launch plan; never starts training."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pyarrow.parquet as pq
from trainer.trainer_utils import load_omni_tokenizer

def asset_fingerprints():
    result={}
    for directory in ['Qwen3-0.6B','SenseVoiceSmall','siglip2-base-p32-256-ve','mimi','campplus']:
        for path in sorted((ROOT/'model'/directory).rglob('*')):
            if not path.is_file() or path.suffix not in {'.safetensors','.pt','.bin','.json','.yaml','.mvn','.model','.txt'}:continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            result[str(path.relative_to(ROOT))]=h.hexdigest()
    return result

def main():
    base=ROOT/os.environ.get('MINIQWEN_PRETRAIN_DATA','dataset/pretrain');catalog=json.loads((base/'manifest.json').read_text())
    voice_path=base/'voice/manifest.json'
    if not voice_path.exists(): raise RuntimeError('Prepare voice conditions first')
    voice=json.loads(voice_path.read_text());tok=load_omni_tokenizer(ROOT/'model/Qwen3-0.6B')
    assistant=tok.encode('<|im_start|>assistant\n',add_special_tokens=False)
    audit={};phases=[]
    for phase,sources,modules,minimum,batch,lr in [
        ('audio',['librispeech','aishell1'],'audio_proj',512,8,3e-4),
        ('vision',['llava'],'vision_proj',256,8,3e-4),
        ('tts',['libritts_r','aishell3'],'talker',768,8,2e-4)]:
        entries={};maximum=0
        for source in sources:
            spec=voice['sources'][source]['data_files'] if phase=='tts' else catalog['sources'][source]['data_files']
            entries[source]=spec;summary={}
            for split,paths in spec.items():
                stats={'rows':0,'max_required':0,'worst_sample':None}
                for path in paths:
                    pf=pq.ParquetFile(ROOT/path)
                    cols=[c for c in ['sample_id','conversations','duration','codec_frames'] if c in pf.schema_arrow.names]
                    for batch_rows in pf.iter_batches(batch_size=512,columns=cols):
                        rows=batch_rows.to_pylist();prompts=[]
                        for row in rows:
                            messages=json.loads(row['conversations'])
                            if phase=='audio':
                                # Conservative upper bound; actual LFR frames are <= ceil(duration/0.06)+2.
                                frames=math.ceil(row['duration']/0.06)+2
                                messages[0]['content']+='\n<|audio_start|>'+('<|audio_pad|>'*frames)+'<|audio_end|>'
                            if phase=='vision':
                                for turn in messages:
                                    turn['content']=turn['content'].replace('<image>','<|vision_start|>'+('<|image_pad|>'*64)+'<|vision_end|>')
                            prompts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False))
                        tokenized=tok(prompts,add_special_tokens=False).input_ids
                        for row,ids in zip(rows,tokenized):
                            required=len(ids)
                            if phase=='tts':
                                starts=[i+len(assistant) for i in range(len(ids)) if ids[i:i+len(assistant)]==assistant]
                                if not starts:raise ValueError('Missing assistant marker')
                                required=80+max(len(ids),starts[-1]+row['codec_frames']+2)
                            stats['rows']+=1
                            if required>stats['max_required']:stats.update(max_required=required,worst_sample=row['sample_id'])
                maximum=max(maximum,stats['max_required']);summary[split]=stats
            audit[source]=summary;print('LENGTH_AUDIT',source,summary,flush=True)
        length=max(minimum,math.ceil(maximum/128)*128)
        if length>4096:raise ValueError(f'{phase}: unexpectedly long sequence; investigate rather than truncate')
        phases.append({'name':phase,'sources':entries,'modules':modules,'max_seq_len':length,'epochs':1,
                       'batch_size':batch,'accumulation_steps':4,'learning_rate':lr,'control_token_lr':1e-5,
                       'eval_interval':250,'save_interval':250,'eval_samples_per_source':128,
                       'early_stop_patience':4,'condition_prefix_length':80 if phase=='tts' else 0,
                       'speaker_warmup_fraction':0.1 if phase=='tts' else 0})
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan={'version':1,'data_catalog_sha256':sha(base/'manifest.json'),'voice_manifest_sha256':sha(voice_path),
          'model_path':'model/Qwen3-0.6B','output_root':'out/miniqwen_pretrain_v1','world_size':4,
          'global_batch':128,'seed':42,'phases':phases,'length_audit':audit,
          'formal_training_started':False,'start_requires_explicit_run':True,
          'asset_sha256':asset_fingerprints()}
    if base != ROOT/'dataset/pretrain':
        plan.update(version=2,data_root=str(base.relative_to(ROOT)),output_root='out/miniqwen_pretrain_v2_clean360')
    path=base/'pretrain_plan.json';path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print('PRETRAIN_PLAN_READY',path,flush=True)


if __name__=='__main__':main()
