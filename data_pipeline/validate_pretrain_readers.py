#!/usr/bin/env python3
"""Prepare actual training caches and check extreme rows at the launch budgets."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import SiglipImageProcessor
from dataset.omni_dataset import OmniDataset
from model.model_omni import MiniQwenOmni
from trainer.trainer_utils import load_omni_tokenizer

def main():
    base=ROOT/os.environ.get('MINIQWEN_PRETRAIN_DATA','dataset/pretrain_v2')
    plan=json.loads((base/'pretrain_plan.json').read_text())
    tok=load_omni_tokenizer(ROOT/plan['model_path'])
    encoder,audio=MiniQwenOmni.load_sensevoice(ROOT/'model/SenseVoiceSmall');del encoder
    vision=SiglipImageProcessor.from_pretrained(ROOT/'model/siglip2-base-p32-256-ve',local_files_only=True)
    result={}
    for phase in plan['phases']:
        views={'train':([f for src in phase['sources'].values() for f in src['train']],True)}
        views.update({name+'/dev':(src['dev'],False) for name,src in phase['sources'].items()})
        worst={v['worst_sample'] for name in phase['sources'] for v in plan['length_audit'][name].values()}
        for name,(paths,training) in views.items():
            ds=OmniDataset(','.join(str(ROOT/f) for f in paths),tok,audio,vision,max_length=phase['max_seq_len'],training=training,augment=False,scheduled_sampling=0,strict_length=True,condition_prefix_length=phase['condition_prefix_length'],voice_conditioning='mixed' if training else 'both',use_modality_boundaries=True,use_talker_ref_boundaries=True)
            indices={0,len(ds)//2,len(ds)-1}
            for i,sid in enumerate(ds.dataset['sample_id']):
                if sid in worst:indices.add(i)
            checks=[]
            for i in sorted(indices):
                ids,mask,text,codes,features,nframes,pixels,speaker=ds[i];row=ds.dataset[i]
                assert text.ne(-100).any(),row['sample_id']
                if phase['name']=='audio':assert nframes>0 and ids[8].eq(ds.audio_token_id).any()
                if phase['name']=='vision':assert pixels is not None and pixels.any() and int(ids[8].eq(ds.image_token_id).sum())==64
                if phase['name']=='tts':
                    assert int(codes[0].ne(-100).sum())==row['codec_frames']+1
                    assert int(codes[1].ne(-100).sum())==row['codec_frames']
                checks.append(row['sample_id'])
            result[phase['name']+'/'+name]={'rows':len(ds),'max_seq_len':phase['max_seq_len'],'checked_sample_ids':checks}
            print('READER_CACHE_PASS',phase['name'],name,len(ds),len(checks),flush=True)
            del ds
    log=ROOT/os.environ.get('MINIQWEN_PRETRAIN_LOG','.runtime/pretrain-v2' if base.name=='pretrain_v2' else '.runtime/pretrain');log.mkdir(parents=True,exist_ok=True)
    path=log/'readers-validation.json';path.write_text(json.dumps({'status':'PASS','views':result},indent=2))
if __name__=='__main__':main()
