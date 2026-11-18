#!/usr/bin/env python3
"""Bounded reader acceptance; optional untrained GPU forward without optimization."""
import os,json,sys,tempfile,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.runtime/numba_cache'))
import pyarrow.parquet as pq
import torch
from transformers import SiglipImageProcessor
from dataset.omni_dataset import OmniDataset
from dataset.sft_dataset import SFTDataset
from model.model_omni import MiniQwenOmni
from trainer.trainer_utils import load_omni_tokenizer
from trainer.training_common import omni_collate_fn

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
 tok=load_omni_tokenizer(ROOT/'model/Qwen3-0.6B');encoder,audio=MiniQwenOmni.load_sensevoice(ROOT/'model/SenseVoiceSmall');del encoder;gc.collect()
 vision=SiglipImageProcessor.from_pretrained(ROOT/'model/siglip2-base-p32-256-ve',local_files_only=True)
 report=[];forward_items=[];manifest=json.loads((ROOT/'dataset/sft/manifest.json').read_text())
 def check(path,kind):
  pf=pq.ParquetFile(path);n=pf.metadata.num_rows
  if not n:return
  if kind=='sft':
   lengths=pf.read(columns=['required_length'])['required_length'].to_pylist();indices=sorted({0,n-1,max(range(n),key=lengths.__getitem__)})
   if path.name.startswith('m4-'):
    images=pf.read(columns=['image_paths'])['image_paths'].to_pylist();indices=sorted(set(indices)|{max(range(n),key=lambda i:len(images[i]))})
   if path.name.startswith('fusion-'):
    tiny=pf.read(columns=['codec_frames','question_audio_paths']).to_pylist()
    match=next((i for i,r in enumerate(tiny) if r['codec_frames'] and r['question_audio_paths']),None)
    if match is not None:indices=sorted(set(indices)|{match})
  else:indices=sorted({0,n-1})
  ds=(SFTDataset if kind=='sft' else OmniDataset)(str(path),tok,audio,vision,max_length=3072 if kind=='sft' else 768,training=False,augment=False,scheduled_sampling=0,strict_length=True,condition_prefix_length=160,voice_conditioning='available_both',use_modality_boundaries=True,use_talker_ref_boundaries=True,max_images=4)
  for i in indices:
   row=ds.dataset.rows[i] if kind=='sft' else ds.dataset[i];item=ds[i]
   ids,mask,text,codes,*_=item
   assert text.ne(-100).any()
   frames=row.get('codec_frames') or 0
   if frames:assert int(codes[0].ne(-100).sum())==frames+1
   batch=omni_collate_fn([item],dynamic_padding=True,pad_to_multiple=8)
   assert int(batch[2].ne(-100).sum())==int(text.ne(-100).sum())
   assert int(batch[3].ne(-100).sum())==int(codes.ne(-100).sum())
   if i==indices[0] or (kind=='sft' and ('long_text_3072' in path.parts or row.get('source') in {'m4','fusion','old_a2a'})):
    forward_items.append((row['sample_id'],item))
   if args.gpu and kind=='sft' and row.get('source') in {'libritts_r','aishell3'} and i==indices[0]:
    from dataset.voice_conditioning import choose_voice_mode
    found={}
    for epoch in range(200):found.setdefault(choose_voice_mode(row,ds.condition_seed,epoch)[1],epoch)
    assert set(found)=={'both','speaker','reference','none'}
    ds.training=True
    for mode,epoch in found.items():
     draw=ds[(i,epoch)]
     assert bool(draw[0][0].eq(ds.audio_spk_token).any())==(mode in {'both','speaker'})
     assert bool(draw[0][0].eq(ds.audio_ref_start_token).any())==(mode in {'both','reference'})
     forward_items.append((row['sample_id']+':'+mode,draw))
    ds.training=False
   report.append({'sample_id':row['sample_id'],'path':str(path.relative_to(ROOT)),'index':i,'text_targets':int(text.ne(-100).sum()),'audio_frames':frames,'supervision_policy':row['supervision_policy']})
  print('READER_PASS',kind,path.parent.name,path.name,flush=True)
 for source,v in manifest['sources'].items():
  spec=json.loads((ROOT/v['manifest']).read_text())
  for key,paths in spec['data_files'].items():
   if key.endswith('/train') and paths:
    check(ROOT/paths[0],'sft')
    if source=='old_a2a' and len(paths)>1:check(ROOT/paths[-1],'sft')
 cat=json.loads((ROOT/'dataset/raw/manifests/pretrain/manifest.json').read_text())
 for source,spec in cat['sources'].items():
  for split in ['train','dev']:
   paths=spec['data_files'][split]
   if paths:check(ROOT/paths[0],'pretrain')
 if args.gpu:
  from model.model_omni import OmniConfig
  from trainer.trainer_utils import init_omni_model,setup_seed
  from trainer.train_pretrain import forward,objective,move
  assert torch.cuda.is_available(),'GPU is unavailable'
  setup_seed(42)
  config=OmniConfig.from_qwen_pretrained(str(ROOT/'model/Qwen3-0.6B'),talker_condition_prefix_length=160,use_mrope=True,use_modality_boundaries=True,use_talker_ref_boundaries=True)
  model,_=init_omni_model(config,from_weight='qwen',tokenizer_path=str(ROOT/'model/Qwen3-0.6B'),audio_encoder_path=str(ROOT/'model/SenseVoiceSmall'),vision_model_path=str(ROOT/'model/siglip2-base-p32-256-ve'),device='cuda:0')
  model.audio_encoder.to('cuda:0');model.vision_encoder.to('cuda:0');model.eval()
  checks=[];forward_batches=[]
  for length in sorted({item[0].shape[-1] for _,item in forward_items}):
   compatible=[pair for pair in forward_items if pair[1][0].shape[-1]==length]
   forward_batches.extend(compatible[i:i+2] for i in range(0,len(compatible),2))
  with torch.inference_mode():
   for subset in forward_batches:
    batch=move(omni_collate_fn([x[1] for x in subset],dynamic_padding=True),torch.device('cuda:0'))
    with torch.autocast('cuda',dtype=torch.bfloat16):
     output=forward(model,batch,'tts');loss=objective(output,batch[2],batch[3],config,1)
    assert torch.isfinite(loss)
    checks.append({'sample_ids':[x[0] for x in subset],'length':batch[0].shape[-1],'loss':float(loss),'text_targets':int(batch[2].ne(-100).sum()),'main_audio_targets':int(batch[3][:,0].ne(-100).sum())})
    del output,loss,batch
    print('UNTRAINED_GPU_FORWARD_PASS',len(checks),flush=True)
  assert all(p.grad is None for p in model.parameters())
  gpuresult={'status':'PASS','initialization':'local pretrained Qwen3-0.6B backbone; untrained randomly initialized multimodal modules','device':torch.cuda.get_device_name(0),'precision':'FP32 parameters + BF16 autocast','checks':checks,'optimizer_steps':0,'checkpoint_saved':False}
  (ROOT/'.runtime/data-finalize/forward-validation.json').write_text(json.dumps(gpuresult,indent=2)+'\n')
 result={'status':'PASS','samples':len(report),'records':report,'GPU_forward':'UNTRAINED_MODEL_PASS' if args.gpu else 'NOT_RUN','optimizer_steps':0}
 (ROOT/'.runtime/data-finalize/reader-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('READER_ACCEPTANCE_PASS',len(report),flush=True)
if __name__=='__main__':main()
