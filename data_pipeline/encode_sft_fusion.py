#!/usr/bin/env python3
"""Encode selected speech-output views with resumable, validated Mimi shards."""
import argparse,collections,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import MimiModel
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sft_common import *

def fusion_view(sid):return int(digest_text('sft-v5-view:42:'+sid)[:16],16)%4

def main():
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--device', default='cpu')
 parser.add_argument('--batch-size', type=int, default=2)
 args = parser.parse_args()
 if args.batch_size < 1:
  parser.error('--batch-size must be positive')
 os.nice(10)
 torch.set_num_threads(2)
 device = torch.device(args.device)
 if device.type == 'cuda' and not torch.cuda.is_available():
  raise RuntimeError('Requested GPU device is unavailable')
 torch.backends.cuda.matmul.allow_tf32 = False
 torch.backends.cudnn.allow_tf32 = False
 while not (META/'media-groups.parquet').exists():time.sleep(5)
 groupmap={r['path']:r for r in pq.read_table(META/'media-groups.parquet').to_pylist()};rows=[]
 for split in ('train','dev'):
  for b in pq.ParquetFile(OUT/'external/fusion'/f'{split}.parquet').iter_batches(batch_size=256,columns=['record_json']):
   for item in b.column(0).to_pylist():
    r=json.loads(item);view=fusion_view(r['sample_id'])
    if view<2:continue
    asset=groupmap[resolve_media(r['image_paths'][0])]
    if asset['blocked_benchmark_image']:continue
    rows.append({'sample_id':r['sample_id'],'path':resolve_media(r['answer_audio_path']),'duration':r['target_seconds']})
 rows.sort(key=lambda r:(r['duration'],r['sample_id']));dest=R/'dataset/raw/corpora/sft_fusion_codec';dest.mkdir(parents=True,exist_ok=True)
 model=MimiModel.from_pretrained(R/'model/mimi',local_files_only=True).eval().float().to(device);model.config.use_streaming=False
 schema=pa.schema([('sample_id',pa.string()),('codec_frames',pa.int32()),('codec_padding_samples',pa.int32()),('codes',pa.list_(pa.int64())),('source_audio_path',pa.string()),('audio_sha256',pa.string())])
 started=time.time();total=0
 def waveform(r):
  data=(R/r['path']).read_bytes();import io
  wav,sr=sf.read(io.BytesIO(data),dtype='float32',always_2d=True);wav=wav.mean(1)
  if not len(wav) or not np.isfinite(wav).all():raise ValueError('Invalid fusion waveform')
  if sr!=24000:
   g=math.gcd(sr,24000);wav=resample_poly(wav,24000//g,sr//g).astype(np.float32)
  padding=(-len(wav))%1920;wav=np.pad(wav,(0,padding));return wav,padding,hashlib.sha256(data).hexdigest()
 with torch.inference_mode():
  for offset in range(0,len(rows),64):
   subset=rows[offset:offset+64];path=dest/f'part-{offset//64:03d}.parquet'
   if path.exists():
    f=pq.ParquetFile(path);assert f.metadata.num_rows==len(subset);assert f.read(columns=['sample_id']).column(0).to_pylist()==[r['sample_id'] for r in subset];total+=len(subset);continue
   output=[]
   for start in range(0,len(subset),args.batch_size):
    selected=subset[start:start+args.batch_size];waves=[waveform(r) for r in selected];lens=torch.tensor([len(w) for w,_,_ in waves],dtype=torch.long);flens=model.get_encoded_length(lens.to(device)).cpu().tolist();x=torch.zeros(len(waves),1,int(lens.max()))
    for i,(w,_,_) in enumerate(waves):x[i,0,:len(w)]=torch.from_numpy(w)
    codes=model.encode(x.to(device),num_quantizers=8,use_streaming=False).audio_codes.cpu()
    for i,r in enumerate(selected):
     frames=int(flens[i]);c=codes[i,:,:frames].transpose(0,1).reshape(-1)
     if len(c)!=8*frames or c.min()<0 or c.max()>=2048:raise ValueError('Invalid Mimi output')
     output.append({'sample_id':r['sample_id'],'codec_frames':frames,'codec_padding_samples':waves[i][1],'codes':c.tolist(),'source_audio_path':r['path'],'audio_sha256':waves[i][2]})
   tmp=path.with_suffix('.parquet.partial');pq.write_table(pa.Table.from_pylist(output,schema=schema),tmp,compression='zstd',row_group_size=32);tmp.replace(path);total+=len(output)
   save(W/'fusion-codec-status.json',{'status':'ENCODING','device':str(device),'rows':total,'total_rows':len(rows),'elapsed_seconds':time.time()-started});print('FUSION_CODEC',total,'/',len(rows),'elapsed',round(time.time()-started),flush=True)
 save(dest/'manifest.json',{'status':'COMPLETE','rows':total,'selected_view_policy':'Seeded views2/3 only; one view per source record; blocked evaluation image groups excluded.','precision':'float32','device':str(device),'reused_existing_shards':True,'batch_size_new_shards':args.batch_size,'sample_rate':24000,'codebooks':8,'frame_rate':12.5,'padding':'right zero to1920 samples, no target cutting','mimi_model':'model/mimi','data_files':[str(p.relative_to(R)) for p in sorted(dest.glob('part-*.parquet'))]});save(W/'fusion-codec-status.json',{'status':'COMPLETE','rows':total});print('FUSION_CODEC_COMPLETE',total,flush=True)
if __name__=='__main__':main()
