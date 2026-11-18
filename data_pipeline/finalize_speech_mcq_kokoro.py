#!/usr/bin/env python3
from __future__ import annotations
import io,json,hashlib,math
from pathlib import Path
import numpy as np, soundfile as sf, pyarrow as pa, pyarrow.parquet as pq
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'dataset/raw/external/speech_mcq/10k_audio_kokoro'; OUT=ROOT/'dataset/sft/external/speech_mcq'
from sft_common import SCHEMA
def split(s): return 'dev' if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%10==0 else 'train'
def main():
 import torch
 from transformers import MimiModel
 records=[json.loads(x) for x in (RAW/'records.jsonl').open()]
 model=MimiModel.from_pretrained(ROOT/'model/mimi',local_files_only=True).eval().float().to('cuda');model.config.use_streaming=False
 rows={'train':[],'dev':[]}
 with torch.inference_mode():
  for off in range(0,len(records),64):
   batch=records[off:off+64]; waves=[]; qbytes=[]; qinfo=[]
   for r in batch:
    data=Path(r['q']).read_bytes(); qbytes.append(data); x,sr=sf.read(io.BytesIO(data),dtype='float32'); x=np.asarray(x).reshape(-1); qinfo.append((len(x)/sr,sr));
    y,sr2=sf.read(r['a'],dtype='float32'); y=np.asarray(y).reshape(-1)
    if sr2!=24000:
     g=math.gcd(sr2,24000); y=resample_poly(y,24000//g,sr2//g).astype('float32')
    pad=(-len(y))%1920; y=np.pad(y,(0,pad)); waves.append(y)
   lens=torch.tensor([len(x) for x in waves],dtype=torch.long); frames=model.get_encoded_length(lens.to('cuda')).cpu().tolist(); inp=torch.zeros(len(waves),1,int(lens.max()))
   for i,x in enumerate(waves):inp[i,0,:len(x)]=torch.from_numpy(x)
   codes=model.encode(inp.to('cuda'),num_quantizers=8,use_streaming=False).audio_codes.cpu()
   for i,r in enumerate(batch):
    fr=int(frames[i]); c=codes[i,:,:fr].transpose(0,1).reshape(-1).tolist(); sid=f'speech_mcq10k:{i+off:05d}'; sp=split(sid); answer=r['answer']; prompt='Listen to the question and answer with only the option letter.'
    msgs=[{'role':'user','content':prompt},{'role':'assistant','content':answer}]
    row={'sample_id':sid,'source':'speech_mcq10k','source_row':i+off,'split':sp,'task_type':'audio_mcq_dual','conversations':json.dumps(msgs,ensure_ascii=False),'question_audios':[qbytes[i]],'question_audio_paths':[str(Path(r['q']).relative_to(ROOT))],'answer_audios':[c],'image_paths':[],'ref_audios':[],'spk_emb':[],'codec_frames':fr,'duration':qinfo[i][0],'sample_rate':qinfo[i][1],'language':'en','speaker_id':'kokoro-af_heart','reference_sample_id':None,'reference_speaker_id':None,'voice_self_fallback':False,'group_id':sid,'length_tier':'main_2048','required_length':0,'text_length':0,'audio_frames_upper_bound':math.ceil(qinfo[i][0]/.06)+2,'voice_conditioning':'none','supervision_policy':'all_text_last_audio','raw_source_parquet':'dataset/raw/external/speech_mcq/source/{0}.parquet'.format(r['source']),'raw_source_row':r['i'],'metadata_json':json.dumps({'source':r['source'],'answer':answer,'output_modalities':['text','audio'],'input_modality':'audio','tts_engine':'kokoro-82m','codec_model':'model/mimi'},ensure_ascii=False)}
    rows[sp].append(row)
   print('ENCODED',min(off+64,len(records)),'/',len(records),flush=True)
 for sp,rs in rows.items():
  pq.write_table(pa.Table.from_pylist(rs,schema=SCHEMA),OUT/f'{sp}.parquet',compression='zstd',row_group_size=128)
 m={'status':'COMPLETE','rows':sum(map(len,rows.values())),'train_rows':len(rows['train']),'dev_rows':len(rows['dev']),'source_counts':{'arc':2251,'boolq':3875,'piqa':3874},'output_modalities':['text','audio'],'input_modality':'audio','codec_frames':sum(r['codec_frames'] for rs in rows.values() for r in rs)}
 (OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print(json.dumps(m,ensure_ascii=False))
if __name__=='__main__':main()
