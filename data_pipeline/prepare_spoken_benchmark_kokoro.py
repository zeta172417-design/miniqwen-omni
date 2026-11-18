#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.machinery, json, os, shutil, subprocess, sys, types
from pathlib import Path

SRC = Path('/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/omni-core-v2.3-standard')
DST = Path('/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/omni-core-v2.3-standard-kokoro')

def setup_kokoro():
    import torch
    tv=types.ModuleType('torchvision'); tv.__path__=[]; tv.__spec__=importlib.machinery.ModuleSpec('torchvision',loader=None,is_package=True)
    tvt=types.ModuleType('torchvision.transforms'); tvt.__spec__=importlib.machinery.ModuleSpec('torchvision.transforms',loader=None)
    class M: NEAREST=0; NEAREST_EXACT=0; BILINEAR=2; BICUBIC=3; BOX=4; HAMMING=5; LANCZOS=1
    tvt.InterpolationMode=M; tv.transforms=tvt; sys.modules['torchvision']=tv; sys.modules['torchvision.transforms']=tvt
    sys.path.insert(0,'/tmp/kokoro_gpu_pkgs')
    from kokoro import KPipeline
    return torch, KPipeline

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int); ap.add_argument('--shards',type=int,default=16); args=ap.parse_args()
    if args.shard is not None:
        torch,KPipeline=setup_kokoro(); torch.set_num_threads(2); pipe=KPipeline(lang_code='a')
        prompts=json.loads((DST/'kokoro_prompts.json').read_text())
        import numpy as np, soundfile as sf
        for item in prompts[args.shard::args.shards]:
            out=DST/'media'/'spoken_qa'/f"{item['idx']:04d}.wav"
            chunks=[]
            for _,_,audio in pipe(item['text'],voice='af_heart',speed=1.0): chunks.append(audio.detach().cpu().numpy())
            sf.write(out,np.concatenate(chunks).astype('float32'),24000,subtype='PCM_16')
        return
    if DST.exists(): shutil.rmtree(DST)
    shutil.copytree(SRC,DST)
    rows=[json.loads(x) for x in (SRC/'manifest.jsonl').read_text().splitlines()]
    prompts=[]; seen=set()
    for r in rows:
        if r.get('task')!='spoken_qa_a2t': continue
        idx=int(Path(r['audio']).stem)
        if idx in seen: continue
        seen.add(idx); prompts.append({'idx':idx,'text':r['metadata']['transcript_prompt']})
    (DST/'kokoro_prompts.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2)+'\n')
    procs=[]
    for shard in range(args.shards):
        env=os.environ.copy(); env.update({'KOKORO_SHARD':str(shard),'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        log=open(DST/f'kokoro-gpu-{shard:02d}.log','w')
        procs.append(subprocess.Popen([sys.executable,__file__,'--shard',str(shard),'--shards',str(args.shards)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in procs]
    if max(codes): raise SystemExit(max(codes))
    (DST/'kokoro_prompts.json').unlink()
    for r in rows:
        if r.get('task','').startswith('spoken_qa_'):
            idx=int(Path(r['audio']).stem); r['audio']=str(DST/'media'/'spoken_qa'/f'{idx:04d}.wav'); r['source']='allenai/ai2_arc:ARC-Easy:test+kokoro-82m'; r.setdefault('metadata',{})['tts_engine']='kokoro-82m'; r['metadata']['voice']='af_heart'
    (DST/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    m=json.loads((DST/'manifest.json').read_text()); m['audio_engine']='kokoro-82m'; m['source_revision']='kokoro-82m'; (DST/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',len(prompts),DST)

if __name__=='__main__': main()
