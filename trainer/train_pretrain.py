#!/usr/bin/env python3
"""Frozen-backbone alignment trainer. Execution requires --execute or bounded --smoke."""
import argparse
from collections import deque
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from dataset.omni_dataset import OmniDataset
from model.model_omni import OmniConfig
from trainer.training_common import configure_trainable_modules, compute_audio_losses, build_talker_attention_mask, omni_collate_fn
from trainer.trainer_utils import init_distributed_mode, setup_seed, init_omni_model, omni_checkpoint, SkipBatchSampler, configure_token_ids, load_omni_tokenizer
from trainer.metrics import empty_metrics, update_metrics, finalise
from trainer.pretrain_checkpoint import recover_checkpoint
from trainer.pretrain_plan import validate_checkpoint_signature
from trainer.pretrain_sampler import EpochBatchSampler, make_pretrain_sampler


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tune_controls(model, phase):
    """Only modality-control rows may change; AdamW decay must remain zero here."""
    names={'audio':['audio_ids','audio_start_ids','audio_end_ids'],
           'vision':['image_ids','vision_start_ids','vision_end_ids'],'tts':[]}[phase]
    ids=sorted({i for name in names for i in getattr(model.config,name)})
    if not ids:return [],[]
    parameter=model.model.embed_tokens.weight
    parameter.requires_grad_(True)
    def mask_gradient(grad):
        selected=torch.tensor(ids,device=grad.device)
        kept=grad.index_select(0,selected)
        output=torch.zeros_like(grad);output.index_copy_(0,selected,kept)
        return output
    parameter.register_hook(mask_gradient)
    return [parameter],ids


def speaker_only(inputs, pad=2049, bos=2048, speaker=2051, available=None):
    output=inputs.clone();main=output[:,0]
    for i in range(output.shape[0]):
        if available is not None and not bool(available[i]):continue
        starts=main[i].eq(bos).nonzero()
        if not len(starts):raise ValueError('Speaker warmup requires audio BOS')
        pos=int(starts[0]);output[i,:8,:pos]=pad;output[i,0,pos-1]=speaker
    return output


def condition_ablation(inputs, mode, pad=2049, bos=2048, speaker=2051):
    output=inputs.clone()
    for i in range(output.shape[0]):
        starts=output[i,0].eq(bos).nonzero()
        if not len(starts):raise ValueError('Missing BOS')
        end=int(starts[0]);prefix=output[i,:8,:end]
        if mode=='speaker':prefix.masked_fill_(prefix[0].ne(speaker).unsqueeze(0),pad)
        elif mode=='reference':prefix.masked_fill_(prefix[0].eq(speaker).unsqueeze(0),pad)
    return output


def move(batch,device):return tuple(x.to(device) if isinstance(x,torch.Tensor) else x for x in batch)

def reduce_validation_sums(sums,device):
    """Reduce additive counts/NLL, never average per-rank metric means."""
    keys=list(sums);values=[]
    for key in keys:
        values.extend(sums[key] if isinstance(sums[key],list) else [sums[key]])
    tensor=torch.tensor(values,device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(tensor)
    values=tensor.cpu().tolist();offset=0;result={}
    for key in keys:
        n=len(sums[key]) if isinstance(sums[key],list) else 1
        result[key]=values[offset:offset+n] if isinstance(sums[key],list) else values[offset]
        offset+=n
    return result

def control_row_comm_hook(state,bucket):
    """Communicate only selected embedding rows; preserve dense optimizer state."""
    control_parameter,ids,world=state
    parameters=bucket.parameters();gradients=bucket.gradients()
    selected=torch.tensor(ids,device=bucket.buffer().device,dtype=torch.long)
    pieces=[]
    for parameter,gradient in zip(parameters,gradients):
        pieces.append(gradient.index_select(0,selected).flatten() if parameter is control_parameter else gradient.flatten())
    compact=torch.cat(pieces)
    dist.all_reduce(compact);compact.div_(world)
    offset=0
    for parameter,gradient,piece in zip(parameters,gradients,pieces):
        values=compact[offset:offset+piece.numel()];offset+=piece.numel()
        if parameter is control_parameter:
            gradient.zero_();gradient.index_copy_(0,selected,values.view(len(ids),*gradient.shape[1:]))
        else:gradient.copy_(values.view_as(gradient))
    future=torch.futures.Future();future.set_result(bucket.buffer());return future


def forward(model,batch,phase):
    ids,attention,labels,audio_labels,audio,lens,pixels,speaker=batch
    result=model(ids,attention_mask=attention,
                 talker_attention_mask=build_talker_attention_mask(attention,audio_labels,ids),
                 audio_inputs=audio,audio_lens=lens,pixel_values=pixels,spk_emb=speaker,
                 use_cache=False,output_audio_logits=phase=='tts',audio_targets=audio_labels,
                 text_logits_mask=labels.ne(-100))
    return result


def objective(result, labels, audio_labels, config, world, return_logging=False,
              normalization_counts=None):
    """Global valid-token means; DDP averaging cancels the world multiplier."""
    mask=labels.ne(-100);targets=labels[mask]
    text=F.cross_entropy(result.logits.reshape(-1,result.logits.shape[-1]),targets,reduction='sum')
    counts=torch.tensor([targets.numel(),int(audio_labels[:,0].ne(-100).sum()),
                         int(audio_labels[:,1].ne(-100).sum())],device=labels.device,dtype=torch.float64)
    total=counts.clone() if normalization_counts is None else normalization_counts
    if normalization_counts is None and dist.is_initialized():dist.all_reduce(total)
    audio,main,residual,_=compute_audio_losses(result,audio_labels,config)
    loss=text*world/total[0].clamp_min(1)
    loss+=main*counts[1]*world/total[1].clamp_min(1)
    loss+=config.residual_codec_loss_weight*residual*counts[2]*world/total[2].clamp_min(1)
    objective_loss=loss+result.aux_loss
    if not return_logging:return objective_loss
    # Detached additive statistics: accumulate microbatches before reducing ranks.
    stats=torch.stack([text.detach().double(),counts[0],
                       main.detach().double()*counts[1],counts[1],
                       residual.detach().double()*counts[2],counts[2],
                       result.aux_loss.detach().double(),counts.new_tensor(1),
                       counts.new_tensor(labels.shape[0])])
    return objective_loss,stats


def training_metrics(stats, residual_weight):
    values=stats.cpu().tolist()
    text=values[0]/max(values[1],1);main=values[2]/max(values[3],1)
    residual=values[4]/max(values[5],1);aux=values[6]/max(values[7],1)
    return {'train/loss':text+main+residual_weight*residual+aux,
            'train/text_loss':text,'train/main_codec_loss':main,
            'train/residual_codec_loss':residual,'train/aux_loss':aux,
            'train/text_tokens':values[1],'train/main_codec_tokens':values[3],
            'train/residual_frames':values[5],'train/samples':values[8],
            'train/rank_microbatches':values[7]}


def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'device':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['device'] is not None:torch.cuda.set_rng_state(state['device'],device)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)


def backbone_digest(model, control_ids):
    digest=hashlib.sha256()
    for name,p in model.model.named_parameters():
        data=p.detach().cpu()
        if name=='embed_tokens.weight' and control_ids:
            keep=torch.ones(data.shape[0],dtype=torch.bool);keep[control_ids]=False;data=data[keep]
        digest.update(data.contiguous().numpy().tobytes())
    return digest.hexdigest()


def timed_batches(loader):
    iterator=iter(loader)
    while True:
        started=time.perf_counter()
        try:item=next(iterator)
        except StopIteration:return
        yield item,time.perf_counter()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default=str(ROOT/'dataset/raw/manifests/pretrain/plan.json'))
    p.add_argument('--phase',choices=['audio','vision','tts'],required=True)
    p.add_argument('--from-weight',default='qwen')
    p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--num-workers',type=int,default=None)
    p.add_argument('--prefetch-factor',type=int,default=None)
    p.add_argument('--memory-cache-mib',type=int,default=None)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--use-swanlab',action='store_true')
    p.add_argument('--ddp-no-sync',type=int,choices=[0,1],default=None)
    p.add_argument('--benchmark-steps',type=int,default=0)
    p.add_argument('--benchmark-warmup',type=int,default=3)
    args=p.parse_args()
    benchmarking=args.benchmark_steps>0
    if not args.execute and not args.smoke:p.error('No training started: supply --execute only after launch authorization, or bounded --smoke')
    if args.smoke and not 1<=args.max_steps<=2:p.error('Smoke is limited to 1–2 optimizer updates')
    output=Path(args.output).resolve()
    if args.smoke and not output.is_relative_to(ROOT/'.runtime'):p.error('Smoke outputs must stay in .runtime')
    if benchmarking and (not output.is_relative_to(ROOT/'.runtime') or args.resume or args.smoke):
        p.error('Benchmarks require fresh outputs below .runtime, no resume/smoke')
    if benchmarking and not 0<=args.benchmark_warmup<args.benchmark_steps:
        p.error('Benchmark warmup must be smaller than benchmark steps')
    local=init_distributed_mode();rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    plan=json.loads(Path(args.plan).read_text());phase=next(x for x in plan['phases'] if x['name']==args.phase)
    if type(phase['epochs']) is not int or phase['epochs']<1:raise ValueError('epochs must be a positive integer')
    if not args.smoke and not benchmarking:
        if (ROOT/plan.get('data_root','dataset/pretrain')/'RETIRED_AFTER_CLEANUP.json').exists():
            raise ValueError('Retired data view; use dataset/pretrain_v2/pretrain_plan.json')
        if world!=plan['world_size']:raise ValueError('Formal world size must match the reviewed plan')
    execution={**plan.get('execution',{}),**phase.get('loader',{})}
    workers=args.num_workers if args.num_workers is not None else execution.get('num_workers',2)
    prefetch=args.prefetch_factor if args.prefetch_factor is not None else execution.get('prefetch_factor',2)
    memory_cache=args.memory_cache_mib if args.memory_cache_mib is not None else execution.get('parquet_memory_cache_mib_per_process',128)
    if workers<0 or prefetch<1 or memory_cache<0:raise ValueError('Invalid data loader settings')
    os.environ['MINIQWEN_PARQUET_MEMORY_CACHE_MIB']=str(memory_cache)
    seed=plan['seed'];setup_seed(seed)
    signature=hashlib.sha256(json.dumps({'plan':plan,'phase':args.phase,'world_size':world},sort_keys=True).encode()).hexdigest()
    state_path=output/'last/checkpoint/trainer_state.pt'
    if args.resume:
        if rank==0:recover_checkpoint(output/'last')
        if world>1:dist.barrier()
    elif (output/'last/.previous-checkpoint').exists():
        raise ValueError('An earlier checkpoint backup exists; use --resume')
    if args.resume and not state_path.exists():raise ValueError('Requested resume checkpoint does not exist')
    if not args.resume and state_path.exists():raise ValueError('Output already has a checkpoint; use --resume or a fresh output')
    saved=torch.load(state_path,map_location='cpu',weights_only=False) if args.resume else None
    if saved and saved.get('optimizer_file') and 'optimizer' not in saved:
        optimizer_path=state_path.parent/saved['optimizer_file']
        if not optimizer_path.is_file():raise ValueError(f'Missing optimizer checkpoint: {optimizer_path}')
        saved['optimizer']=torch.load(optimizer_path,map_location='cpu',weights_only=False)
    if saved and 'optimizer' not in saved:raise ValueError('Resume requires last checkpoint with optimizer state')
    migrated_signature=validate_checkpoint_signature(plan,args.phase,world,saved.get('pretrain_signature'),ROOT) if saved else None
    if migrated_signature and rank==0:print('EXECUTION_MIGRATION',args.phase,migrated_signature,flush=True)
    if saved and args.max_steps and saved.get('global_step',0)>=args.max_steps:
        raise ValueError('Requested step limit is already reached; no extra update will run')
    config=OmniConfig.from_qwen_pretrained(str(ROOT/plan['model_path']),talker_condition_prefix_length=160,
                                         use_mrope=True,use_modality_boundaries=True,use_talker_ref_boundaries=True)
    weight=str(state_path.parent) if saved else args.from_weight
    model,tokenizer=init_omni_model(config,from_weight=weight,tokenizer_path=str(ROOT/plan['model_path']),
                                  audio_encoder_path=str(ROOT/'model/SenseVoiceSmall'),
                                  vision_model_path=str(ROOT/'model/siglip2-base-p32-256-ve'),device=str(device))
    if model.audio_encoder is not None:model.audio_encoder.to(device)
    if model.vision_encoder is not None:model.vision_encoder.to(device)
    configure_trainable_modules(model,phase['modules'])
    controls,control_ids=tune_controls(model,args.phase)
    control_keys={id(x) for x in controls}
    groups=[{'params':[x for x in model.parameters() if x.requires_grad and id(x) not in control_keys],
             'lr':phase['learning_rate'],'base_lr':phase['learning_rate'],'name':'alignment','weight_decay':0.01}]
    if controls:groups.append({'params':controls,'lr':phase['control_token_lr'],'base_lr':phase['control_token_lr'],'name':'control_rows','weight_decay':0.0})
    assert all(not x.requires_grad for layer in model.model.layers for x in layer.parameters())
    optimizer=torch.optim.AdamW(groups)
    no_sync=bool(phase.get('ddp_no_sync',plan.get('execution',{}).get('ddp_no_sync',False)) if args.ddp_no_sync is None else args.ddp_no_sync)
    distributed_eval=bool(plan.get('execution',{}).get('distributed_eval',False)) and world>1
    if saved:optimizer.load_state_dict(saved['optimizer'])
    def dataset(paths,training):
        return OmniDataset(','.join(str(ROOT/x) for x in paths),tokenizer,model.audio_processor,model.vision_processor,
                           max_length=phase['max_seq_len'],training=training,augment=False,scheduled_sampling=0,
                           strict_length=True,condition_prefix_length=phase['condition_prefix_length'],
                           voice_conditioning=(phase.get('voice_conditioning','mixed') if training
                                               else phase.get('eval_voice_conditioning','both')),
                           use_modality_boundaries=True,use_talker_ref_boundaries=True)
    train=dataset([f for src in phase['sources'].values() for f in src['train']],True)
    dev={name:dataset(src['dev'],False) for name,src in phase['sources'].items()}
    sampler=make_pretrain_sampler(train.dataset,num_replicas=world,rank=rank,seed=seed,mode=phase.get('sampling',plan.get('execution',{}).get('sampling','global_random')))
    sampler.set_epoch(0)
    batch_size=phase['batch_size'];accum=phase['accumulation_steps']
    micro_iters=math.ceil(len(sampler)/batch_size)*phase['epochs'];total_steps=math.ceil(micro_iters/accum)
    start_micro=saved.get('micro_step',0) if saved else 0
    step=saved.get('global_step',0) if saved else 0
    best=saved.get('best_loss',float('inf')) if saved else float('inf');bad=saved.get('bad_evals',0) if saved else 0
    finished=bool(saved and saved.get('phase_complete'))
    if saved:restore_rng(saved['rng_state_by_rank'][rank],device)
    frozen_before=backbone_digest(model,control_ids) if args.smoke else None
    changed_probe=[(p,p.detach().flatten()[:256].cpu().clone()) for p in model.parameters()
                   if args.smoke and p.requires_grad and id(p) not in control_keys]
    initial_step=step
    if args.smoke and device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    raw=model
    # Talker already attaches zero-valued dependencies for inactive branches.
    # find_unused reconstruction would drop the custom codec output attributes.
    if world>1:model=DDP(model,device_ids=[local],find_unused_parameters=False)
    if world>1 and controls and phase.get('compress_control_gradients',plan.get('execution',{}).get('compress_control_gradients',False)):
        model.register_comm_hook((controls[0],control_ids,world),control_row_comm_hook)
    collate=partial(omni_collate_fn,dynamic_padding=True,pad_to_multiple=8)
    loader=DataLoader(train,batch_sampler=EpochBatchSampler(sampler,batch_size,phase['epochs'],start_micro,condition_epochs=True),collate_fn=collate,
                      num_workers=workers,pin_memory=device.type=='cuda',
                      **({'prefetch_factor':prefetch,'persistent_workers':True} if workers else {}),
                      generator=torch.Generator().manual_seed(seed+rank))
    autocast=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    swanlab=None
    if args.use_swanlab and rank==0:
        import swanlab as swanlab
        tracking=plan.get('tracking',{})
        options=dict(project=tracking.get('project','MiniQwen-Omni-Pretrain'),
                     workspace=tracking.get('workspace'),mode='cloud',logdir=str(output/'swanlog'),
                     name=tracking.get('name_prefix','')+args.phase,
                     config={'signature':signature,'phase':phase,'world_size':world,
                             'resumed_global_step':step,'execution':plan.get('execution',{}),
                             'migrated_from_signature':migrated_signature})
        if saved and saved.get('swanlab_id'):options.update(id=saved['swanlab_id'],resume='allow')
        swanlab.init(**options)
        run=swanlab.get_run()
        atomic_json(output/'swanlab.json',{key:str(getattr(run,key,'')) for key in ['id','url','name']})
    def evaluate():
        raw.eval();metrics={}
        with torch.inference_mode():
            for name,ds in dev.items():
                indices=random.Random(seed).sample(range(len(ds)),min(len(ds),phase['eval_samples_per_source']))
                if args.phase=='tts':indices=[i for i in indices if not ds.dataset[i].get('voice_self_fallback')]
                if distributed_eval:indices=indices[rank::world]
                modes=['correct','speaker','reference'] if args.phase=='tts' else ['correct','missing_modality']
                for mode in modes:
                    sums=empty_metrics()
                    for item in DataLoader(Subset(ds,indices),batch_size=1,collate_fn=collate,num_workers=0):
                        item=list(move(item,device))
                        if mode in {'speaker','reference'}:item[0]=condition_ablation(item[0],mode)
                        if mode=='missing_modality':
                            if args.phase=='audio':item[4]=torch.zeros_like(item[4])
                            else:item[6]=None
                        with autocast():result=forward(raw,item,args.phase)
                        update_metrics(sums,result,item[2],item[3],config.audio_stop_token)
                    if distributed_eval:sums=reduce_validation_sums(sums,device)
                    metrics[name+'/'+mode]=finalise(sums,config.residual_codec_loss_weight)
        raw.train()
        score=sum(v['text_loss']+v.get('audio_loss',0) for k,v in metrics.items() if k.endswith('/correct'))/len(dev)
        return score,metrics
    loss_history=deque(saved.get('loss_history',[]) if saved else [],maxlen=100)
    def save(kind,micro,complete=False,with_optimizer=True):
        states=[None]*world
        local_rng=rng_state(device)
        if world>1:dist.all_gather_object(states,local_rng)
        else:states=[local_rng]
        if rank==0:
            omni_checkpoint(config,weight=kind,save_dir=str(output),model=raw,optimizer=optimizer,tokenizer=tokenizer,
                            save_optimizer_state=with_optimizer,epoch=micro//math.ceil(len(sampler)/batch_size),step=micro,
                            micro_step=micro,global_step=step,phase_complete=complete,pretrain_signature=signature,
                            best_loss=best,bad_evals=bad,rng_state_by_rank=states,stage_id=args.phase,training_format_version=5,
                            execution_migrated_from=migrated_signature,loss_history=list(loss_history))
        if world>1:dist.barrier()
    if rank==0:print('PRETRAIN_PHASE',args.phase,'rows',len(train),'global_batch',batch_size*world*accum,'max_seq',phase['max_seq_len'],'frozen_backbone',True,flush=True)
    model.train();last_eval=None
    if not saved and not benchmarking and plan.get('execution',{}).get('eval_at_start',False):
        before_eval=rng_state(device)
        if world>1:dist.barrier()
        baseline,metrics=evaluate() if rank==0 or distributed_eval else (0,{})
        restore_rng(before_eval,device)
        if rank==0:
            atomic_json(output/'validation-initial.json',{'step':0,'objective':baseline,'metrics':metrics})
            if swanlab:
                logged={'dev/objective':baseline}
                for source,values in metrics.items():
                    for key,value in values.items():
                        if isinstance(value,(int,float)):logged[f'dev/{source}/{key}']=value
                swanlab.log(logged,step=0)
            print('PRETRAIN_DEV_INITIAL',args.phase,baseline,flush=True)
        if world>1:dist.barrier()
    window_stats=torch.zeros(9,device=device,dtype=torch.float64)
    benchmark_started=None;benchmark_times=[];benchmark_last=None;benchmark_waits=[];data_wait=0.0
    if benchmarking and device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    if not finished:
        for micro,(item,wait_seconds) in enumerate(timed_batches(loader),start=start_micro+1):
            data_wait+=wait_seconds
            batch=list(move(item,device))
            if args.phase=='tts' and step<math.ceil(total_steps*phase['speaker_warmup_fraction']):
                batch[0]=speaker_only(batch[0],available=batch[7].ne(0).any(dim=1))
            window_start=((micro-1)//accum)*accum
            divisor=min(accum,micro_iters-window_start)
            lrstep=step+1;warm=max(1,math.ceil(total_steps*0.03))
            factor=min(1,lrstep/warm)*(0.1+0.9*(1+math.cos(math.pi*min(lrstep,total_steps)/total_steps))/2)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            update_boundary=micro%accum==0 or micro==micro_iters
            sync_context=model.no_sync() if no_sync and world>1 and not update_boundary else nullcontext()
            with sync_context:
                with autocast():
                    result=forward(model,batch,args.phase)
                    loss,stats=objective(result,batch[2],batch[3],config,world,return_logging=True)
                    loss=loss/divisor
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite pretrain loss')
                loss.backward()
            window_stats.add_(stats)
            if micro%accum and micro!=micro_iters:continue
            norm=torch.nn.utils.clip_grad_norm_([x for x in raw.parameters() if x.requires_grad],1.0)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
            if world>1:dist.all_reduce(window_stats)
            logged=training_metrics(window_stats,config.residual_codec_loss_weight)
            window_stats.zero_()
            loss_history.append(logged['train/loss'])
            logged.update({'train/loss_ma100':sum(loss_history)/len(loss_history),
                           'train/grad_norm':float(norm),'train/learning_rate':optimizer.param_groups[0]['lr'],
                           'train/phase_progress':step/total_steps,'train/data_wait_seconds':data_wait})
            train_logged=dict(logged)
            if rank==0:
                print('PRETRAIN_STEP',args.phase,step,'/',total_steps,'loss',round(logged['train/loss'],5),'grad',round(float(norm),3),flush=True)
                atomic_json(output/'train-latest.json',{'step':step,**logged})
                if swanlab:swanlab.log(logged,step=step)
            if benchmarking:
                if device.type=='cuda':torch.cuda.synchronize(device)
                instant=time.perf_counter()
                if step==args.benchmark_warmup:
                    benchmark_started=instant;benchmark_last=instant
                elif step>args.benchmark_warmup:
                    if benchmark_last is None:benchmark_started=benchmark_last=instant
                    else:benchmark_times.append(instant-benchmark_last);benchmark_waits.append(data_wait);benchmark_last=instant
                data_wait=0.0
                if step>=args.benchmark_steps:break
                continue
            data_wait=0.0
            stopping=args.max_steps>0 and step>=args.max_steps
            checking=step%phase['eval_interval']==0 or micro==micro_iters or stopping
            if checking:
                if world>1:dist.barrier()
                score,metrics=evaluate() if rank==0 or distributed_eval else (0,{})
                values=torch.tensor([score],device=device,dtype=torch.float64)
                if world>1:dist.broadcast(values,src=0)
                score=float(values[0]);improved=score<best
                if improved:best=score;bad=0
                else:bad+=1
                if rank==0:
                    last_eval={'step':step,'objective':score,'metrics':metrics}
                    atomic_json(output/'validation.json',last_eval)
                    print('PRETRAIN_DEV',args.phase,step,score,flush=True)
                    if swanlab:
                        logged={'dev/objective':score,'dev/best_objective':best}
                        for source,values in metrics.items():
                            for key,value in values.items():
                                if isinstance(value,(int,float)):logged[f'dev/{source}/{key}']=value
                        swanlab.log(logged,step=step)
            complete=micro==micro_iters
            if checking or step%phase['save_interval']==0:save('last',micro,complete=complete)
            if stopping or complete:break
    if benchmarking:
        measured=torch.tensor([sum(benchmark_times),torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(measured,op=dist.ReduceOp.MAX)
        probe={}
        for name,param in raw.named_parameters():
            if not param.requires_grad:continue
            probe[name]=param.detach()[control_ids].cpu() if id(param) in control_keys else param.detach().cpu()
        output.mkdir(parents=True,exist_ok=True)
        torch.save(probe,output/f'parameters-rank-{rank}.pt')
        atomic_json(output/f'benchmark-rank-{rank}.json',{'phase':args.phase,'steps':step,'warmup':args.benchmark_warmup,
            'measured_updates':len(benchmark_times),'seconds':float(measured[0]),
            'seconds_per_update':float(measured[0])/max(len(benchmark_times),1),
            'samples_per_second':batch_size*world*accum*len(benchmark_times)/max(float(measured[0]),1e-9),
            'peak_memory_bytes':int(measured[1]),'step_seconds':benchmark_times,'ddp_no_sync':no_sync,
            'batch_size':batch_size,'accumulation_steps':accum,'world_size':world,
            'last_loss':float(loss)*divisor,'last_grad_norm':float(norm),
            'num_workers':workers,'prefetch_factor':prefetch if workers else None,'memory_cache_mib_per_process':memory_cache,
            'data_wait_seconds':benchmark_waits,'mean_data_wait_seconds':sum(benchmark_waits)/max(len(benchmark_waits),1)})
        if rank==0:print('BENCHMARK_COMPLETE',output,flush=True)
    if args.smoke:
        after=backbone_digest(raw,control_ids)
        if after!=frozen_before:raise RuntimeError('Frozen backbone/ordinary vocabulary rows changed')
        updated=any(not torch.equal(before,p.detach().flatten()[:256].cpu()) for p,before in changed_probe)
        if step>initial_step and not updated:raise RuntimeError('Alignment parameters did not update')
        atomic_json(output/f'smoke-rank-{rank}.json',{'frozen_backbone_exact':True,'updates':step,'phase':args.phase,
                    'control_rows':control_ids,'world_size':world,'resume':bool(saved),'alignment_updated':updated,
                    'last_train_metrics':train_logged,'loss_history_length':len(loss_history),
                    'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0})
        if rank==0:print('PRETRAIN_SMOKE_PASS',args.phase,'updates',step,flush=True)
    if swanlab:swanlab.finish()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
