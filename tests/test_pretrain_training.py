import unittest
from types import SimpleNamespace
import torch
import json
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import load_omni_tokenizer
from model.model_omni import MiniQwenOmni, OmniConfig
from trainer.train_pretrain import tune_controls, speaker_only, condition_ablation, reduce_validation_sums, control_row_comm_hook
from trainer.training_common import build_talker_attention_mask
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('pretrain_voice',Path(__file__).resolve().parents[1]/'data_pipeline/prepare_pretrain_voice.py')
voice=importlib.util.module_from_spec(spec);spec.loader.exec_module(voice)
choose_pools,choose_reference=voice.choose_pools,voice.choose_reference

class PretrainTrainingTest(unittest.TestCase):
    def test_control_hook_reduces_only_allowed_rows(self):
        from unittest.mock import patch
        embedding=torch.nn.Parameter(torch.zeros(10,4));other=torch.nn.Parameter(torch.zeros(3))
        storage=torch.zeros(43);gradient=storage[:40].view(10,4);gradient[2]=1;gradient[5]=3
        storage[40:]=torch.tensor([2.,4.,6.])
        bucket=SimpleNamespace(parameters=lambda:[embedding,other],
            gradients=lambda:[gradient,storage[40:]],buffer=lambda:storage)
        communicated=[]
        def remote_sum(tensor):
            communicated.append(tensor.numel());tensor.mul_(2)
        with patch('trainer.train_pretrain.dist.all_reduce',side_effect=remote_sum):
            result=control_row_comm_hook((embedding,[2,5],2),bucket).wait()
        self.assertIs(result,storage)
        self.assertEqual(communicated,[11])
        self.assertTrue(torch.equal(gradient[2],torch.ones(4)))
        self.assertTrue(torch.equal(gradient[5],torch.ones(4)*3))
        self.assertEqual(int(gradient[[0,1,3,4,6,7,8,9]].count_nonzero()),0)
        self.assertTrue(torch.equal(storage[40:],torch.tensor([2.,4.,6.])))

    def test_distributed_validation_reduces_sums_before_division(self):
        from unittest.mock import patch
        sums={'text_nll_sum':6.0,'text_tokens':2,
                                    'audio_nll_sum':[1.0,2.0],'audio_tokens':[3,4]}
        def add_remote(tensor):
            tensor.add_(torch.tensor([4.,8.,3.,4.,5.,6.],dtype=tensor.dtype))
        with patch('trainer.train_pretrain.dist.is_initialized',return_value=True), \
             patch('trainer.train_pretrain.dist.all_reduce',side_effect=add_remote):
            result=reduce_validation_sums(sums,'cpu')
        self.assertEqual(result['text_nll_sum']/result['text_tokens'],1.)
        self.assertEqual(result['audio_nll_sum'],[4.,6.])
        self.assertEqual(result['audio_tokens'],[8.,10.])
        self.assertEqual(sums['text_tokens'],2)

    def test_legacy_unconditioned_targets_remain_complete(self):
        tok=load_omni_tokenizer(Path(__file__).resolve().parents[1]/'model/Qwen3-0.6B')
        row={'sample_id':'legacy-unconditioned','conversations':json.dumps([
            {'role':'user','content':'Read hello'}, {'role':'assistant','content':'hello'}]),
            'answer_audios':[list(range(8))*5], 'spk_emb':[], 'ref_audios':[]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unconditioned.parquet';pq.write_table(pa.Table.from_pylist([row]),path)
            def reader(mode):
                return OmniDataset(str(path),tok,max_length=256,training=True,augment=False,
                    scheduled_sampling=0,strict_length=True,condition_prefix_length=80,
                    voice_conditioning=mode,use_talker_ref_boundaries=True)
            for mode in ['available_mixed','available_both']:
                ids,mask,labels,audio,*_=reader(mode)[0]
                self.assertEqual(int(audio[0].ne(-100).sum()),6)
                self.assertFalse(ids[0].eq(2051).any())
                self.assertFalse(ids[0].eq(2052).any())
                warm=speaker_only(ids.unsqueeze(0),available=torch.tensor([False]))
                self.assertTrue(torch.equal(warm[0],ids))
            with self.assertRaisesRegex(ValueError,'condition is unavailable'):
                reader('mixed')[0]

    def test_audio_response_never_exposes_question_transcript(self):
        from unittest.mock import patch
        ds=object.__new__(OmniDataset)
        ds.tokenizer=SimpleNamespace(apply_chat_template=lambda messages,**kwargs:json.dumps(messages))
        ds.audio_token='<|audio_pad|>';ds.training=True;ds.augment_enabled=False
        messages=[{'role':'user','content':'SECRET QUESTION TRANSCRIPT'},
                  {'role':'assistant','content':'answer'}]
        for training in (False,True):
            ds.training=training
            for draw in (0.1,0.5,0.7,0.9):
                with patch('dataset.omni_dataset.random.random',return_value=draw):
                    prompt=ds.create_chat_prompt(messages,3,task_type='audio_response')
                self.assertNotIn('SECRET QUESTION TRANSCRIPT',prompt)
                self.assertEqual(prompt.count('<|audio_pad|>'),3)

    def test_strict_voice_reader_preserves_reference_and_target(self):
        tok=load_omni_tokenizer(Path(__file__).resolve().parents[1]/'model/Qwen3-0.6B')
        row={'sample_id':'voice-test','conversations':json.dumps([{'role':'user','content':'Read: hello'},{'role':'assistant','content':'hello'}]),'answer_audios':[list(range(8))*5],'spk_emb':[.1]*192,'ref_audios':list(range(8))*50,'voice_self_fallback':False}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'voice.parquet';pq.write_table(pa.Table.from_pylist([row]),path)
            def reader(mode='both',length=256):
                return OmniDataset(str(path),tok,max_length=length,training=True,augment=False,scheduled_sampling=0,strict_length=True,condition_prefix_length=80,voice_conditioning=mode,use_modality_boundaries=True,use_talker_ref_boundaries=True)
            both=reader();x,mask,labels,audio,*_=both[0]
            self.assertFalse(mask[:80].any());self.assertEqual(int(audio[0].ne(-100).sum()),6)
            self.assertEqual(int(x[0].lt(2048).sum()),55)
            self.assertTrue(torch.equal(x,both[0][0]))
            for mode,has_spk,has_ref in [('speaker',True,False),('reference',False,True),('mixed',None,None)]:
                y=reader(mode)[0][0]
                self.assertTrue(torch.equal(audio,reader(mode)[0][3]))
                if has_spk is not None:self.assertEqual(bool(y[0].eq(2051).any()),has_spk)
                if has_ref is not None:self.assertEqual(bool(y[0].eq(2052).any()),has_ref)
            with self.assertRaises(ValueError):reader(length=96)[0]
            row['ref_audios'].append(1)
            path=Path(tmp)/'incomplete.parquet';pq.write_table(pa.Table.from_pylist([row]),path)
            with self.assertRaisesRegex(ValueError,'incomplete reference'):reader()[0]

    def test_control_rows_no_decay_or_other_vocabulary_updates(self):
        m=SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(10,4)),config=SimpleNamespace(audio_ids=[2],audio_start_ids=[3],audio_end_ids=[4]))
        params,ids=tune_controls(m,'audio');old=params[0].detach().clone()
        opt=torch.optim.AdamW(params,lr=.1,weight_decay=0)
        params[0].sum().backward();opt.step()
        self.assertTrue(torch.equal(old[[0,1,5,6,7,8,9]],params[0][[0,1,5,6,7,8,9]]))
        self.assertFalse(torch.equal(old[ids],params[0][ids]))

    def test_warmup_ablation_and_attention(self):
        x=torch.full((1,9,12),2049,dtype=torch.long);x[:,8]=0
        x[0,0,1]=2051;x[0,:8,3:6]=7;x[0,0,8]=2048;x[0,:8,9:]=4
        warm=speaker_only(x)
        self.assertEqual(int(warm[0,0,7]),2051)
        self.assertTrue(torch.equal(warm[:,:,8:],x[:,:,8:]))
        ref=condition_ablation(x,'reference');spk=condition_ablation(x,'speaker')
        self.assertEqual(int(ref[0,0,1]),2049);self.assertEqual(int(spk[0,0,1]),2051)
        mask=build_talker_attention_mask(torch.zeros(1,12,dtype=torch.long),torch.full((1,8,12),-100),x)
        self.assertTrue(mask[0,1]);self.assertTrue(mask[0,3]);self.assertFalse(mask[0,0])

    def test_reference_different_text_and_fallback(self):
        rows=[dict(source='s',split='train',speaker_id='a',sample_id=str(i),duration=4,text_normalized=str(i)) for i in range(3)]
        pool=choose_pools(rows)[('s','train','a')]
        for target in rows:
            ref,fallback=choose_reference(target,pool)
            self.assertFalse(fallback);self.assertNotEqual(ref['text_normalized'],target['text_normalized'])
        self.assertTrue(choose_reference(rows[0],rows[:1])[1])
        with self.assertRaises(ValueError):choose_pools(rows+[dict(rows[0],split='dev')])

    def test_conditioned_generation_cache_and_no_cache(self):
        config=OmniConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,num_talker_hidden_layers=1,talker_hidden_size=32,code_predictor_hidden_size=32,code_predictor_num_layers=1,code_predictor_num_attention_heads=4,code_predictor_num_key_value_heads=2,code_predictor_intermediate_size=64,audio_vocab_size=2112,audio_codebook_size=2048,spk_emb_size=16,accept_hidden_layer=1,use_mrope=True,mrope_section=[2,1,1],audio_head_type='main_codec_predictor',eos_token_id=2,pad_token_id=0,think_end_ids=[],talker_condition_prefix_length=80,use_talker_ref_boundaries=True)
        model=MiniQwenOmni(config,audio_encoder_path=None,vision_model_path=None).eval()
        model._sample_from_logits=lambda logits,*args,**kwargs:int(logits.argmax())
        with torch.no_grad():model.talker.lm_head.weight.zero_()
        results=[]
        for cache in (True,False):
            with torch.no_grad():out=list(model.stream_generate(torch.tensor([[1,3]]),eos_token_id=2,max_new_tokens=3,temperature=1,top_p=1,top_k=1,rp=1,use_cache=cache,return_audio_codes=True,spk_emb=torch.zeros(1,16),ref_codes=torch.zeros(1,8,50,dtype=torch.long),newline_token_id=4,pad_token_id=0))
            results.append([(tokens.tolist() if tokens is not None else None,frame) for tokens,frame in out])
        self.assertEqual(results[0],results[1]);self.assertTrue(any(frame is not None for _,frame in results[0]))
if __name__=='__main__':unittest.main()
