"""End-to-end label and voice masking contracts for rebuilt Parquet."""
import json
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dataset.omni_dataset import OmniDataset
from dataset.voice_conditioning import choose_voice_mode
from trainer.trainer_utils import load_omni_tokenizer
from trainer.pretrain_sampler import EpochBatchSampler

def test_all_turn_text_only_last_turn_audio_and_condition_dropout(tmp_path):
    tok=load_omni_tokenizer('model/Qwen3-0.6B')
    conversation=[{'role':'user','content':'Read hello'}, {'role':'assistant','content':'hello'}, {'role':'user','content':'Read bye'}, {'role':'assistant','content':'bye'}]
    rows=[]
    for i in range(160):
        rows.append(dict(sample_id=str(i),source='libritts_r',task_type='tts',conversations=json.dumps(conversation),answer_audios=[[7]*32,[9]*40],ref_audios=[3]*400,spk_emb=[.1]*192,supervision_policy='all_text_last_audio'))
    path=tmp_path/'data.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
    ds=OmniDataset(str(path),tok,max_length=320,training=True,augment=False,scheduled_sampling=0,strict_length=True,voice_conditioning='task_mixed',condition_prefix_length=160,use_talker_ref_boundaries=True)
    found={}
    for i,row in enumerate(rows):found.setdefault(choose_voice_mode(row,ds.condition_seed,0)[1],i)
    assert set(found)=={'none','speaker','reference','both'}
    for mode,i in found.items():
        ids,mask,text,audio,*_,spk=ds[(i,0)]
        assert bool(ids[0].eq(ds.audio_spk_token).any())==(mode in {'both','speaker'})
        assert bool(ids[0].eq(ds.audio_ref_start_token).any())==(mode in {'both','reference'})
        assert bool(spk.ne(0).any())==(mode in {'both','speaker'})
        assert int(audio[0].ne(-100).sum())==6
        assert not audio.eq(7).any()
        decoded=tok.decode(text[text.ne(-100)].tolist())
        assert 'hello' in decoded and 'bye' in decoded
        assert 'Read hello' not in decoded
        again=ds[(i,0)]
        assert torch.equal(ids,again[0])
    row=dict(rows[0],source='old_t2a',spk_emb=[],ref_audios=[])
    assert choose_voice_mode(row,42)==('legacy_unconditioned_t2a','none')

def test_epoch_keys_survive_loader_prefetch():
    class Sampler:
        def __len__(self):return 3
        def set_epoch(self,e):self.epoch=e
        def __iter__(self):return iter(range(3))
    assert list(EpochBatchSampler(Sampler(),2,2,condition_epochs=True))==[[(0,0),(1,0)],[(2,0)],[(0,1),(1,1)],[(2,1)]]

def test_missing_conditions_never_drop_samples_and_mixture():
    counts=Counter(choose_voice_mode({'source':'aishell3','sample_id':str(i),'spk_emb':[1],'ref_audios':[1]},42)[1] for i in range(10000))
    for mode,target in [('both',.6),('speaker',.15),('reference',.1),('none',.15)]:assert abs(counts[mode]/10000-target)<.02
    assert all(choose_voice_mode({'source':'aishell3','sample_id':str(i)},42)[1]=='none' for i in range(100))
