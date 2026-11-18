"""Final SFT Parquet reader: preserve image/audio instructions and load source media on demand."""
import json
from pathlib import Path
from dataset.omni_dataset import OmniDataset

ROOT=Path(__file__).resolve().parents[1]
class MediaRows:
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    @property
    def row_group_ranges(self):return self.rows.row_group_ranges
    def __getitem__(self,index):
        row=dict(self.rows[index]);messages=json.loads(row['conversations'])
        row['image_bytes']=[(ROOT/path).read_bytes() for path in row.get('image_paths') or []]
        paths=row.get('question_audio_paths') or []
        if paths:
            # Prefer embedded audio when both representations are present.
            # Some externally rebuilt shards retain the old path column for
            # provenance while storing the validated waveform inline.
            if row.get('question_audios'):
                row['question_audio_paths'] = []
                return row
            users=sum(m['role']=='user' for m in messages)
            if len(paths)!=1:raise ValueError('Expected one final question audio path')
            row['question_audios']=[b'']*(users-1)+[(ROOT/paths[0]).read_bytes()]
        return row
class SFTDataset(OmniDataset):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('augment',False);kwargs.setdefault('scheduled_sampling',0)
        kwargs.setdefault('strict_length',True);kwargs.setdefault('condition_prefix_length',160)
        kwargs.setdefault('voice_conditioning','available_both')
        kwargs.setdefault('use_modality_boundaries',True);kwargs.setdefault('use_talker_ref_boundaries',True)
        super().__init__(*args,**kwargs);self.dataset=MediaRows(self.dataset)
    def create_chat_prompt(self,conversations,audio_features_length=0,task_type=None):
        # Prepared user content is task instruction plus any image markers, not the spoken-Q transcript.
        # ASR formatting appends waveform markers while retaining those instructions in all modalities.
        return super().create_chat_prompt(conversations,audio_features_length,task_type='asr' if audio_features_length else task_type)
    def __getitem__(self,index):
        if isinstance(index, tuple):
            index, self.condition_epoch = index
        original=self.voice_conditioning
        try:
            row=self.dataset.rows[index]
            if row['required_length']>self.max_length:raise ValueError('SFT row exceeds requested reader budget')
            self.voice_conditioning='task_mixed' if self.training else 'available_both'
            return super().__getitem__(index)
        finally:self.voice_conditioning=original


def is_prepared_sft(data_path):
    """Select the reader from the first Parquet schema without scanning its rows."""
    import pyarrow.parquet as pq

    first_path = str(data_path).split(',')[0].strip()
    fields = set(pq.ParquetFile(first_path).schema_arrow.names)
    return {'length_tier', 'required_length', 'image_paths'}.issubset(fields)


def load_manifest_paths(manifest_path, tier='all', split='train'):
    """Resolve accepted SFT shards for the ordinary trainer entry point."""
    manifest = json.loads(Path(manifest_path).read_text())
    tiers = list(manifest['tiers']) if tier == 'all' else [tier]
    paths = []
    maximum = 0
    for name in tiers:
        specification = manifest['tiers'][name]
        paths.extend(str(ROOT / path) for path in specification['data_files'].get(split, []))
        maximum = max(maximum, specification['max_length'])
    if not paths:
        raise ValueError(f'No {split} SFT files for tier {tier}')
    return ','.join(paths), maximum
