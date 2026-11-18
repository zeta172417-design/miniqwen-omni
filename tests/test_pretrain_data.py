"""Checks for the pretrain source converters and explicit ASR instruction path."""
import io
import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path
import tarfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

spec = importlib.util.spec_from_file_location('prepare_pretrain_data', Path(__file__).resolve().parents[1] / 'data_pipeline/prepare_pretrain_data.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
catalog_spec = importlib.util.spec_from_file_location('build_pretrain_catalog', Path(__file__).resolve().parents[1] / 'data_pipeline/build_pretrain_catalog.py')
catalog_module = importlib.util.module_from_spec(catalog_spec)
catalog_spec.loader.exec_module(catalog_module)


class PretrainDataTest(unittest.TestCase):
    def test_official_clean_view_preserves_full_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, destination = base / 'full.parquet', base / 'clean.parquet'
            rows = [{'sample_id': 'a', 'duration': 1.0, 'codec_frames': 10},
                    {'sample_id': 'b', 'duration': 2.0, 'codec_frames': 20},
                    {'sample_id': 'c', 'duration': 3.0, 'codec_frames': 30}]
            pq.write_table(pa.Table.from_pylist(rows), source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            result = catalog_module.clean_file(source, destination, ['b'])
            self.assertEqual(result['rows'], 2)
            self.assertEqual(result['seconds'], 4.0)
            self.assertEqual(result['codec_frames'], 40)
            self.assertEqual(pq.read_table(destination)['sample_id'].to_pylist(), ['a', 'c'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_download_retries_probe_resumes_partial_and_verifies_sha256(self):
        payload = b'abcdefg'
        ranges = []
        class Response:
            status_code = 206
            def __init__(self, start, end):
                self.start, self.end = start, end
                self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def iter_content(self, size): yield payload[self.start:self.end + 1]
        def get(url, headers, **kwargs):
            value = headers['Range']
            ranges.append(value)
            if len(ranges) == 1:
                raise prep.requests.ReadTimeout('transient metadata probe timeout')
            start, end = map(int, value.removeprefix('bytes=').split('-'))
            return Response(start, end)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parts = base / 'raw/example/file.parts'
            parts.mkdir(parents=True)
            (parts / '000000.tmp').write_bytes(payload[:3])
            with patch.object(prep, 'DATA', base), patch.object(prep.requests, 'get', side_effect=get), patch.object(prep.time, 'sleep'):
                prep.download({'path': 'file', 'url': 'https://example.test/file', 'size': len(payload),
                               'sha256': hashlib.sha256(payload).hexdigest()}, 'example', 2)
            self.assertIn('bytes=3-6', ranges)
            self.assertEqual((base / 'raw/example/file').read_bytes(), payload)
            self.assertTrue(json.loads((base / 'raw/example/file.download.json').read_text())['upstream_sha256_verified'])

    def test_tts_aishell3_wav_suffix_and_pinyin_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / 'media/aishell3/train/wav/SSB0005'
            folder.mkdir(parents=True)
            sf.write(folder / 'SSB00050001.wav', np.zeros(2400), 24000)
            (folder.parent.parent / 'content.txt').write_text('SSB00050001.wav\t你 ni3 好 hao3\n')
            with patch.object(prep, 'DATA', base), patch.object(prep, 'ROOT', base):
                prep.write_tts('aishell3')
            row = json.loads((base / 'processed/aishell3/segments.jsonl').read_text())
            self.assertEqual(row['text_normalized'], '你好')
            self.assertIn('ni3', row['transcript_original'])
            self.assertEqual(row['speaker_id'], 'SSB0005')

    def test_llava_identical_images_share_split_and_captions_are_retained(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base / 'raw/llava'
            raw.mkdir(parents=True)
            data = io.BytesIO()
            Image.new('RGB', (8, 8)).save(data, format='PNG')
            records = []
            with zipfile.ZipFile(raw / 'images.zip', 'w') as z:
                for i in range(2):
                    z.writestr(f'{i}.png', data.getvalue())
                    records.append({'id': i, 'image': f'{i}.png', 'conversations': [
                        {'from': 'human', 'value': '<image>Describe this.'}, {'from': 'gpt', 'value': f'Caption {i}'}]})
            (raw / 'blip_laion_cc_sbu_558k.json').write_text(json.dumps(records))
            with patch.object(prep, 'DATA', base):
                prep.write_llava()
            counts = [pq.read_table(base / f'processed/llava/{s}.parquet').num_rows for s in ('train', 'dev')]
            self.assertEqual(sorted(counts), [0, 2])

    def test_asr_conversion_keeps_original_and_speaker_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audio = io.BytesIO()
            sf.write(audio, np.zeros(1600, dtype=np.float32), 16000, format='FLAC')
            for folder, sid, speaker in [('train.100', '1-2-3', 1), ('validation', '4-5-6', 4)]:
                path = base / 'raw/librispeech/clean' / folder
                path.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist([dict(id=sid, speaker_id=speaker, text='HELLO WORLD',
                                                          audio={'bytes': audio.getvalue(), 'path': 'ignored.flac'})]), path / '0000.parquet')
            with patch.object(prep, 'DATA', base):
                prep.write_asr('librispeech')
            row = pq.read_table(base / 'processed/librispeech/train.parquet').to_pylist()[0]
            self.assertEqual(row['text_original'], 'HELLO WORLD')
            self.assertEqual(row['text_target'], 'HELLO WORLD')
            self.assertEqual(row['task_type'], 'asr')
            self.assertAlmostEqual(row['duration'], 0.1)
            self.assertNotIn('HELLO WORLD', __import__('json').loads(row['conversations'])[0]['content'])

    def test_unsafe_archives_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name, kind in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
                archive = base / 'unsafe.tar.gz'
                with tarfile.open(archive, 'w:gz') as t:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    t.addfile(member)
                destination = base / 'output'
                destination.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    prep.extract_archive(archive, destination)
                self.assertFalse((base / 'escape').exists())

    def test_streaming_archive_extraction_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / 'safe.tar.gz'
            payload = b'public audio sample'
            with tarfile.open(archive, 'w:gz') as t:
                member = tarfile.TarInfo('train/speaker/sample.wav')
                member.size = len(payload)
                t.addfile(member, io.BytesIO(payload))
            destination = base / 'output'
            destination.mkdir()
            prep.extract_archive(archive, destination)
            self.assertEqual((destination / 'train/speaker/sample.wav').read_bytes(), payload)

    def test_explicit_asr_always_preserves_instruction_and_waveform(self):
        from dataset.omni_dataset import OmniDataset
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return str(messages)
        ds = object.__new__(OmniDataset)
        ds.audio_token = '<|audio_pad|>'
        ds.tokenizer = Tokenizer()
        conversations = [{'role': 'user', 'content': 'Transcribe verbatim.'},
                         {'role': 'assistant', 'content': 'HELLO WORLD'}]
        for training in (False, True):
            ds.training = training
            for random_value in (0.1, 0.5, 0.7, 0.9):
                with patch('dataset.omni_dataset.random.random', return_value=random_value):
                    prompt = ds.create_chat_prompt(conversations, 3, task_type='asr')
                self.assertIn('Transcribe verbatim.', prompt)
                self.assertEqual(prompt.count('<|audio_pad|>'), 3)


if __name__ == '__main__':
    unittest.main()
