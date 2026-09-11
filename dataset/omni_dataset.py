import torch
import os
import math
import random
import numpy as np
import soundfile as sf
import librosa
import json
import io
import time
import itertools
from PIL import Image
from scipy.signal import resample
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq
import datasets as hf_datasets
import torch.distributed as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是miniqwen-omni，一个小巧但有用的多模态助手。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是miniqwen-omni，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are miniqwen-omni, a lightweight multimodal assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are miniqwen-omni, a small but useful multimodal model."
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content):
    return prompt_content


class OmniDataset(Dataset):
    def __init__(self, data_path, tokenizer, audio_processor=None, vision_processor=None,
                 max_length=1200, audio_special_token='<|audio_pad|>', image_special_token='<|image_pad|>',
                 audio_start_special_token='<|audio_start|>', audio_end_special_token='<|audio_end|>',
                 vision_start_special_token='<|vision_start|>', vision_end_special_token='<|vision_end|>',
                 audio_bos_token=2048,
                 audio_stop_token=2050,  # <|audio_stop|>
                 audio_pad_token=2049,  # <|audio_pad|>
                 audio_spk_token=2051,  # <|audio_spk|>
                 audio_ref_start_token=2052,
                 audio_ref_end_token=2053,
                 audio_vocab_size=2112,  # 2048 mimi codes + 64 special tokens
                 scheduled_sampling=0.05,
                 image_token_len=64,
                 max_images=4,
                 audio_head_type='main_codec_predictor',
                 use_modality_boundaries=False,
                 use_talker_ref_boundaries=False,
                 training=True):
        super().__init__()
        paths = [p.strip() for p in data_path.split(',')]
        cache_dir = os.environ.get('MINIQWEN_DATASET_CACHE', os.path.join(os.path.dirname(paths[0]), '.miniqwen_cache'))
        parquet_batch_size = int(os.environ.get('MINIQWEN_PARQUET_BATCH_SIZE', '4096'))
        if parquet_batch_size <= 0:
            raise ValueError('MINIQWEN_PARQUET_BATCH_SIZE must be positive')
        hf_datasets.disable_progress_bars()

        def load_mapped_dataset():
            # Hugging Face's packaged parquet reader uses a Dataset Scanner.
            # For the full A2A file its single 414k-row group contains a nested
            # binary column above 2 GiB, and the scanner fails before applying
            # its requested batch size. ParquetFile.iter_batches streams the
            # exact same file correctly, so use it to feed the regular HF Arrow
            # cache writer. The returned object remains a memory-mapped Dataset.
            builder = hf_datasets.load_dataset_builder(
                'parquet',
                data_files=paths,
                cache_dir=cache_dir,
                batch_size=parquet_batch_size,
            )

            def generate_tables(files):
                for file_idx, file_path in enumerate(itertools.chain.from_iterable(files)):
                    with open(file_path, 'rb') as parquet_stream:
                        parquet_file = pq.ParquetFile(parquet_stream)
                        for batch_idx, record_batch in enumerate(parquet_file.iter_batches(
                            batch_size=parquet_batch_size,
                            columns=builder.config.columns,
                            use_threads=False,
                        )):
                            table = pa.Table.from_batches([record_batch])
                            yield f'{file_idx}_{batch_idx}', builder._cast_table(table)

            builder._generate_tables = generate_tables
            builder.download_and_prepare()
            return builder.as_dataset(split='train', in_memory=False)

        started = time.time()
        if dist.is_initialized() and dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                print(f'准备数据集缓存: {",".join(paths)} (parquet batch={parquet_batch_size})', flush=True)
                self.dataset = load_mapped_dataset()
            dist.barrier()
            if dist.get_rank() != 0:
                self.dataset = load_mapped_dataset()
        else:
            print(f'准备数据集缓存: {",".join(paths)} (parquet batch={parquet_batch_size})', flush=True)
            self.dataset = load_mapped_dataset()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f'数据集已就绪: {len(self.dataset)} samples, {time.time() - started:.1f}s', flush=True)
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.audio_start_token = audio_start_special_token
        self.audio_end_token = audio_end_special_token
        self.vision_start_token = vision_start_special_token
        self.vision_end_token = vision_end_special_token
        self.use_modality_boundaries = bool(use_modality_boundaries)
        self.use_talker_ref_boundaries = bool(use_talker_ref_boundaries)
        self.image_token_len = image_token_len
        self.max_images = max_images
        image_payload = image_special_token * image_token_len
        self.image_token = (
            vision_start_special_token + image_payload + vision_end_special_token
            if self.use_modality_boundaries else image_payload
        )
        self.audio_stop_token = audio_stop_token
        self.audio_bos_token = audio_bos_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_ref_start_token = audio_ref_start_token
        self.audio_ref_end_token = audio_ref_end_token
        self.audio_vocab_size = audio_vocab_size
        self.audio_head_type = audio_head_type
        if self.audio_head_type not in {'main_codec_predictor', 'legacy_parallel_delay'}:
            raise ValueError(f'unsupported audio_head_type: {self.audio_head_type}')
        self.training = bool(training)
        self.scheduled_sampling_prob = scheduled_sampling if self.training else 0.0
        self.text_vocab_size = len(tokenizer)
        # tokenizer.vocab_size excludes Qwen's added control tokens.  Random
        # scheduled-sampling replacements must never synthesize image/audio or
        # chat markers such as <|image_pad|>.
        self.base_text_vocab_size = tokenizer.vocab_size
        self.base_audio_vocab_size = min(2048, audio_vocab_size)
        safe_ids = tokenizer.encode(' ', add_special_tokens=False)
        self.safe_text_token_id = safe_ids[0]
        image_ids = tokenizer.encode(image_special_token, add_special_tokens=False)
        audio_ids = tokenizer.encode(audio_special_token, add_special_tokens=False)
        if len(image_ids) != 1 or len(audio_ids) != 1:
            raise ValueError("audio/image placeholders must each encode to exactly one token")
        self.image_token_id = image_ids[0]
        self.audio_token_id = audio_ids[0]
        self.audio_start_ids = tokenizer.encode(audio_start_special_token, add_special_tokens=False)
        self.audio_end_ids = tokenizer.encode(audio_end_special_token, add_special_tokens=False)
        self.vision_start_ids = tokenizer.encode(vision_start_special_token, add_special_tokens=False)
        self.vision_end_ids = tokenizer.encode(vision_end_special_token, add_special_tokens=False)
        for name, ids in (
            ('audio_start', self.audio_start_ids), ('audio_end', self.audio_end_ids),
            ('vision_start', self.vision_start_ids), ('vision_end', self.vision_end_ids),
        ):
            if len(ids) != 1:
                raise ValueError(f"{name} placeholder must encode to exactly one token")
        self.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        self.assistant_start_ids = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        self.assistant_end_ids = tokenizer.encode('<|im_end|>', add_special_tokens=False)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def process_audio(audio_path, audio_processor):
        """加载音频并预处理成fbank，返回 (fbank (T,560), valid_len=encoder输出帧数)"""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        inputs = audio_processor(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def augment_wav(self, wav, sr=16000):
        if not self.training:
            return np.asarray(wav, dtype=np.float32)
        # 随机变速(0.7~1.6x)：改变音频时长和音调，覆盖快/慢语速
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.6)
            wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
        # 随机加噪：叠加轻微高斯白噪声，模拟录音环境差异
        if random.random() < 0.3:
            noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
            wav = wav + noise
        # 随机音量：缩放振幅0.8~1.2倍，模拟说话音量变化
        if random.random() < 0.3:
            wav = wav * random.uniform(0.8, 1.2)
        # 随机时间遮蔽：将0.25秒片段置零，模拟短暂静音/丢包
        if random.random() < 0.2 and len(wav) > sr:
            start = random.randint(0, len(wav) - sr // 4)
            wav[start:start + sr // 4] = 0
        # 随机低通滤波：移动平均模糊高频，模拟电话/低质量麦克风
        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            wav = np.convolve(wav, np.ones(k) / k, mode='same').astype(np.float32)
        # 随机混响：合成指数衰减脉冲响应卷积，模拟房间反射/回声
        if random.random() < 0.3:
            ir_len = int(sr * random.uniform(0.05, 0.2))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode='same').astype(np.float32)
        # 随机粉红噪声：1/f噪声模拟房间环境底噪（空调/远处人声）
        if random.random() < 0.2:
            pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
            pink /= np.max(np.abs(pink)) + 1e-6
            wav = wav + pink * random.uniform(0.003, 0.015)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def augment_mel(self, fbank):
        if not self.training:
            return fbank
        # fbank: (T, 560) — SenseVoice LFR 后的特征（时间维在前，频率维在后）
        T, D = fbank.shape
        # SpecAugment频率遮蔽：随机抹掉1~64维，防止对特定频段过拟合
        if random.random() < 0.5:
            f = random.randint(1, 64)
            f0 = random.randint(0, D - f)
            fbank[:, f0:f0 + f] = 0
        # SpecAugment时间遮蔽：随机抹掉1~min(10,T)帧，提升对不完整输入的容错
        if random.random() < 0.5 and T > 1:
            t = random.randint(1, min(10, T))
            t0 = random.randint(0, T - t)
            fbank[t0:t0 + t, :] = 0
        return fbank

    def load_audio_inputs(self, audio_bytes):
        if not audio_bytes: return None, 0
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav = self.augment_wav(wav.astype(np.float32))
        inputs = self.audio_processor(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return self.augment_mel(inputs.input_features.squeeze(0)), valid_len

    def load_image_inputs(self, image_bytes_list):
        if not image_bytes_list or self.vision_processor is None:
            return None, 0
        images = []
        for raw in image_bytes_list[:self.max_images]:
            try:
                images.append(Image.open(io.BytesIO(raw)).convert('RGB'))
            except Exception:
                continue
        if not images:
            return None, 0
        inputs = self.vision_processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'] if hasattr(inputs, 'keys') else inputs.pixel_values
        return pixel_values, len(images)

    @staticmethod
    def retain_image_markers(conversations, count):
        """Keep the first ``count`` image markers and remove every unmatched one."""
        remaining = count
        output = []
        for turn in conversations:
            content = str(turn.get('content', ''))
            pieces = content.split('<image>')
            if len(pieces) > 1:
                rebuilt = pieces[0]
                for piece in pieces[1:]:
                    if remaining > 0:
                        rebuilt += '<image>'
                        remaining -= 1
                    rebuilt += piece
                content = rebuilt
            output.append({**turn, 'content': content})
        return output

    def format_audio_tokens(self, length):
        payload = self.audio_token * length
        if getattr(self, 'use_modality_boundaries', False):
            return self.audio_start_token + payload + self.audio_end_token
        return payload

    def create_chat_prompt(self, conversations, audio_features_length=0):
        conversations = pre_processing_chat(conversations, add_system_ratio=0.2 if self.training else 0.0)
        messages = []
        is_last_user = lambda i: i == max(j for j, t in enumerate(conversations) if t['role'] == 'user')
        for idx, turn in enumerate(conversations):
            role, content = turn['role'], turn['content']
            if role == 'user' and is_last_user(idx) and audio_features_length > 0:
                ap = self.format_audio_tokens(audio_features_length)
                if not self.training:
                    # Do not leak an audio question's transcript through the
                    # text channel during evaluation.
                    content = ap
                else:
                    r = random.random()
                    if r < 0.4: content = ap
                    elif r < 0.6: content = content
                    elif r < 0.8: content = ap + '\n\n' + content
                    else: content = content + '\n\n' + ap
            if self.training and content.count('<image>') == 1:
                r = random.random()
                if r < 0.2: content = '<image>\n' + content.replace('<image>', '').strip()
                elif r < 0.4: content = '<image>\n\n' + content.replace('<image>', '').strip()
                elif r < 0.6: content = content.replace('<image>', '').strip() + '\n' + '<image>'
                else: content = content.replace('<image>', '').strip() + '\n\n' + '<image>'
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return post_processing_chat(prompt)

    
    def generate_text_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        ranges = []
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.assistant_start_ids)] == self.assistant_start_ids:
                start = i + len(self.assistant_start_ids)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.assistant_end_ids)] == self.assistant_end_ids:
                        break
                    end += 1
                ranges.append((start, end))
                for j in range(start, min(end + len(self.assistant_end_ids), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.assistant_end_ids) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels, ranges

    def truncate_for_training(self, input_ids):
        """Truncate context while always retaining the final assistant target.

        Plain right truncation can remove ``<|im_start|>assistant`` for long
        prompts, leaving an all--100 label tensor.  Keep the final assistant
        span first, then fill the remaining context budget from the right.  A
        complete audio/image marker run is retained when it fits; partial runs
        are never retained because they would mismatch encoder features.
        """
        if len(input_ids) <= self.max_length:
            return input_ids

        starts = [
            i for i in range(len(input_ids))
            if input_ids[i:i + len(self.assistant_start_ids)] == self.assistant_start_ids
        ]
        if not starts:
            return input_ids[:self.max_length]

        assistant_marker = starts[-1]
        target = input_ids[assistant_marker:]
        if len(target) >= self.max_length:
            return target[:self.max_length]

        context = input_ids[:assistant_marker]
        context_budget = self.max_length - len(target)
        modality_ids = {
            self.audio_token_id, self.image_token_id,
            *getattr(self, 'audio_start_ids', []), *getattr(self, 'audio_end_ids', []),
            *getattr(self, 'vision_start_ids', []), *getattr(self, 'vision_end_ids', []),
        }
        mandatory = [i for i, token_id in enumerate(context) if token_id in modality_ids]

        if len(mandatory) <= context_budget:
            selected = set(mandatory)
            for i in range(len(context) - 1, -1, -1):
                if len(selected) >= context_budget:
                    break
                selected.add(i)
            kept_context = [token_id for i, token_id in enumerate(context) if i in selected]
        else:
            # The modality itself is longer than the available context.  Drop
            # all of its markers instead of keeping a mismatched partial run.
            non_modality = [token_id for token_id in context if token_id not in modality_ids]
            kept_context = non_modality[-context_budget:] if context_budget else []
        return kept_context + target
    
    def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
        """Scheduled Sampling: 用随机值替代部分GT，让模型学会从错误历史中恢复"""
        if self.scheduled_sampling_prob <= 0:
            return input_ids
        if getattr(self, 'audio_head_type', 'legacy_parallel_delay') == 'main_codec_predictor':
            # Corrupt only actual previous-frame feedback. BOS and every
            # Talker control token must remain intact.
            audio_mask = (
                (input_ids[:8] >= 0) & (input_ids[:8] < self.base_audio_vocab_size)
            ).all(dim=0)
            audio_mask &= (audio_labels != -100).any(dim=0)
            audio_mask &= torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob
        else:
            audio_mask = (audio_labels != -100).any(dim=0)
            audio_mask &= torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob
        for i in range(8):
            random_codes = torch.randint(0, self.base_audio_vocab_size, input_ids[i].shape)
            input_ids[i] = torch.where(audio_mask, random_codes, input_ids[i])
        # 保护 image token 的连续性
        protected = torch.zeros_like(input_ids[8], dtype=torch.bool)
        protected_ids = {
            self.image_token_id,
            *getattr(self, 'audio_start_ids', []), *getattr(self, 'audio_end_ids', []),
            *getattr(self, 'vision_start_ids', []), *getattr(self, 'vision_end_ids', []),
        }
        audio_token_id = getattr(self, 'audio_token_id', None)
        if audio_token_id is not None:
            protected_ids.add(audio_token_id)
        for token_id in protected_ids:
            protected |= input_ids[8] == token_id
        text_mask = (text_labels != -100) & ~protected & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        random_tokens = torch.randint(0, self.base_text_vocab_size, input_ids[8].shape)
        input_ids[8] = torch.where(text_mask, random_tokens, input_ids[8])
        return input_ids

    def sanitize_image_markers(self, input_ids, image_count=0, has_real_image=None):
        """Only real images may own complete, bounded image-token runs."""
        # Keep compatibility with callers from the previous scalar-image
        # dataset API while validating the exact number of image blocks now.
        if has_real_image is not None:
            image_count = 1 if has_real_image else 0
        marker_count = input_ids.count(self.image_token_id)
        run_lengths = []
        i = 0
        while i < len(input_ids):
            if input_ids[i] != self.image_token_id:
                i += 1
                continue
            start = i
            while i < len(input_ids) and input_ids[i] == self.image_token_id:
                i += 1
            run_lengths.append(i - start)
        boundaries_valid = True
        vision_start_ids = getattr(self, 'vision_start_ids', [])
        vision_end_ids = getattr(self, 'vision_end_ids', [])
        if getattr(self, 'use_modality_boundaries', False):
            boundaries_valid = (
                len(vision_start_ids) == 1
                and len(vision_end_ids) == 1
                and input_ids.count(vision_start_ids[0]) == image_count
                and input_ids.count(vision_end_ids[0]) == image_count
            )
        valid = (
            image_count > 0
            and marker_count == image_count * self.image_token_len
            and run_lengths == [self.image_token_len] * image_count
            and boundaries_valid
        )
        has_image_controls = marker_count or any(
            token_id in input_ids for token_id in (*vision_start_ids, *vision_end_ids)
        )
        if has_image_controls and not valid:
            image_related = {
                self.image_token_id, *vision_start_ids, *vision_end_ids,
            }
            input_ids = [
                self.safe_text_token_id if token_id in image_related else token_id
                for token_id in input_ids
            ]
        return input_ids, valid

    def __getitem__(self, index: int):
        row = self.dataset[index]
        conversations = json.loads(row['conversations'])
        question_audios = row.get('question_audios') or []
        answer_audios = row.get('answer_audios') or []
        image_bytes = row.get('image_bytes') or []
        if image_bytes and not isinstance(image_bytes, list): image_bytes = [image_bytes]
        ref_audios = row.get('ref_audios') or []
        spk_emb_raw = row.get('spk_emb') or []
        
        # 随机截断到某一轮（每轮=user+assistant）
        asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
        if self.training and len(asst_indices) > 1:
            rand_idx = random.randint(0, len(asst_indices) - 1)
            # 从随机轮次开始，向前回退直到长度安全
            for i in range(rand_idx, -1, -1):
                conversations = conversations[:asst_indices[i] + 1]
                test_prompt = self.create_chat_prompt(conversations, 0)
                if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
                    break
        
        # Images and <image> markers are paired in source order. Keep the first
        # configured pairs so legacy scalar-binary and list<binary> rows share
        # the same path.
        pixel_values = None
        marker_count = sum(str(turn.get('content', '')).count('<image>') for turn in conversations)
        requested_images = min(len(image_bytes), marker_count, self.max_images)
        conversations = self.retain_image_markers(conversations, requested_images)
        image_count = 0
        if requested_images and self.vision_processor:
            pixel_values, image_count = self.load_image_inputs(image_bytes[:requested_images])
            if image_count != requested_images:
                conversations = self.retain_image_markers(conversations, image_count)
        
        # 只加载最后一个user的audio（按user轮次索引访问）
        audio_inputs, audio_len, audio_features_length = None, 0, 0
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if question_audios and user_count > 0 and user_count <= len(question_audios) and self.audio_processor:
            audio_bytes = question_audios[user_count - 1]
            if audio_bytes:
                mel, valid_len = self.load_audio_inputs(audio_bytes)
                if mel is not None:
                    audio_inputs = mel.unsqueeze(0)
                    audio_len = valid_len
                    audio_features_length = valid_len or 1
        
        # 混合训练时，无音频样本返回dummy tensor保持batch索引尽可能对齐 (SenseVoice: T x 560)
        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = torch.zeros(1, 3, 256, 256)
        
        # 从answer_audios获取最后一个assistant的音频codes
        last_audio_codes = None
        asst_count = sum(1 for t in conversations if t['role'] == 'assistant')
        if answer_audios and asst_count > 0 and asst_count <= len(answer_audios):
            tokens = answer_audios[asst_count - 1]
            if tokens:
                audio_codes_8layers = [[] for _ in range(8)]
                for i in range(0, len(tokens) - 7, 8):
                    for j in range(8): audio_codes_8layers[j].append(tokens[i + j])
                if self.audio_head_type == 'legacy_parallel_delay':
                    for layer in audio_codes_8layers:
                        layer.append(self.audio_stop_token)
                last_audio_codes = audio_codes_8layers
        
        # 生成prompt (text input_ids)
        prompt = self.create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None: prompt = prompt.replace('<image>', self.image_token)
        full_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        unpadded_ids = self.truncate_for_training(full_ids)
        unpadded_ids, valid_image_markers = self.sanitize_image_markers(unpadded_ids, image_count)
        attention_mask = [1] * len(unpadded_ids)
        input_ids = list(unpadded_ids)

        # If a too-long modality could not fit alongside the assistant target,
        # disable its encoder input for this sample as no injection marker is
        # present anymore.
        if self.audio_token_id not in input_ids and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if not valid_image_markers and self.vision_processor:
            pixel_values = torch.zeros(1, 3, 256, 256)
        
        # PAD input_ids到max_length
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        attention_mask += [0] * (self.max_length - len(attention_mask))
        
        # 生成labels（只训练最后一个assistant）
        text_labels, assistant_ranges = self.generate_text_labels(input_ids)
        for start, end in assistant_ranges[:-1]:
            mask_end = min(end + len(self.assistant_end_ids), self.max_length)
            text_labels[start:mask_end] = [-100] * (mask_end - start)
        
        # Generate eight aligned codec streams. The new head treats every
        # column as one real Mimi frame; the legacy branch retains its old
        # delayed layout solely for loading old checkpoints.
        Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(8)]
        audio_labels = [[-100] * self.max_length for _ in range(8)]
        if assistant_ranges and last_audio_codes:
            assistant_start, assistant_end = assistant_ranges[-1]
            for pos in range(assistant_start, assistant_end):
                if input_ids[pos:pos + len(self.think_end_ids)] == self.think_end_ids:
                    assistant_start = pos + len(self.think_end_ids)
                    break
            # Talker-only conditioning prefix. ref/spk never enter the Thinker
            # text sequence: [SPK][REF_START][ref codes...][REF_END].
            has_spk = bool(spk_emb_raw)
            has_ref = bool(ref_audios) and (not self.training or random.random() > 0.5)
            spk_reserve = 1 if has_spk else 0
            if has_ref:
                ref_codes = [[] for _ in range(8)]
                for i in range(0, len(ref_audios) - 7, 8):
                    for j in range(8): ref_codes[j].append(ref_audios[i + j])
                available = assistant_start - spk_reserve
                if self.use_talker_ref_boundaries and available >= 2:
                    code_count = min(len(ref_codes[0]), available - 2)
                    ref_start = assistant_start - code_count - 2
                    ref_end = assistant_start - 1
                    if self.audio_head_type == 'main_codec_predictor':
                        Y_audio_layers[0][ref_start] = self.audio_ref_start_token
                        Y_audio_layers[0][ref_end] = self.audio_ref_end_token
                    for layer_idx in range(8):
                        if self.audio_head_type == 'legacy_parallel_delay':
                            Y_audio_layers[layer_idx][ref_start] = self.audio_ref_start_token
                        codes = ref_codes[layer_idx][-code_count:] if code_count else []
                        for code_idx, code in enumerate(codes):
                            Y_audio_layers[layer_idx][ref_start + 1 + code_idx] = code
                        if self.audio_head_type == 'legacy_parallel_delay':
                            Y_audio_layers[layer_idx][ref_end] = self.audio_ref_end_token
                elif self.use_talker_ref_boundaries:
                    has_ref = False
                    ref_start = assistant_start
                else:
                    ref_start = max(spk_reserve, assistant_start - len(ref_codes[0]))
                    for layer_idx in range(8):
                        codes = ref_codes[layer_idx][-(assistant_start - ref_start):]
                        for code_idx, code in enumerate(codes):
                            Y_audio_layers[layer_idx][ref_start + code_idx] = code
            else:
                ref_start = assistant_start
            if has_spk and ref_start > 0:
                spk_pos = ref_start - 1
                if self.audio_head_type == 'main_codec_predictor':
                    Y_audio_layers[0][spk_pos] = self.audio_spk_token
                else:
                    for layer_idx in range(8):
                        Y_audio_layers[layer_idx][spk_pos] = self.audio_spk_token

            if self.audio_head_type == 'main_codec_predictor':
                # Unshifted layout before the global next-token slice:
                # input:  BOS, frame_0, ..., frame_last
                # target:      frame_0, ..., frame_last, EOS(main only)
                Y_audio_layers[0][assistant_start] = self.audio_bos_token
                # Reserve one final target slot for Main EOS.
                available_frames = max(0, self.max_length - assistant_start - 2)
                frame_count = min(len(last_audio_codes[0]), available_frames)
                for frame_idx in range(frame_count):
                    input_pos = assistant_start + 1 + frame_idx
                    for layer_idx in range(8):
                        code = last_audio_codes[layer_idx][frame_idx]
                        if not 0 <= code < self.base_audio_vocab_size:
                            raise ValueError(
                                f'sample {index} codebook {layer_idx} has invalid Mimi code {code}'
                            )
                        Y_audio_layers[layer_idx][input_pos] = code
                        audio_labels[layer_idx][input_pos] = code
                eos_pos = assistant_start + 1 + frame_count
                if frame_count and eos_pos < self.max_length:
                    audio_labels[0][eos_pos] = self.audio_stop_token
            else:
                for layer_idx in range(8):
                    codes = last_audio_codes[layer_idx]
                    start_pos = assistant_start + layer_idx + 1
                    for i, code in enumerate(codes):
                        if start_pos + i < self.max_length:
                            Y_audio_layers[layer_idx][start_pos + i] = code
                            audio_labels[layer_idx][start_pos + i] = code
        
        # 构造9路输入：input_ids = (9, T) = 8路audio + 1路text
        X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)  # (8, T-1)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)  # (T-1,)
        input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)  # (9, T-1)
        text_labels = torch.tensor(text_labels[1:], dtype=torch.long)  # (T-1,)
        audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)  # (8, T-1)
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)

        if not (text_labels != -100).any():
            raise ValueError(f"sample {index} has no assistant labels after truncation")
        if last_audio_codes and not (audio_labels != -100).any():
            raise ValueError(f"sample {index} has no audio labels after truncation")
        
        input_ids = self.apply_scheduled_sampling(input_ids, audio_labels, text_labels)
        spk_emb = torch.tensor(spk_emb_raw, dtype=torch.float32) if spk_emb_raw else torch.zeros(192)
        return input_ids, attention_mask, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb


# 测试parquet数据读取
if __name__ == '__main__':
    for path in ['sft_a2a.parquet']:
        if not os.path.exists(path): continue
        t = pa.Table.from_batches(pq.ParquetFile(path).iter_batches())
        conversations = json.loads(t['conversations'][0].as_py())
        answer_audios = t['answer_audios'][0].as_py() if 'answer_audios' in t.column_names else []
        user_msg = conversations[0]
        asst_msg = conversations[1] if len(conversations) > 1 else {}
        print(f'{path}: {len(t)}条, 列{t.column_names}')
        print(f'  User: {user_msg["content"][:50]}...')
        print(f'  Asst: {asst_msg.get("content", "")[:50]}...')
        if answer_audios:
            print(f'  answer_audios: {len(answer_audios)}轮, 首轮{len(answer_audios[0])}tokens')
