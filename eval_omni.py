import argparse
import os
import random
import time
import warnings
import torch
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from transformers import MimiModel
from model.model_omni import MiniQwenOmni
from trainer.trainer_utils import load_omni_tokenizer, configure_token_ids, format_audio_prompt, format_image_prompt
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed, log_model_params
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def init_model(args):
    tokenizer = load_omni_tokenizer(args.load_from)
    model = MiniQwenOmni.from_pretrained(
        args.load_from,
        dtype=torch.bfloat16,
        audio_encoder_path=None,
        vision_model_path=None,
    )
    audio_encoder, audio_processor = MiniQwenOmni.load_sensevoice(args.audio_encoder_dir)
    vision_encoder, vision_processor = MiniQwenOmni.load_vision(args.vision_dir)
    object.__setattr__(model, 'audio_encoder', audio_encoder)
    object.__setattr__(model, 'audio_processor', audio_processor)
    object.__setattr__(model, 'vision_encoder', vision_encoder)
    object.__setattr__(model, 'vision_processor', vision_processor)
    configure_token_ids(model.config, tokenizer)
    log_model_params(model)
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    model.mimi_model = MimiModel.from_pretrained(args.mimi_dir).eval()
    return model.eval().to(args.device), tokenizer


def eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, output_name, pixel_values=None, history=None, audio_lens=None, ref_codes=None, spk_emb=None):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(args.open_thinking))
    x = torch.tensor(tokenizer(inputs_text, add_special_tokens=False).data['input_ids'], dtype=torch.long, device=args.device)[None, ...]
    newline_ids = tokenizer.encode('\n', add_special_tokens=False)
    if len(newline_ids) != 1:
        raise ValueError(f'newline must encode to one token, got {newline_ids}')

    audio_frames = []
    with torch.no_grad():
        generation_started = time.perf_counter()
        res_y = model.generate(x, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p, stream=True,
                               top_k=args.top_k, return_audio_codes=True, open_thinking=bool(args.open_thinking),
                               newline_token_id=newline_ids[0], pad_token_id=tokenizer.pad_token_id,
                               audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values,
                               ref_codes=ref_codes, spk_emb=spk_emb)
        print('📒 [Thinker]: ', end='', flush=True)
        history_idx = 0
        for y, audio_frame in res_y:
            if y is not None:
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if answer and answer[-1] != '�':
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)
            if audio_frame:
                audio_frames.append(audio_frame)
        print()
        generation_seconds = time.perf_counter() - generation_started

        if audio_frames:
            audio_seconds = len(audio_frames) / 12.5
            rtf = generation_seconds / max(audio_seconds, 1e-9)
            print(f'🎹 [Talker]: {len(audio_frames)} frames | generation={generation_seconds:.2f}s | RTF={rtf:.3f}', end=" ")
            if args.decode_audio:
                try:
                    codes = [f for f in audio_frames if f and len(f) == 8]
                    if not codes:
                        print('⚠️  生成的Mimi codes为空，跳过保存。')
                        return
                    mimi_codes = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0).to(args.device)
                    # Mimi's valid code range is [0, 2047].  Talker uses every
                    # id >=2048 as a stop/special id, so none may reach decode.
                    filtered = torch.where(mimi_codes >= 2048, torch.zeros_like(mimi_codes), mimi_codes)
                    audio = model.mimi_model.decode(filtered).audio_values
                    output_path = os.path.join(args.output_dir, output_name)
                    wav_path = output_path.rsplit('.', 1)[0] + '.wav'
                    sf.write(wav_path, audio.squeeze().float().cpu().numpy(), 24000)
                    AudioSegment.from_wav(wav_path).export(output_path, format='mp3', bitrate='64k')
                    os.remove(wav_path)
                    print(f'| Audio decoded to: {output_path}')
                except Exception as e:
                    print(f'⚠️  保存音频失败: {str(e)}')
            else:
                print("(decode_audio=off)\n")


def main():
    parser = argparse.ArgumentParser(description="MiniQwen-Omni Chat")
    parser.add_argument('--load_from', required=True, type=str, help="MiniQwen-Omni HF checkpoint目录")
    parser.add_argument('--audio_encoder_dir', default=project_path('model', 'SenseVoiceSmall'), type=str, help="SenseVoice模型目录")
    parser.add_argument('--vision_dir', default=project_path('model', 'siglip2-base-p32-256-ve'), type=str, help="SigLIP2模型目录")
    parser.add_argument('--mimi_dir', default=project_path('model', 'mimi'), type=str, help="Mimi codec模型目录")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--max_samples', default=0, type=int, help="每种模式最多评估样本数；0表示全部")
    parser.add_argument('--temperature', default=None, type=float, help="Thinker温度；默认读取generation_config")
    parser.add_argument('--top_p', default=None, type=float, help="nucleus采样；默认读取generation_config")
    parser.add_argument('--top_k', default=None, type=int, help="top-k；默认读取generation_config")
    parser.add_argument('--output_dir', default=project_path('output_audio'), type=str, help="输出音频保存目录")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--audio_dir', default=project_path('dataset', 'eval_omni'), type=str, help="测试音频目录")
    parser.add_argument('--image_dir', default=project_path('dataset', 'eval_omni'), type=str, help="测试图像目录")
    parser.add_argument('--open_thinking', default=0, type=int, choices=[0, 1], help="是否开启思考模式；开启时音频等待</think>后开始")
    parser.add_argument('--decode_audio', default=1, type=int, help="是否解码音频输出（0=否，1=是）")
    parser.add_argument('--mode', default='0', type=str, help="评估模式：-1=all 0=text 1=multi 2=audio 3=clone 4=image 5=mix（逗号组合，如 2,5）")
    parser.add_argument('--prompt_lang', default=0, type=int, choices=[0, 1, 2], help="问题语言：0=英文 1=中文 2=英文+中文")
    parser.add_argument('--seed', default=42, type=int, help="固定生成随机种子，便于架构实验复现")
    args = parser.parse_args()
    args.load_from = os.path.abspath(args.load_from)
    modes = set(args.mode.replace(',', '').replace('-1', '012345'))

    def limited(values):
        return values[:args.max_samples] if args.max_samples > 0 else values
    
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = init_model(args)
    setup_seed(args.seed)

    if '0' in modes:
        print('\n\n==================== text -> {text, audio} ====================')
        test_prompts_en = [
            "Tell me an interesting fact about space.", "How do I make a cup of coffee?", "What's the weather like today?",
            "Will it rain tomorrow?", "Tell me a joke.", "Can you sing a song for me?", "Please introduce yourself."
        ]
        test_prompts_zh = [
            "告诉我一个关于太空的有趣事实。", "如何制作一杯咖啡？", "今天的天气怎么样？",
            "明天会下雨吗？", "给我讲个笑话吧", "你能为我唱首歌吗？", "介绍一下你自己"
        ]
        test_prompts = [test_prompts_en, test_prompts_zh, test_prompts_en + test_prompts_zh][args.prompt_lang]
        test_prompts = limited(test_prompts)
        for idx, prompt in enumerate(test_prompts):
            print(f'\n📝 [text-{idx+1}]: {prompt}')
            eval_sample(model, tokenizer, args, idx, prompt, None, f"text-{idx:02d}.mp3")

    if '1' in modes:
        print('\n\n==================== multi-turn -> {text, audio} ====================')
        multi_turn_tests_zh = [
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
                ],
                "prompt": "我想找点事做，你有什么建议吗？"
            },
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                    {"role": "user", "content": "我想找点事做，你有什么建议吗？"},
                    {"role": "assistant", "content": "可以听听音乐或者看看书，放松一下心情。"}
                ],
                "prompt": "好的，那我去照做了，谢谢你"
            }
        ]
        multi_turn_tests_en = [
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ],
                "prompt": "I want to find something to do. Do you have any suggestions?"
            },
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                    {"role": "user", "content": "I want to find something to do. Do you have any suggestions?"},
                    {"role": "assistant", "content": "You can listen to music or read a book to relax a little."}
                ],
                "prompt": "Okay, I will try that. Thank you."
            }
        ]
        multi_turn_tests = [multi_turn_tests_en, multi_turn_tests_zh, multi_turn_tests_en + multi_turn_tests_zh][args.prompt_lang]
        multi_turn_tests = limited(multi_turn_tests)
        for idx, test in enumerate(multi_turn_tests):
            print(f'\n💬 [multi-{idx+1}]')
            for msg in test["history"]: print(f'   {msg["role"]}: {msg["content"]}')
            print(f'   user: {test["prompt"]}')
            eval_sample(model, tokenizer, args, idx, test["prompt"], None, f"multi-{idx:02d}.mp3", history=test["history"])

    if '2' in modes:
        print('\n\n==================== audio -> {text, audio} ====================')
        audio_files_en = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('audio-en-') and f.lower().endswith(('.mp3', '.wav'))])
        audio_files_zh = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('audio-zh-') and f.lower().endswith(('.mp3', '.wav'))])
        audio_files = [audio_files_en, audio_files_zh, audio_files_en + audio_files_zh][args.prompt_lang]
        audio_files = limited(audio_files)
        for idx, audio_file in enumerate(audio_files):
            print(f'\n🎤 [audio-{idx+1}]: {audio_file}')
            mel, valid_len = OmniDataset.process_audio(os.path.join(args.audio_dir, audio_file), model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(args.device)
            audio_lens = torch.tensor([valid_len], device=args.device)
            audio_token_len = valid_len or 1
            prompt = format_audio_prompt(model.config, audio_token_len)
            eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"audio-{idx:02d}-{os.path.splitext(audio_file)[0]}.mp3", audio_lens=audio_lens)

    if '3' in modes:
        print('\n\n==================== clone voice -> {text, audio} ====================')
        clone_prompts_en = ["Hello, please introduce yourself.", "What's the weather like today?", "Tell me a joke."]
        clone_prompts_zh = ["你好，请介绍一下你自己。", "今天天气怎么样？", "给我讲个笑话吧。"]
        clone_prompts = [clone_prompts_en, clone_prompts_zh, clone_prompts_en + clone_prompts_zh][args.prompt_lang]
        voices_pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'speaker', 'voices_unseen.pt')
        voices = [('default', None, None)]
        if os.path.exists(voices_pt):
            voice_data = torch.load(voices_pt, map_location=args.device)
            for speaker, v in sorted(voice_data.items()):
                rc = v['ref_codes'].unsqueeze(0).to(args.device)
                spk_dtype = next(model.talker.spk_proj.parameters()).dtype
                se = v['spk_emb'].to(dtype=spk_dtype).unsqueeze(0).to(args.device) if 'spk_emb' in v else None
                voices.append((speaker, rc, se))
        # Mode 3 should exercise actual reference conditioning when a voice
        # pack is available; keep the default voice only as a fallback.
        clone_voices = voices[1:] if len(voices) > 1 else voices
        clone_cases = [
            (speaker, rc, se, idx, prompt)
            for speaker, rc, se in clone_voices
            for idx, prompt in enumerate(clone_prompts)
        ]
        for speaker, rc, se, idx, prompt in limited(clone_cases):
            info = f'ref_codes: {rc.shape[2]} frames, spk_emb: {"+" if se is not None else "-"}' if rc is not None else ('spk_emb only' if se is not None else 'default')
            print(f'\n🎵 [clone: {speaker}] {info}')
            print(f'  📝 [text-{idx+1}]: {prompt}')
            history = [{"role": "system", "content": "你是一个专业的语音助手，请用给定的音色风格来回答用户的问题。请尽量详细地回答，给出有价值的信息。"}]
            eval_sample(model, tokenizer, args, idx, prompt, None, f"clone-{speaker}-{idx:02d}.mp3", ref_codes=rc, history=history, spk_emb=se)

    if '4' in modes:
        print('\n\n==================== image -> {text, audio} ====================')
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        image_files = limited(image_files)
        for idx, image_file in enumerate(image_files):
            print(f'\n🖼️ [image-{idx+1}]: {image_file}')
            image = Image.open(os.path.join(args.image_dir, image_file)).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}
            prompts = [["Please describe this image."], ["请描述这张图片"], ["Please describe this image.", "请描述这张图片"]][args.prompt_lang]
            for lang_idx, prompt_text in enumerate(prompts):
                prompt = prompt_text + "\n\n" + format_image_prompt(model.config)
                eval_sample(model, tokenizer, args, idx, prompt, None, f"image-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3", pixel_values=pixel_values)

    if '5' in modes:
        print('\n\n==================== text+audio+image -> {text, audio} ====================')
        img_audio_files = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('img-') and f.lower().endswith(('.mp3', '.wav'))])
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        image_files = limited(image_files)
        text_hints = [["Please answer me: "], ["请回答我："], ["Please answer me: ", "请回答我："]][args.prompt_lang]
        for idx, image_file in enumerate(image_files):
            audio_file = random.choice(img_audio_files)
            image = Image.open(os.path.join(args.image_dir, image_file)).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}
            for lang_idx, text_hint in enumerate(text_hints):
                print(f'\n🌀 [mix-{idx+1}-{lang_idx}]: {text_hint} | {audio_file} | {image_file}')
                mel, valid_len = OmniDataset.process_audio(os.path.join(args.audio_dir, audio_file), model.audio_processor)
                audio_inputs = mel.unsqueeze(0).to(args.device)
                audio_lens = torch.tensor([valid_len], device=args.device)
                audio_token_len = valid_len or 1
                prompt = text_hint + format_audio_prompt(model.config, audio_token_len) + "\n\n" + format_image_prompt(model.config)
                eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"mix-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3", pixel_values=pixel_values, audio_lens=audio_lens)


if __name__ == "__main__":
    main()
