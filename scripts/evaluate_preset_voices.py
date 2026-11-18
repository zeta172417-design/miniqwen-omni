#!/usr/bin/env python3
"""Paired fixed-text comparison of stored voice presets and no conditioning."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VOICES = ['unconditioned', 'dylan', 'eric', 'serena', 'uncle_fu', 'vivian',
          'arthur', 'chelsie', 'cherry', 'ethan', 'jennifer', 'momo', 'moon']
SHARED = ROOT.parent / 'Omni'
MANIFEST = SHARED / 'benchmark-data/omni-core-v2.3-standard/manifest.jsonl'


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def worker(args):
    import torch
    import soundfile as sf
    from transformers import MimiModel
    from benchmark.run_voice_clone import load_voice_model
    from benchmark.metrics import audio_diagnostics

    run = Path(args.run_dir)
    protocol = json.loads((run / 'protocol.json').read_text())
    samples = read_rows(run / 'texts.jsonl')
    presets = {}
    for filename in ('voices.pt', 'voices_unseen.pt'):
        presets.update(torch.load(ROOT / 'model/speaker' / filename, map_location='cpu', weights_only=True))
    model, tokenizer, generation_token, template_kwargs = load_voice_model(
        {'kwargs': {'checkpoint': protocol['checkpoint']}}, 'cuda:0')
    mimi = MimiModel.from_pretrained(ROOT / 'model/mimi', local_files_only=True).eval().to('cuda:0')
    newline = tokenizer.encode('\n', add_special_tokens=False)
    assert len(newline) == 1
    out = run / f'generated-{args.worker}.jsonl'
    existing = {r['sample_id']: r for r in read_rows(out)} if out.exists() else {}
    cases = [(v, sample) for v in VOICES for sample in samples]
    cases = [case for i, case in enumerate(cases) if i % 4 == args.worker]
    with out.open('a', buffering=1) as stream:
        for index, (voice, sample) in enumerate(cases, 1):
            sample_id = voice + '--' + sample['sample_id']
            if sample_id in existing and existing[sample_id]['status'] == 'ok':
                continue
            torch.manual_seed(20260914)
            torch.cuda.manual_seed_all(20260914)
            ref_codes = spk_emb = None
            if voice != 'unconditioned':
                values = presets[voice]
                codes = values['ref_codes']
                start = max(0, (codes.shape[-1] - 50) // 2)
                ref_codes = codes[:, start:start + 50].unsqueeze(0).to('cuda:0')
                spk_emb = values['spk_emb'].to(device='cuda:0', dtype=next(model.talker.spk_proj.parameters()).dtype).unsqueeze(0)
            rendered = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': sample['prompt']}],
                tokenize=False, add_generation_prompt=True, **template_kwargs)
            ids = torch.tensor(tokenizer(rendered, add_special_tokens=False)['input_ids'], device='cuda:0').unsqueeze(0)
            frames, final_ids = [], None
            started = time.monotonic()
            row = {**sample, 'sample_id': sample_id, 'text_sample_id': sample['sample_id'],
                   'voice': voice, 'model_id': 'miniqwen-sft-step2500', 'status': 'ok',
                   'generated_audio': None, 'generated_text': '', 'error': None}
            try:
                with torch.inference_mode():
                    for generated_ids, frame in model.generate(
                        ids, max_new_tokens=256, stream=True, return_audio_codes=True,
                        open_thinking=False, do_sample=False, talker_do_sample=True,
                        temperature=1.0, top_k=1, top_p=1.0,
                        ref_codes=ref_codes, spk_emb=spk_emb,
                        **{generation_token: newline[0], 'pad_token_id': tokenizer.pad_token_id},
                    ):
                        if generated_ids is not None:
                            final_ids = generated_ids
                        if frame and len(frame) == 8:
                            frames.append(frame)
                    if final_ids is not None:
                        row['generated_text'] = tokenizer.decode(final_ids[0], skip_special_tokens=True).strip()
                    if frames:
                        codec = torch.tensor(frames, device='cuda:0').T.unsqueeze(0)
                        codec = torch.where(codec < 2048, codec, torch.zeros_like(codec))
                        waveform = mimi.decode(codec).audio_values.squeeze().float().cpu().numpy()
                        audio = run / 'audio' / voice / (sample['sample_id'] + '.wav')
                        audio.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(audio, waveform, 24000, subtype='PCM_16')
                        row['generated_audio'] = str(audio)
                    else:
                        row['status'] = 'empty_audio'
            except Exception as exc:
                import traceback
                traceback.print_exc()
                row.update(status='error', error=repr(exc))
            row['codec_frames'] = len(frames)
            row['audio_cap_reached'] = len(frames) >= 255
            row['latency_seconds'] = time.monotonic() - started
            row['audio_diagnostics'] = audio_diagnostics(row['generated_audio'])
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f'[{index}/{len(cases)}] {sample_id}: {row["status"]}, frames={len(frames)}, seconds={row["latency_seconds"]:.2f}', flush=True)


def report(run):
    import numpy as np
    from benchmark.metrics import normalize_asr_text
    all_rows = []
    for i in range(4):
        # Last row wins on a resumed failed case.
        all_rows.extend({r['sample_id']: r for r in read_rows(run / f'scored-{i}.jsonl')}.values())
    assert len(all_rows) == 832
    baseline = {r['text_sample_id']: r for r in all_rows if r['voice'] == 'unconditioned'}
    groups = {}
    rng = np.random.default_rng(42)
    for voice in VOICES:
        rows = [r for r in all_rows if r['voice'] == voice]
        good = [r for r in rows if r.get('utmos') is not None and r['status'] == 'ok']
        delta = np.array([r['utmos'] - baseline[r['text_sample_id']]['utmos'] for r in good
                          if baseline[r['text_sample_id']].get('utmos') is not None])
        boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1) if len(delta) else np.array([np.nan])
        # Corpus WER weighted by reference word count, alongside sample mean.
        weights = [len(normalize_asr_text(r['reference']).split()) for r in good]
        groups[voice] = dict(count=len(rows), valid=len(good),
                            utmos=statistics.mean(r['utmos'] for r in good) if good else None,
                            delta_mean=float(delta.mean()) if len(delta) else None,
                            delta_ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                            wer=sum(r.get('tts_wer', 1) * n for r, n in zip(good, weights)) / sum(weights) if sum(weights) else None,
                            mean_sample_wer=statistics.mean(r.get('tts_wer', 1) for r in good) if good else None,
                            mean_duration=statistics.mean(r['audio_diagnostics']['duration_seconds'] for r in good) if good else None,
                            cap_count=sum(r['audio_cap_reached'] for r in rows),
                            invalid_audio_count=sum(not r['audio_diagnostics']['audio_valid'] for r in rows))
    result = dict(protocol=json.loads((run / 'protocol.json').read_text()), voices=groups)
    save(run / 'summary.json', result)
    lines = ['# 预设音色与无参考生成对照', '',
             '使用暂停的 SFT step 2500 checkpoint，同一组 64 条英文 LibriSpeech test-clean TTS 文本，每种条件各生成 64 条，共 832 条。', '',
             '预设同时提供原有 speaker 向量与居中最多 50 帧 Mimi 参考编码；不改动原预设文件。无参考组两者均不提供。所有组使用相同提示词、逐条重置的随机种子及 256 个生成步上限。文本使用 greedy；音频采用 Qwen-Omni Talker 采样：主码 temperature=0.9、top_k=50、top_p=1.0，残差码 temperature=1.0、top_k=50、top_p=0.8。每种条件每条文本生成一次。', '',
             '默认/zero-shot 分组沿用原预设文件命名，不代表本次重新验证了说话人是否出现在全部训练数据中。', '',
             '| 音色 | 分组 | 有效/总数 | UTMOS ↑ | 相对无参考变化 | 配对变化 95% CI | WER ↓ | 达到生成上限 |',
             '|---|---|---:|---:|---:|---|---:|---:|']
    for voice in VOICES:
        v = groups[voice]
        category = '无参考' if voice == 'unconditioned' else ('默认' if voice in VOICES[1:6] else 'zero-shot')
        lo, hi = v['delta_ci95']
        lines.append(f'| {voice} | {category} | {v["valid"]}/{v["count"]} | {v["utmos"]:.4f} | {v["delta_mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {v["wer"] * 100:.2f}% | {v["cap_count"]} |')
    lines += ['', 'UTMOS 为自动预测的自然度分数（约 1–5），不是人工 MOS；变化区间按同文本配对 bootstrap 10,000 次计算。区间描述文本抽样差异，不包含不同生成随机种子的变异，也未进行多重比较校正。WER 使用同一 SenseVoice 识别器，按参考词数加权。评测仅覆盖英语朗读，不等同于中文、A2A 对话或音色相似度评价。', '',
              '## 结果解读', '',
              '本次 12 个预设的 UTMOS 均低于无参考组，配对变化的 95% 区间全部低于 0；WER 也均高于无参考组。预设中 UTMOS 最高的是 dylan，但它有 15/64 条触及生成上限，WER 为 32.60%，因此不能仅按预设内部的自然度排名确定默认音色。', '',
              '当前 SFT 尚未完成。这一结果支持当前英文朗读继续采用无参考条件；它不能证明这些说话人本身的音质差，也不能判定参考编码、speaker 向量或条件分支中的哪一项造成下降。后续可对较好的候选做 speaker-only / reference-only / 两者同时提供的对照，并在 SFT 完成后复测。', '',
              '## 听音样例', '', '每组提供相同的前三条文本。', '']
    selected = sorted(baseline)[:3]
    for text_id in selected:
        lines += [f'### {text_id}', '', baseline[text_id]['reference'], '']
        for voice in VOICES:
            r = next(r for r in all_rows if r['voice'] == voice and r['text_sample_id'] == text_id)
            if r['generated_audio']:
                path = r['generated_audio']
                lines += [f'**{voice}** · UTMOS {r.get("utmos", float("nan")):.3f}', '',
                          f'<audio controls preload="none" src="{path}"></audio>', '', f'[收听 WAV]({path})', '']
    lines += ['## 文件', '', f'- 完整结果：[{run / "summary.json"}]({run / "summary.json"})',
              f'- 全部音频：`{run / "audio"}`', '',
              '训练保持暂停。最近断点包含优化器和 4 个 rank 的随机状态；恢复将从 step 2500 继续，重做暂停前未保存的 96 个更新。', '']
    (ROOT / 'reports/PRESET_VOICE_NATURALNESS_SFT2500.md').write_text('\n'.join(lines))
    save(ROOT / 'reports/PRESET_VOICE_NATURALNESS_SFT2500.json', result)
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


def controller(args):
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    logs = run / 'logs'
    logs.mkdir(exist_ok=True)
    state = dict(status='RUNNING', pid=os.getpid(), started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def status(**kwargs):
        state.update(kwargs)
        save(run / 'status.json', state)
    python = ROOT / '.venv/bin/python'
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TORCH_HOME=str(ROOT / '.runtime/benchmark_cache/torch'), HF_HOME=str(ROOT / '.runtime/hf_cache'),
               NUMBA_CACHE_DIR=str(ROOT / '.runtime/numba_cache'), PYTHONUNBUFFERED='1',
               TRANSFORMERS_VERBOSITY='error', HF_HUB_DISABLE_PROGRESS_BARS='1')
    jobs = []
    def parallel(phase, commands):
        jobs.clear()
        status(phase=phase)
        try:
            for i, command in enumerate(commands):
                log = (logs / f'{phase}-{i}.log').open('a', buffering=1)
                process = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES':str(i)},
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                jobs.append((process, log))
            status(worker_pids=[p.pid for p, _ in jobs])
            while True:
                codes = [p.poll() for p, _ in jobs]
                if any(c not in (None, 0) for c in codes):
                    raise RuntimeError(f'{phase} workers failed: {codes}')
                if all(c == 0 for c in codes):
                    break
                time.sleep(5)
        finally:
            for p, log in jobs:
                if p.poll() is None:
                    p.terminate()
                log.close()
    try:
        parallel('generate', [[str(python), str(Path(__file__).resolve()), '--run-dir', str(run), '--worker', str(i)] for i in range(4)])
        parallel('score', [[str(python), str(ROOT / 'benchmark/evaluate_audio.py'), '--input', str(run / f'generated-{i}.jsonl'),
                            '--output', str(run / f'scored-{i}.jsonl'), '--asr', str(ROOT / 'model/SenseVoiceSmall'),
                            '--device', 'cuda:0', '--utmos-repo', str(SHARED / 'SpeechMOS'), '--skip-metrics',
                            '--reference-cache', str(run / 'reference-cache')] for i in range(4)])
        report(run)
        status(status='COMPLETE', phase='COMPLETE', finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as exc:
        status(status='FAILED', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        report(Path(args.run_dir).resolve())
    elif args.worker is not None:
        worker(args)
    else:
        controller(args)
