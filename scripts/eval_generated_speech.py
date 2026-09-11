#!/usr/bin/env python3
"""Transcribe fixed architecture-eval outputs and measure speech/text consistency."""

import argparse
import json
import re
import sys
from pathlib import Path

import librosa

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_arch_eval_report import parse_generation  # noqa: E402


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-log', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = AutoModel(
        model=args.model, trust_remote_code=True, disable_update=True,
        device='cpu', disable_log=True,
    )
    samples = parse_generation(Path(args.generation_log))
    utterances = []
    total_edits = total_reference_words = 0
    for sample in samples:
        output_path = Path(sample.get('output', ''))
        if not output_path.is_absolute():
            output_path = (Path(args.generation_log).parent / output_path).resolve()
        if not output_path.is_file():
            continue
        waveform, _ = librosa.load(output_path, sr=16000, mono=True)
        result = model.generate(
            input=waveform, cache={}, language='en', use_itn=True,
            disable_pbar=True, disable_log=True,
        )
        transcript = rich_transcription_postprocess(result[0]['text']).strip() if result else ''
        reference_words = words(sample.get('response', ''))
        hypothesis_words = words(transcript)
        edits = edit_distance(reference_words, hypothesis_words)
        total_edits += edits
        total_reference_words += len(reference_words)
        utterances.append({
            'kind': sample.get('kind'),
            'audio': str(output_path),
            'reference': sample.get('response', ''),
            'asr': transcript,
            'wer': edits / max(len(reference_words), 1),
            'frames': sample.get('frames', 0),
            'generation_seconds': sample.get('generation_seconds'),
            'rtf': sample.get('rtf'),
        })
    payload = {
        'utterances': utterances,
        'micro_wer': total_edits / max(total_reference_words, 1),
        'reference_words': total_reference_words,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Generated-speech ASR: {len(utterances)} samples, micro WER={payload['micro_wer']:.4f}")


if __name__ == '__main__':
    main()
