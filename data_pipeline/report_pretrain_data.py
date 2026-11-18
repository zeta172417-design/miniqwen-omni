#!/usr/bin/env python3
"""Render actual download/conversion state; never equate partial bytes to ready data."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    rows = []
    details = []
    catalog_path = ROOT / 'dataset/pretrain/manifest.json'
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else None
    for source in ['librispeech', 'aishell1', 'llava', 'libritts_r', 'aishell3']:
        raw = ROOT / 'dataset/pretrain/raw' / source
        output = ROOT / 'dataset/pretrain/processed' / source
        receipts = list(raw.rglob('*.download.json')) if raw.exists() else []
        downloaded = sum(json.loads(p.read_text())['size'] for p in receipts)
        manifest = output / 'manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text())
            status = data['status']
            splits = catalog['sources'][source]['splits'] if catalog else data['splits']
            counts = ', '.join(f"{s}: {v.get('rows', 0) if isinstance(v, dict) else v:,}" for s, v in splits.items())
            details.append(f"- `{source}`：`{manifest.relative_to(ROOT)}`；隔离 {data.get('quarantined', 0)} 条。")
        else:
            status, counts = ('downloading / conversion pending' if raw.exists() else 'not started'), '—'
        hours = ', '.join(f"{s}: {v['seconds']/3600:.2f}h" for s, v in splits.items()
                          if isinstance(v, dict) and 'seconds' in v) if manifest.exists() else '—'
        rows.append(f'| {source} | {downloaded / 1e9:.2f} GB | {status} | {counts} | {hours or "—"} |')
        quarantine = output / 'quarantine.json'
        if quarantine.exists():
            reasons = Counter(item['error'] for item in json.loads(quarantine.read_text()))
            if reasons:
                details.append(f'- `{source}` 隔离原因：' + '；'.join(f'{reason}: {count}' for reason, count in reasons.items()) + '。原始语料保留。')
        if manifest.exists() and data.get('data_files'):
            details.append(f'- `{source}` 的完整编码分片路径见源 manifest 的 `data_files`；默认训练分片以统一清单为准。')
    text = '''# Pretrain 第一批数据准备状态

更新：%s。仅报告已落盘状态，不代表正式训练完成。

| 数据源 | 已完成下载并有校验记录的原始文件 | 状态 | 样本数 | 实际音频时长 |
|---|---:|---|---|---|
%s

## 已确认的处理原则

- 基础 catalog 的 TTS 不加入音色参考；有官方 normalized text 时使用该文本。后续条件视图单独存放在 `dataset/pretrain/voice/`。
- 原音频、原始转写、拼音/规范化字段保留；不补标点、不自动切句、不按长度静默删除。
- 损坏/空媒体单独记录；正式训练时的长度预算另行确定。
- ASR train/dev 保持官方边界；不导出 test 为训练数据。
- LLaVA 自建约 1%% 的按图片原始字节 SHA256 分组 dev；尚不声称完成跨数据集近重复去污染。
- AISHELL-3 从官方 train 按 speaker SHA256 留出约 5%% 说话人作 dev，官方 test 不导出为训练数据。
- `raw_segments_ready_codec_pending` 只表示 TTS 原始分段与文本就绪，尚不能直接训练音频输出。
- `training_format_ready` 表示格式与编码已完成；不是正式训练或全批量训练稳定性已验证。
- 早期基础 reader 验收使用代表性样本和 8192 长度；后续正式预算、条件视图与训练入口验收以 readiness 报告为准。

## 产物与复现

%s

```bash
.venv/bin/python -B -u data_pipeline/run_pretrain_data.py
.venv/bin/python -B -u data_pipeline/run_pretrain_tts.py
.venv/bin/python data_pipeline/build_pretrain_catalog.py --libritts-quality exclude_official
.venv/bin/python -B -u data_pipeline/validate_pretrain_data.py
.venv/bin/python data_pipeline/report_pretrain_data.py
```

网络下载需要有外网访问的执行环境。原始大文件和处理产物位于 `dataset/pretrain/`，已排除 Git；日志位于 `.runtime/pretrain/`。
默认优先使用已核对版本/哈希的 ModelScope 国内镜像；LibriTTS-R 保持官方归档并按官方 MD5 核验。固定上游版本、实际镜像及文件 SHA256 另见 [PRETRAIN_DATA_SOURCES_V1.json](PRETRAIN_DATA_SOURCES_V1.json)。
本次没有启动新的正式训练，没有覆盖基线 checkpoint，也没有修改系统/verl 依赖。
详细方案见 [PRETRAIN_SFT_DATA_PLAN_V1.md](PRETRAIN_SFT_DATA_PLAN_V1.md)。
''' % (datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), '\n'.join(rows), '\n'.join(details) or '完整转换尚未完成。')
    target = ROOT / 'reports/PRETRAIN_DATA_STATUS.md'
    validation_path = ROOT / 'reports/PRETRAIN_DATA_VALIDATION_V1.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        text += f"\n验收：{validation['unit_tests_passed']} 项数据处理测试、{validation['total_reader_samples']} 条真实样本读取/标签检查通过。记录见 [PRETRAIN_DATA_VALIDATION_V1.json](PRETRAIN_DATA_VALIDATION_V1.json)。\n"
    if catalog:
        text += '\n统一训练文件清单：`dataset/pretrain/manifest.json`。表中数量对应该清单的默认视图；原始与完整编码保留。\n'
        quality = catalog['sources']['libritts_r']
        text += f"\nLibriTTS-R 官方修复失败名单处理：`{catalog['libritts_quality_policy']}`；标记数量 {quality['official_flagged_counts']}；默认视图排除数量 {quality['excluded_counts']}。\n"
        if catalog['quality_decision_pending']:
            text += '\n待用户确认是否从默认视图排除官方修复失败名单；当前未执行该筛选。\n'
    if (ROOT/'dataset/pretrain/voice/manifest.json').exists():
        note='\n> 后续已补齐同一参考裁剪的 CAMPPlus + Mimi 条件视图、speaker-only warmup 与条件缺失策略；当前执行配置与验收见 [PRETRAIN_READINESS_V1.md](PRETRAIN_READINESS_V1.md)。本表仍表示基础数据视图。\n'
        first,rest=text.split('\n',1);text=first+note+rest
    target.write_text(text)
    print(text)


if __name__ == '__main__':
    render()
