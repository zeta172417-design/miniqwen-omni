"""Task-aware training-time voice-condition sampling."""
from __future__ import annotations
import hashlib

CONDITION_WEIGHTS = (("both", 60), ("speaker", 15), ("reference", 10), ("none", 15))

def voice_pool(row: dict) -> str:
    source = str(row.get("source") or "")
    task = str(row.get("task_type") or "")
    if source == "old_t2a": return "legacy_unconditioned_t2a"
    if task.startswith("tts") or source in {"libritts_r", "aishell3", "old_a2a"}:
        return "conditioned_tts"
    return "other_audio"

def _available_modes(row: dict) -> list[str]:
    has_ref, has_spk = bool(row.get("ref_audios") or []) and not row.get("voice_self_fallback", False), bool(row.get("spk_emb") or [])
    modes = ["both"] if has_ref and has_spk else []
    if has_spk: modes.append("speaker")
    if has_ref: modes.append("reference")
    modes.append("none")
    return modes

def choose_voice_mode(row: dict, seed: int, epoch: int = 0) -> tuple[str, str]:
    pool = voice_pool(row)
    if pool == "legacy_unconditioned_t2a": return pool, "none"
    if pool != "conditioned_tts": return pool, str(row.get("voice_conditioning") or "available_both")
    available = set(_available_modes(row))
    weighted = [(m, w) for m, w in CONDITION_WEIGHTS if m in available]
    draw = int.from_bytes(hashlib.sha256(f"voice-condition:{seed}:{epoch}:{row.get('sample_id','')}".encode()).digest()[:8], "big") % sum(w for _, w in weighted)
    for mode, weight in weighted:
        if draw < weight: return pool, mode
        draw -= weight
    return pool, weighted[-1][0]
