"""Single source of truth for voice-reference budgets and preset names."""

MIMI_FRAME_RATE = 12.5
TRAINING_REFERENCE_MAX_SECONDS = 4.0
LEGACY_REFERENCE_MAX_SECONDS = 12.0
INFERENCE_REFERENCE_MAX_SECONDS = 12.0
TRAINING_REFERENCE_MAX_FRAMES = int(TRAINING_REFERENCE_MAX_SECONDS * MIMI_FRAME_RATE)
LEGACY_REFERENCE_MAX_FRAMES = int(LEGACY_REFERENCE_MAX_SECONDS * MIMI_FRAME_RATE)
INFERENCE_REFERENCE_MAX_FRAMES = int(INFERENCE_REFERENCE_MAX_SECONDS * MIMI_FRAME_RATE)
# Two boundary tokens and one speaker token are reserved alongside the
# reference frames. Keep a small margin for future control tokens.
CONDITION_PREFIX_LENGTH = INFERENCE_REFERENCE_MAX_FRAMES + 10
PRESET_VOICES = ("dylan", "eric", "serena", "uncle_fu", "vivian")


def crop_reference_codes(codes, max_frames=INFERENCE_REFERENCE_MAX_FRAMES):
    """Center-crop a (8,T) or (B,8,T) Mimi code tensor to the shared budget."""
    length = codes.shape[-1]
    if length <= max_frames:
        return codes
    start = (length - max_frames) // 2
    return codes[..., start:start + max_frames]
