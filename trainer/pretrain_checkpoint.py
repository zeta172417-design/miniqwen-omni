"""Recover the last committed pretrain directory after an interrupted rename."""
import json
import os
from pathlib import Path

def complete_checkpoint(path):
    path=Path(path)
    if not (path/'config.json').is_file() or not (path/'trainer_state.pt').is_file():return False
    if (path/'model.safetensors').is_file():return True
    index=path/'model.safetensors.index.json'
    if not index.is_file():return False
    try:weights=set(json.loads(index.read_text())['weight_map'].values())
    except (ValueError,KeyError,TypeError):return False
    return bool(weights) and all((path/name).is_file() for name in weights)

def optimizer_checkpoint(path):
    """Return whether a checkpoint contains a resumable optimizer state.

    Older runs embedded it in trainer_state.pt; current runs keep it in a
    separate optimizer.pt to avoid a large serialization memory spike.
    """
    path=Path(path)
    state=path/'trainer_state.pt'
    if not state.is_file(): return False
    try:
        import torch
        value=torch.load(state,map_location='meta',weights_only=False)
        if 'optimizer' in value: return True
        name=value.get('optimizer_file')
        return bool(name and (path/name).is_file())
    except Exception:
        return False

def recover_checkpoint(root):
    """Restore only an existing committed backup, never an unfinished temp dir."""
    root=Path(root);stable=root/'checkpoint';previous=root/'.previous-checkpoint'
    if stable.exists():
        if not complete_checkpoint(stable):raise RuntimeError(f'Incomplete stable checkpoint; inspect before resuming: {stable}')
        return stable
    if previous.exists():
        if not complete_checkpoint(previous):raise RuntimeError(f'Incomplete checkpoint backup: {previous}')
        os.replace(previous,stable)
        return stable
    return None
