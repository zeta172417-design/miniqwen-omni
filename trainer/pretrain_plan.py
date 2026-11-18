"""Explicit, checked migration of execution-only pretrain changes."""
import copy
import hashlib
import json
from pathlib import Path


def plan_signature(plan,phase,world_size):
    return hashlib.sha256(json.dumps({'plan':plan,'phase':phase,'world_size':world_size},sort_keys=True).encode()).hexdigest()


def training_contract(plan):
    result=copy.deepcopy(plan)
    for key in ['version','code_sha256','execution','tracking','resume_parent_plan','resume_parent_plan_sha256','comparison_notes']:
        result.pop(key,None)
    for phase in result['phases']:
        for key in ['ddp_no_sync','compress_control_gradients']:
            phase.pop(key,None)
    return result


def validate_checkpoint_signature(plan,phase,world_size,saved_signature,root):
    """Return original signature for a permitted migration, or None if current."""
    if saved_signature==plan_signature(plan,phase,world_size):return None
    parent_path=plan.get('resume_parent_plan')
    if not parent_path:raise ValueError('Existing checkpoint has different plan signature')
    parent_path=Path(root)/parent_path
    content=parent_path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=plan.get('resume_parent_plan_sha256'):
        raise ValueError('Resume parent plan fingerprint changed')
    parent=json.loads(content)
    if training_contract(parent)!=training_contract(plan):
        raise ValueError('Execution migration changed data, batch, schedule or training contract')
    if saved_signature!=plan_signature(parent,phase,world_size):
        raise ValueError('Checkpoint does not match declared resume parent')
    return saved_signature
