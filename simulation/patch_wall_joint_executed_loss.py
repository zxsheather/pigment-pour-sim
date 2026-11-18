"""Add an opt-in loss for the executed gripper and base outputs to a Wall-X copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


HELPER_ANCHOR = 'logger = logging.get_logger(__name__)\n'
HELPER = '''


def _joint_executed_control_loss(loss, valid_mask, action_chunk, multiplier):
    """Reweight the first eight gripper and base elements of 32x26 joint chunks."""
    if action_chunk.ndim != 3 or action_chunk.shape[1:] != (32, 26):
        raise ValueError("Executed-control weighting requires 32x26 joint chunks")
    if valid_mask is None or valid_mask.shape != loss.shape:
        raise ValueError("Executed-control weighting requires the joint action mask")
    if loss.shape != (action_chunk.shape[0] * 32, 26) or not 1.0 < multiplier <= 4.0:
        raise ValueError("Invalid executed-control loss inputs")
    weights = loss.new_ones((32, 26))
    weights[:8, 6:9] = multiplier
    weighted_mask = valid_mask.reshape(-1, 32, 26) * weights
    return (loss.reshape(-1, 32, 26) * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)
'''
REDUCTION = '''                    if valid_mask is not None and not valid_mask.all():
                        flow_loss = _flow_loss.sum() / valid_mask.sum()
                    else:
                        flow_loss = _flow_loss.mean()
'''
NEW_REDUCTION = '''                    multiplier = getattr(self.config, "joint_executed_control_loss_multiplier", None)
                    if multiplier is not None:
                        flow_loss = _joint_executed_control_loss(_flow_loss, valid_mask, action_chunk, multiplier)
                    elif valid_mask is not None and not valid_mask.all():
                        flow_loss = _flow_loss.sum() / valid_mask.sum()
                    else:
                        flow_loss = _flow_loss.mean()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.wall_root / 'wall_x/model/core/vla_mixin.py'
    original = path.read_text()
    if (original.count(HELPER_ANCHOR) != 1 or original.count(REDUCTION) != 1
            or '_joint_executed_control_loss' in original):
        raise ValueError('Expected the unmodified pinned Wall-X flow reduction')
    patched = original.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER, 1)
    patched = patched.replace(REDUCTION, NEW_REDUCTION, 1)
    ast.parse(patched)
    with args.output.open('x') as report_file:
        temp = path.with_suffix('.executed-loss.tmp')
        with temp.open('x') as source_file:
            source_file.write(patched)
        temp.replace(path)
        report_file.write(json.dumps(dict(file=str(path.resolve()),
            before_sha256=hashlib.sha256(original.encode()).hexdigest(),
            after_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            configuration_key='joint_executed_control_loss_multiplier',
            weighted_steps=list(range(8)), weighted_action_channels=[6, 7, 8],
            only_valid_labels_contribute=True, serving_model_changed=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
