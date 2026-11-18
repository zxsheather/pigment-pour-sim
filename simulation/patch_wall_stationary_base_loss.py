"""Add supervision for accumulated base drift in stationary demonstration windows."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


HELPER = '''

def _stationary_base_drift_penalty(prediction, labels, dof_mask, flow_loss_mask, weight):
    """Penalize prefix displacement from denoised base velocity predictions."""
    if (labels.ndim != 3 or labels.shape[1:] != (8, 26)
            or prediction.shape != (labels.shape[0] * 8, 26) or not 0 < weight <= 16):
        raise ValueError("Stationary base supervision requires eight frames of 26-slot joint actions")
    teacher = labels[..., 7:9].float()
    stationary = (teacher.abs() <= 1e-6).all(dim=-1)
    if dof_mask is not None:
        stationary &= dof_mask.reshape_as(labels)[..., 7:9].bool().all(dim=-1)
    if flow_loss_mask is not None:
        stationary &= flow_loss_mask.reshape(labels.shape[:2]).bool()
    stationary_window = stationary.all(dim=1)[:, None, None]
    residual = prediction.reshape_as(labels)[..., 7:9].float() - teacher
    residual = residual.masked_fill(~stationary_window, 0.)
    # Both axes are normalized by their fixed command scale. Dividing the
    # prefix sums by the horizon expresses displacement over the same duration.
    drift = residual.cumsum(dim=1) / labels.shape[1]
    return torch.nn.functional.pad(weight * drift.square().reshape(-1, 2), (7, 17))
'''
ANCHOR = '''            if dof_mask is not None:
                dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1])
                loss = loss * dof_mask
'''
ADDITION = '''            base_weight = getattr(self.config, "joint_stationary_base_drift_weight", None)
            if base_weight is not None:
                loss = loss + _stationary_base_drift_penalty(
                    x_pred, action_chunk, dof_mask, flow_loss_mask, base_weight
                )

'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-root', type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.wall_root / 'wall_x/model/core/action/processor.py'
    before = source.read_text()
    if (args.output.exists() or source.is_symlink() or before.count(ANCHOR) != 1
            or before.count('loss + _closed_gripper_overopening_penalty(') != 1
            or '_stationary_base_drift_penalty' in before):
        raise ValueError('Expected a new isolated Wall-X processor copy with the gripper loss')
    norms = args.norm_stats.read_bytes()
    stats = json.loads(norms)['norm_stats']['action']
    low, high = stats['q01'][7:9], stats['q99'][7:9]
    if len(low) != 2 or len(high) != 2 or not all(
            math.isfinite(hi) and 0 < hi and lo == -hi for lo, hi in zip(low, high, strict=True)):
        raise ValueError('Base actions must use symmetric finite normalization around zero')
    definitions = before.index('\n\nclass ')
    after = before[:definitions] + '\n' + HELPER + before[definitions:]
    after = after.replace(ANCHOR, ADDITION + ANCHOR, 1)
    ast.parse(after)

    temporary = source.with_suffix('.stationary-base.tmp')
    with temporary.open('x') as handle:
        handle.write(after)
    temporary.replace(source)
    report = dict(kind='wall_x_stationary_base_drift_supervision', file=str(source.resolve()),
                  before_sha256=digest(before.encode()), after_sha256=digest(after.encode()),
                  training_config_field='joint_stationary_base_drift_weight',
                  normalizer_sha256=digest(norms), base_action_indices=[7, 8],
                  base_action_affine=[low, high], horizon=8,
                  only_fully_unmasked_stationary_target_windows=True,
                  retains_flow_and_gripper_loss=True, serving_changes=False)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
