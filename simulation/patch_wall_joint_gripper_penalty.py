"""Penalize excess opening on narrow teacher targets in a Wall-X trainer copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


HELPER_ANCHOR = 'import torch\n'
HELPER = '''


def _closed_gripper_overopening_penalty(action_prediction, action_chunk, multiplier):
    """Apply to the first eight joint actions that request at most 20 mm."""
    if action_chunk.ndim != 3 or action_chunk.shape[2] != 26 or not 8 <= action_chunk.shape[1] <= 32:
        raise ValueError("Gripper overopening penalty requires 8–32 joint actions with 26 slots")
    batch, horizon, slots = action_chunk.shape
    if action_prediction.shape != (batch * horizon, slots) or not 0 < multiplier <= 16:
        raise ValueError("Invalid gripper overopening penalty inputs")
    prediction = action_prediction.reshape_as(action_chunk)
    # The joint export maps full physical aperture [0, 0.0975] m to [-1, 1].
    teacher = action_chunk[:, :8, 6]
    narrow = teacher <= (2 * .020 / .0975 - 1)
    excess = (prediction[:, :8, 6] - teacher).clamp_min(0)
    penalty = torch.zeros_like(prediction)
    # The trainer averages loss across the whole chunk. Preserve the auxiliary
    # term's contribution when that denominator changes with the horizon.
    penalty[:, :8, 6] = (multiplier * horizon / 32) * excess.square() * narrow
    return penalty.reshape_as(action_prediction)
'''
TARGET = '''            if dof_mask is not None:
                dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1])
                loss = loss * dof_mask
'''
ADDITION = '''            multiplier = getattr(self.config, "joint_closed_gripper_overopening_weight", None)
            if multiplier is not None:
                loss = loss + _closed_gripper_overopening_penalty(x_pred, action_chunk, multiplier)

'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.wall_root / 'wall_x/model/core/action/processor.py'
    before = source.read_text()
    if (before.count(HELPER_ANCHOR) != 1 or before.count(TARGET) != 1
            or '_closed_gripper_overopening_penalty' in before):
        raise ValueError('Expected the unmodified pinned Wall-X flow loss')
    definitions = before.index('\n\nclass ')
    after = before[:definitions] + HELPER + before[definitions:]
    after = after.replace(TARGET, ADDITION + TARGET, 1)
    ast.parse(after)
    with args.output.open('x') as report:
        temporary = source.with_suffix('.gripper-penalty.tmp')
        with temporary.open('x') as stream:
            stream.write(after)
        temporary.replace(source)
        details = dict(file=str(source.resolve()),
                       before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                       after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                       configuration_key='joint_closed_gripper_overopening_weight',
                       evaluated_steps=list(range(8)), gripper_action_index=6,
                       teacher_max_full_aperture_m=.020, full_aperture_normalization_m=[0., .0975],
                       uses_predicted_denoised_action=True, normal_wall_loss_preserved=True,
                       auxiliary_weight_scaled_relative_to_horizon32=True,
                       previous_action_labels_unchanged=True, serving_model_changed=False)
        report.write(json.dumps(details, indent=2)+'\n')


if __name__ == '__main__':
    main()
