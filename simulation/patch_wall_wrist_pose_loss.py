"""Add opt-in cylinder wrist supervision to an isolated Wall-X training copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


PROCESSOR = 'wall_x/model/core/action/processor.py'
MIXIN = 'wall_x/model/core/vla_mixin.py'
MODEL = 'wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py'
HELPER = 'wall_x/model/core/action/cylinder_wrist_pose_loss.py'


MODES = {
    'absolute': ('wrist_pose_auxiliary_loss', 'joint_wrist_pose_loss'),
    'coupled': ('wrist_pose_with_base_motion_loss', 'joint_base_and_wrist_pose_loss'),
    'increment': ('servo_increment_wrist_pose_loss', 'joint_servo_wrist_pose_loss'),
}


def processor_edits(mode):
    name, config = MODES[mode]
    masks = '' if mode == 'absolute' else ', dof_mask, flow_loss_mask'
    return (
        ('import torch\n', f'import torch\n\nfrom .cylinder_wrist_pose_loss import {name}\n'),
        ('        flow_loss_mask=None,\n    ):\n        with torch.autocast',
         '        flow_loss_mask=None,\n        proprioception=None,\n    ):\n        with torch.autocast'),
        ('            if dof_mask is not None:\n                dof_mask = dof_mask.reshape',
         f'            wrist_config = getattr(self.config, "{config}", None)\n'
         '            if wrist_config is not None:\n'
         f'                loss = loss + {name}(x_pred, action_chunk, proprioception, wrist_config{masks})\n\n'
         '            if dof_mask is not None:\n                dof_mask = dof_mask.reshape'),
    )


EDITS = {
    MIXIN: (
        ('                    action_hidden_states, flow, action_chunk, dof_mask, flow_loss_mask\n',
         '                    action_hidden_states, flow, action_chunk, dof_mask, flow_loss_mask,\n'
         '                    proprioception=kwargs.get("proprioception"),\n'),
    ),
    MODEL: (
        ('            flow_loss_mask=flow_loss_mask,\n            _lm_loss_mask=_lm_loss_mask,\n',
         '            flow_loss_mask=flow_loss_mask,\n            _lm_loss_mask=_lm_loss_mask,\n'
         '            proprioception=proprioception,\n'),
    ),
}


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--couple-base', dest='mode', action='store_const', const='coupled',
                         help='Include the base twist prefix and emphasize narrow teacher targets.')
    options.add_argument('--servo-increments', dest='mode', action='store_const', const='increment',
                         help='Compose rate-limited arm increments before comparing wrist poses.')
    parser.set_defaults(mode='absolute')
    args = parser.parse_args()
    root = args.wall_root.resolve()
    helper = root / HELPER
    if args.output.exists():
        raise ValueError('Expected a new patch report')
    module = Path(__file__).with_name('cylinder_wrist_pose_loss.py').read_text()
    processor = processor_edits(args.mode)
    edits = {PROCESSOR: processor, **EDITS}
    existing_module = None
    if helper.exists():
        if args.mode != 'increment':
            raise ValueError('Expected an unpatched Wall-X copy for this mode')
        existing_module = helper.read_text()
        if ('def wrist_markers(' not in existing_module or 'def servo_arm_targets(' in existing_module
                or processor[1][1] not in (root / PROCESSOR).read_text()
                or any(after not in (root / path).read_text() for path, pairs in EDITS.items()
                       for _, after in pairs)):
            raise ValueError('Existing wrist patch does not have the expected proprioception path')
        edits = {PROCESSOR: (processor[0], processor[2])}
        module = existing_module.rstrip() + '\n\n\n' + module[module.index('def servo_arm_targets('):]
    changes, source_audit = {}, {}
    for name, pairs in edits.items():
        file = root / name
        original = file.read_text()
        updated = original
        for before, after in pairs:
            if updated.count(before) != 1:
                raise ValueError(f'Unexpected pinned Wall-X source for wrist supervision: {name}')
            updated = updated.replace(before, after, 1)
        ast.parse(updated)
        changes[file] = updated
        source_audit[name] = dict(before_sha256=digest(original), after_sha256=digest(updated))
    ast.parse(module)
    changes[helper] = module
    source_audit[HELPER] = dict(after_sha256=digest(module))
    if existing_module is not None:
        source_audit[HELPER]['before_sha256'] = digest(existing_module)

    temporary = {}
    for file, text in changes.items():
        path = file.with_suffix('.wrist-pose.tmp')
        with path.open('x') as f:
            f.write(text)
        temporary[file] = path
    for file, path in temporary.items():
        path.replace(file)
    report = dict(kind='wall_x_opt_in_wrist_pose_loss', training_wall_root=str(root), files=source_audit,
                  training_config_field=MODES[args.mode][1],
                  uses_only_arm_state_and_joint_targets=args.mode != 'coupled',
                  base_motion_coupled=args.mode == 'coupled', servo_increments=args.mode == 'increment',
                  serving_code_modified=False,
                  retained_existing_flow_and_gripper_losses=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
