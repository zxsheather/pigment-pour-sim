"""In the opt-in increment layout, padded frames hold the last arm command."""

import argparse
import hashlib
import json
from pathlib import Path

from cylinder_joint_actions import JOINT_INCREMENT_ACTION_KEY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    loader = args.wall_root/'wall_x/data/backends/lerobot/loader.py'
    old = loader.read_text()
    marker = '        action = data[self._action_key_mapping["action"]]\n'
    patch = f'''        if self._dof_config.get({JOINT_INCREMENT_ACTION_KEY!r}) == 6:
            padding = data[self._action_key_mapping["action"] + "_is_pad"]
            action = action.clone()
            action[padding, :6] = 0
'''
    if old.count(marker) != 1 or patch in old or args.output.exists():
        raise ValueError('Expected unmodified increment padding in the pinned Wall loader')
    updated = old.replace(marker, marker + patch)
    compile(updated, str(loader), 'exec')
    loader.write_text(updated)
    with args.output.open('x') as audit:
        json.dump(dict(loader=str(loader), action_layout_key=JOINT_INCREMENT_ACTION_KEY,
                       before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                       after_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                       padded_arm_increment_zero=True, genuine_frames_preserved=True), audit, indent=2)
        audit.write('\n')


if __name__ == '__main__':
    main()
