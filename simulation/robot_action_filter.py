"""Optional task-independent low-pass filter for predicted robot commands.

Uses only robot-frame Cartesian commands and an initial robot state. Gripper
commands pass through so closure preload is preserved. No object/task feedback.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from cylinder_actions import rotation_from_rows


class RobotActionFilter:
    def __init__(self, initial_state, seconds=0., dt=.05):
        if not np.isfinite(seconds) or seconds < 0 or dt <= 0:
            raise ValueError("Invalid robot filter time constant")
        self.seconds = float(seconds)
        self.alpha = 1. if seconds == 0 else float(-np.expm1(-dt/seconds))
        self.last = np.asarray(initial_state, dtype=float).copy()
        assert self.last.shape == (23,)

    def __call__(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (23,) or not np.isfinite(command).all():
            raise ValueError("Invalid robot filter command")
        if self.seconds == 0:
            return command.copy()
        output = command.copy()
        output[10:13] = self.last[10:13] + self.alpha*(command[10:13]-self.last[10:13])
        before = rotation_from_rows(self.last[13:19])
        target = rotation_from_rows(command[13:19])
        delta = Rotation.from_matrix(target @ before.T).as_rotvec()
        rotation = Rotation.from_rotvec(self.alpha*delta).as_matrix() @ before
        output[13:19] = rotation[:2].reshape(6)
        output[20:23] = self.last[20:23] + self.alpha*(command[20:23]-self.last[20:23])
        self.last = output.copy()
        return output
