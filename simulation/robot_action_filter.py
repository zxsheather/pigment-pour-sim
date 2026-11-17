"""Optional task-independent low-pass filter for predicted robot commands.

Uses only robot-frame Cartesian commands and an initial robot state. Gripper
commands pass through so closure preload is preserved. No object/task feedback.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from cylinder_actions import rotation_from_rows


def limit_pose_step(command, robot_state, translation=0., rotation=0.):
    """Bound a Cartesian command relative to measured robot pose before IK.

    A zero bound disables that limit. Base/gripper outputs pass through.
    This is a generic robot servo envelope, independent of task/object state.
    """
    output=np.asarray(command,dtype=float).copy()
    state=np.asarray(robot_state,dtype=float)
    if output.shape!=(23,) or state.shape!=(23,) or translation<0 or rotation<0:
        raise ValueError("Invalid Cartesian servo envelope")
    if translation:
        delta=output[10:13]-state[10:13];size=np.linalg.norm(delta)
        if size>translation:output[10:13]=state[10:13]+delta*(translation/size)
    if rotation:
        before=rotation_from_rows(state[13:19]);target=rotation_from_rows(output[13:19])
        delta=Rotation.from_matrix(target@before.T).as_rotvec();size=np.linalg.norm(delta)
        if size>rotation:
            target=Rotation.from_rotvec(delta*(rotation/size)).as_matrix()@before
            output[13:19]=target[:2].reshape(6)
    return output


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
