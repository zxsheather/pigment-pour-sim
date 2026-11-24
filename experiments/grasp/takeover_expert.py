"""Teacher-only recovery from actual model states, including fallen bottles.

This controller uses object pose/contact truth. It is never part of the
student observation or the model-only performance evaluation.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from reactive_expert import ReactiveExpert, local_command, tool_world


def bottle_tilt_degrees(env):
    return float(np.degrees(np.arccos(np.clip(abs(env.d.xmat[env.bottle].reshape(3, 3)[2, 2]), 0, 1))))


class AxisRecoveryExpert:
    """Approach perpendicular to the cylinder axis, from above when fallen."""

    grasp_height_m = .005
    grip_goal = 2.2

    def __init__(self, env):
        self.env = env
        self.phase = "axis_align"
        self.contact_seconds = 0.
        self.recoveries = 0
        self.lift_goal = None
        self.frame = self.choose_frame()

    def choose_frame(self):
        env = self.env
        axis = env.d.xmat[env.bottle].reshape(3, 3)[:, 2].copy()
        down = np.array([0., 0., -1.])
        approach = down - axis * np.dot(down, axis)
        if np.linalg.norm(approach) < .1:
            approach = np.array([1., 0., 0.])
            approach -= axis * np.dot(approach, axis)
        approach /= np.linalg.norm(approach)
        current = env.d.xmat[env.ee].reshape(3, 3)
        options = [np.column_stack([approach, np.cross(sign * axis, approach), sign * axis])
                   for sign in [1., -1.]]
        return min(options, key=lambda frame: Rotation.from_matrix(frame @ current.T).magnitude())

    def command(self):
        env = self.env
        bottle = env.d.xpos[env.bottle].copy()
        tool = tool_world(env)
        pose = env.arm_pose("right")
        if self.phase == "axis_lift_hold" and np.linalg.norm(tool - bottle) > .06:
            self.phase = "axis_align"
            self.recoveries += 1
            self.contact_seconds = 0.
            self.frame = self.choose_frame()
        target_rotation = env.d.xmat[env.arm_base].reshape(3, 3).T @ self.frame
        rotation = Rotation.from_matrix(target_rotation).as_euler("xyz")
        current = env.d.xmat[env.ee].reshape(3, 3)
        angle = float(Rotation.from_matrix(self.frame @ current.T).magnitude())
        grasp = bottle + [0., 0., self.grasp_height_m]
        distance = float(np.linalg.norm(tool - grasp))
        contacts = env.finger_contacts()
        if self.phase == "axis_align":
            self.contact_seconds = 0.
            if distance < .009 and angle < .05 and pose[6] >= 4.1:
                self.phase = "axis_close"
            else:
                approach = self.frame[:, 0]
                error = tool - grasp
                transverse = np.linalg.norm(error - approach * np.dot(error, approach))
                aligned = np.clip(1 - max(transverse - .006, 0.) / .035, 0., 1.)
                opened = np.clip((pose[6] - 3.6) / .5, 0., 1.)
                oriented = np.clip(1 - max(angle - .05, 0.) / .14, 0., 1.)
                fraction = min(aligned, opened, oriented)
                target = grasp - approach * .10 * (1 - fraction) + np.array([0., 0., .015]) * (1 - fraction)
                return local_command(env, target, 4.5, rotation), self.phase
        if self.phase == "axis_close":
            if distance > .025 or angle > .10 or (pose[6] < self.grip_goal + .05 and contacts < 2):
                self.phase = "axis_align"
                self.recoveries += 1
                self.contact_seconds = 0.
                return local_command(env, grasp - self.frame[:, 0] * .10 + [0., 0., .015], 4.5, rotation), self.phase
            self.contact_seconds = self.contact_seconds + env.step_time if contacts == 2 else 0.
            if self.contact_seconds >= .08:
                self.phase = "axis_lift_hold"
                self.lift_goal = bottle + [0., 0., .14]
            else:
                return local_command(env, grasp, self.grip_goal, rotation), self.phase
        return local_command(env, self.lift_goal, self.grip_goal, rotation), "axis_lift_hold"


class EndOnRecoveryExpert(AxisRecoveryExpert):
    """Pinch the upper cylinder sides with a horizontal, end-on approach."""

    grasp_height_m = .050
    grip_goal = 1.0

    def choose_frame(self):
        env = self.env
        axis = env.d.xmat[env.bottle].reshape(3, 3)[:, 2].copy()
        axis[2] = 0.
        axis /= np.linalg.norm(axis)
        up = np.array([0., 0., 1.])
        current = env.d.xmat[env.ee].reshape(3, 3)
        options = [np.column_stack([sign * axis, np.cross(up, sign * axis), up])
                   for sign in [1., -1.]]
        return min(options, key=lambda frame: Rotation.from_matrix(frame @ current.T).magnitude())


def make_takeover_teacher(env, mode):
    if bottle_tilt_degrees(env) > 35:
        if mode == "axis":
            return AxisRecoveryExpert(env)
        if mode == "endon":
            return EndOnRecoveryExpert(env)
    return ReactiveExpert(env)
