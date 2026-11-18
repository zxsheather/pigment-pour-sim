"""Plan and execute bottle moves in the attached-bottle MuJoCo scene."""

from dataclasses import dataclass

import mujoco
import numpy as np

from simulation.world_model.actuators import ARM_JOINTS
from simulation.world_model.collision import workspace_contacts


@dataclass(frozen=True)
class BottlePose:
    mouth_xyz_m: tuple[float, float, float]
    bottle_quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class MoveResult:
    status: str
    reason: str = ""


@dataclass
class _Move:
    target: BottlePose
    duration_s: float
    references: np.ndarray
    elapsed_s: float = 0.


class PourMotion:
    """Plan local Cartesian moves for the scene with a rigidly held bottle."""

    def __init__(self, model):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.mouth = model.site("held_bottle_mouth").id
        self.bottle = model.body("held_water_bottle").id
        joints = [model.joint(name) for name in (*ARM_JOINTS, "lift_joint")]
        self.qpos = [joint.qposadr[0] for joint in joints]
        self.dofs = [joint.dofadr[0] for joint in joints]
        self.servos = [model.actuator(f"{joint.name}_servo").id for joint in joints]
        self.lower, self.upper = model.jnt_range[[joint.id for joint in joints]].T
        self.gains = model.actuator_gainprm[self.servos, 0]
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros_like(self.jac_pos)
        self._active = None
        self._result = MoveResult("idle")

    def _pose(self, data):
        return BottlePose(tuple(data.site(self.mouth).xpos), tuple(data.body(self.bottle).xquat))

    def _solve(self, position, quaternion):
        data = self.scratch
        rotation_error = np.empty(3)
        for _ in range(70):
            mujoco.mj_forward(self.model, data)
            position_error = position - data.site(self.mouth).xpos
            mujoco.mju_subQuat(rotation_error, quaternion, data.body(self.bottle).xquat)
            if np.linalg.norm(position_error) < .00035 and np.linalg.norm(rotation_error) < .002:
                if workspace_contacts(self.model, data):
                    raise ValueError("Planned bottle path contacts the workspace")
                # Offset the servo setpoint to supply the modeled static load.
                load = data.qfrc_bias[self.dofs] - data.qfrc_gravcomp[self.dofs]
                reference = data.qpos[self.qpos] + load / self.gains
                if np.any(reference < self.lower) or np.any(reference > self.upper):
                    raise ValueError("Servo reference would exceed a joint limit")
                return reference
            mujoco.mj_jacSite(self.model, data, self.jac_pos, self.jac_rot, self.mouth)
            rotation = data.body(self.bottle).xmat.reshape(3, 3)
            jac = np.vstack((self.jac_pos[:, self.dofs], .2 * self.jac_rot[:, self.dofs]))
            error = np.r_[position_error, .2 * (rotation @ rotation_error)]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .004**2 * np.eye(6), error)
            data.qpos[self.qpos] = np.clip(
                data.qpos[self.qpos] + np.clip(delta, -.08, .08), self.lower, self.upper
            )
        raise ValueError("Inverse kinematics could not reach the requested bottle pose")

    def _plan(self, data, target, duration_s):
        position = np.asarray(target.mouth_xyz_m, dtype=float)
        quaternion = np.asarray(target.bottle_quat_wxyz, dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.all(np.isfinite(position)) or not np.isfinite(duration_s)
                or duration_s <= 0 or not np.isclose(np.linalg.norm(quaternion), 1.)):
            raise ValueError("Expected a finite bottle pose, unit quaternion, and positive duration")
        self.scratch.qpos[:] = data.qpos
        self.scratch.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.scratch)
        origin = self._pose(self.scratch)
        mujoco.mj_jacSite(self.model, self.scratch, self.jac_pos, self.jac_rot, self.mouth)
        linear_velocity = self.jac_pos @ data.qvel
        rotation_frame = self.scratch.body(self.bottle).xmat.reshape(3, 3)
        angular_velocity = rotation_frame.T @ (self.jac_rot @ data.qvel)
        angular_delta = np.empty(3)
        mujoco.mju_subQuat(angular_delta, quaternion, np.asarray(origin.bottle_quat_wxyz))
        references = []
        for fraction in np.linspace(0., 1., max(2, round(duration_s / .02) + 1)):
            blend = fraction**2 * (3. - 2. * fraction)
            # Match measured velocity at the start and come to rest at the target.
            tangent = duration_s * fraction * (1. - fraction)**2
            rotation = np.array(origin.bottle_quat_wxyz)
            mujoco.mju_quatIntegrate(rotation, blend * angular_delta + tangent * angular_velocity, 1.)
            xyz = (origin.mouth_xyz_m + blend * (position - origin.mouth_xyz_m)
                   + tangent * linear_velocity)
            references.append(self._solve(xyz, rotation))
        return np.asarray(references)

    def request_move(self, target_pose, duration_s, robot_state):
        """Plan synchronously and replace the active move only when accepted."""
        try:
            references = self._plan(robot_state, target_pose, duration_s)
        except ValueError as error:
            return MoveResult("rejected", str(error))
        target = BottlePose(tuple(target_pose.mouth_xyz_m), tuple(target_pose.bottle_quat_wxyz))
        self._active = _Move(target, duration_s, references)
        self._result = MoveResult("executing")
        return MoveResult("accepted")

    def update(self, dt, robot_state):
        """Write the next servo targets to MjData. The caller steps physics."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Expected a finite positive control timestep")
        move = self._active
        if move is None:
            return self._result
        if workspace_contacts(self.model, robot_state):
            return self._fail(robot_state, "Workspace contact during execution")
        if move.elapsed_s >= move.duration_s:
            pose = self._pose(robot_state)
            error = np.empty(3)
            mujoco.mju_subQuat(error, np.asarray(move.target.bottle_quat_wxyz),
                               np.asarray(pose.bottle_quat_wxyz))
            if (np.linalg.norm(np.subtract(move.target.mouth_xyz_m, pose.mouth_xyz_m)) <= .005
                    and np.linalg.norm(error) <= np.deg2rad(1.)):
                self._active = None
                self._result = MoveResult("reached")
                return self._result
            if move.elapsed_s >= move.duration_s + .8:
                return self._fail(robot_state, "Bottle pose did not settle within tolerance")
        move.elapsed_s += dt
        references = move.references
        index = min(move.elapsed_s / move.duration_s, 1.) * (len(references) - 1)
        left = int(index)
        right = min(left + 1, len(references) - 1)
        blend = index - left
        robot_state.ctrl[self.servos] = (1. - blend) * references[left] + blend * references[right]
        return self._result

    def _fail(self, data, reason):
        data.ctrl[self.servos] = np.clip(data.qpos[self.qpos], self.lower, self.upper)
        self._active = None
        self._result = MoveResult("failed", reason)
        return self._result
