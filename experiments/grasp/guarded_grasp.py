"""Local preclosure alignment using privileged simulation object geometry.

The model proposes the approach. This explicit controller gates closure,
reopens/retreats when needed, and aligns before seating. It is not perception.
The post-contact lift is inherited unchanged from the previous depth probe.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from diagnostic_environment import PhysicsTrace
from grasp_depth_probe import SeatingTrace


@dataclass(frozen=True)
class GuardConfig:
    target_x_m: float = .132
    min_x_m: float = .123
    max_x_m: float = .142
    target_z_m: float = -.005
    lateral_tolerance_m: float = .004
    height_tolerance_m: float = .006
    axis_tolerance_deg: float = 6.
    open_min: float = 4.1
    close_request_below: float = 3.6
    pregrasp_x_m: float = .185
    max_translation_step_m: float = .0024
    max_rotation_step_rad: float = .025
    ready_seconds: float = .12
    timeout_seconds: float = 5.
    max_retries: int = 2


def measurements(env):
    rotation = env.d.xmat[env.ee].reshape(3, 3)
    axis = env.d.xmat[env.bottle].reshape(3, 3)[:, 2].copy()
    if axis[2] < 0:
        axis *= -1
    center = rotation.T @ (env.d.xpos[env.bottle] - env.d.xpos[env.ee])
    return {"center_m": center.tolist(),
        "axis_error_deg": float(np.degrees(np.arccos(np.clip(rotation[:, 2] @ axis, -1., 1.)))),
        "bottle_tilt_deg": float(np.degrees(np.arccos(np.clip(axis[2], -1., 1.)))),
        "opening_min": min(env.gripper_feedback()["opening_motor_units"])}


def geometry_ready(value, config, opening=True):
    x, y, z = value["center_m"]
    return (config.min_x_m <= x <= config.max_x_m
        and abs(y) <= config.lateral_tolerance_m
        and abs(z - config.target_z_m) <= config.height_tolerance_m
        and value["axis_error_deg"] <= config.axis_tolerance_deg
        and (not opening or value["opening_min"] >= config.open_min))


def closing_geometry_valid(value):
    # Slightly wider tolerances allow physical settling while the jaws close.
    x, y, z = value["center_m"]
    return (.115 <= x <= .150 and abs(y) <= .010
        and abs(z + .005) <= .025 and value["axis_error_deg"] <= 10.)


def bounded_pose(actual, target, config):
    command = np.asarray(target, dtype=float).copy()
    delta = command[:3] - actual[:3]
    command[:3] = actual[:3] + delta * min(1., config.max_translation_step_m / max(np.linalg.norm(delta), 1e-12))
    current = Rotation.from_euler("xyz", actual[3:6])
    error = (Rotation.from_euler("xyz", command[3:6]) * current.inv()).as_rotvec()
    error *= min(1., config.max_rotation_step_rad / max(np.linalg.norm(error), 1e-12))
    command[3:6] = (Rotation.from_rotvec(error) * current).as_euler("xyz")
    return command


class GuardedTrace(SeatingTrace):
    def __init__(self, geometry, hand_bodies, lift_axis, config=None):
        super().__init__(0., geometry, hand_bodies, lift_axis)
        self.config = config or GuardConfig()
        self.phase = "policy"
        self.started_s = None
        self.phase_start_s = 0.
        self.ready_s = 0.
        self.retries = 0
        self.events = []
        self.guard_log = []
        self.target_rotation = None
        self.approach_x = None
        self.anchor = None
        self.abort_reason = None

    def change(self, phase, now, value, reason):
        self.events.append({"time_s": now, "from": self.phase, "to": phase,
            "reason": reason, "geometry": value})
        self.phase, self.phase_start_s, self.ready_s = phase, now, 0.

    def aligned_rotation(self, env):
        current = env.d.xmat[env.ee].reshape(3, 3)
        z = env.d.xmat[env.bottle].reshape(3, 3)[:, 2].copy()
        if z[2] < 0:
            z *= -1
        reference = current[:, 0] if self.approach_x is None else self.approach_x
        x = reference - z * (reference @ z)
        if np.linalg.norm(x) < .1:
            reference = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
            x = reference - z * (reference @ z)
        x /= np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z])

    def target(self, env, center, world_rotation):
        # Keep the published TCP unchanged; place the flange around the bottle.
        flange = env.d.xpos[env.bottle] - world_rotation @ np.asarray(center)
        world_tcp = flange + world_rotation @ env.tcp_offset
        base = env.d.xmat[env.arm_base].reshape(3, 3)
        local = base.T @ (world_tcp - env.d.xpos[env.arm_base])
        return np.r_[local - env.state_offsets["right"],
            Rotation.from_matrix(base.T @ world_rotation).as_euler("xyz"), 4.5]

    def handoff_allowed(self, env):
        return self.phase == "close" and closing_geometry_valid(measurements(env))

    def guarded_command(self, env, raw, step, chunk):
        now, cfg = step * .04, self.config
        value, actual = measurements(env), env.arm_pose("right")
        if self.handoff is not None:
            if self.phase != "lift_hold":
                self.change("lift_hold", now, value, "80ms bilateral contact with valid seating geometry")
            command = super().command(raw, step, chunk)
        else:
            requested_close = raw[6] < cfg.close_request_below
            x, y, z = value["center_m"]
            near = .09 <= x <= .22 and abs(y) <= .05 and abs(z + .005) <= .06
            if self.phase == "policy" and requested_close and near:
                self.started_s = now
                self.approach_x = env.d.xmat[env.ee].reshape(3, 3)[:, 0].copy()
                self.target_rotation = self.aligned_rotation(env)
                if geometry_ready(value, cfg):
                    self.anchor = actual.copy()
                    self.change("ready", now, value, "already sufficiently deep and aligned; no insertion")
                elif (abs(y) <= cfg.lateral_tolerance_m and abs(z - cfg.target_z_m) <= cfg.height_tolerance_m
                        and value["axis_error_deg"] <= cfg.axis_tolerance_deg and value["opening_min"] >= cfg.open_min):
                    self.change("seat", now, value, "aligned; correct measured depth only")
                else:
                    self.change("retreat", now, value, "open and withdraw before correcting alignment")
            if (self.phase not in ["policy", "abort"] and
                    (now - self.started_s > cfg.timeout_seconds or value["bottle_tilt_deg"] > 20.)):
                self.abort_reason = "alignment_timeout" if now - self.started_s > cfg.timeout_seconds else "bottle_tilt_out_of_scope"
                self.anchor = actual.copy()
                self.change("abort", now, value, self.abort_reason)
            if self.phase in ["align", "seat"]:
                # A bottle can rock during approach and settle while we retreat.
                # Keep approach heading, but track its current axis while open.
                self.target_rotation = self.aligned_rotation(env)
            command = np.asarray(raw, dtype=float).copy()
            if self.phase == "policy":
                if requested_close:
                    command[6] = 4.5
            elif self.phase == "abort":
                command = self.anchor.copy()
                command[6] = 4.5
            elif self.phase == "retreat":
                rotation = env.d.xmat[env.ee].reshape(3, 3)
                command = self.target(env, [cfg.pregrasp_x_m, y, z], rotation)
                if x >= cfg.pregrasp_x_m - .008 and value["opening_min"] >= cfg.open_min:
                    self.change("align", now, value, "open jaws withdrawn clear of bottle")
            elif self.phase == "align":
                command = self.target(env, [cfg.pregrasp_x_m, 0., cfg.target_z_m], self.target_rotation)
                if (abs(y) <= cfg.lateral_tolerance_m and abs(z - cfg.target_z_m) <= cfg.height_tolerance_m
                        and value["axis_error_deg"] <= cfg.axis_tolerance_deg
                        and abs(x - cfg.pregrasp_x_m) <= .010 and value["opening_min"] >= cfg.open_min):
                    self.change("seat", now, value, "lateral, height and axis aligned while open")
            elif self.phase == "seat":
                desired_x = cfg.target_x_m if not cfg.min_x_m <= x <= cfg.max_x_m else x
                command = self.target(env, [desired_x, 0., cfg.target_z_m], self.target_rotation)
                if geometry_ready(value, cfg):
                    self.anchor = actual.copy()
                    self.change("ready", now, value, "measured seating depth and full alignment reached")
            elif self.phase == "ready":
                command = self.anchor.copy()
                command[6] = 4.5
                self.ready_s = self.ready_s + .04 if geometry_ready(value, cfg) else 0.
                if not geometry_ready(value, cfg):
                    self.change("seat", now, value, "alignment changed before closure")
                elif self.ready_s >= cfg.ready_seconds - 1e-9:
                    self.anchor = actual.copy()
                    self.change("close", now, value, "geometry and opening stable for 120ms")
            elif self.phase == "close":
                command = self.anchor.copy()
                command[6] = 2.2
                if not closing_geometry_valid(value) or now - self.phase_start_s > .8:
                    self.retries += 1
                    if self.retries > cfg.max_retries:
                        self.abort_reason = "closure_failed_after_retries"
                        self.anchor = actual.copy()
                        self.change("abort", now, value, self.abort_reason)
                    else:
                        self.change("retreat", now, value, "closure geometry lost or no sustained contact; reopen")
                    command = actual.copy()
                    command[6] = 4.5
            if self.phase not in ["policy", "close", "abort"]:
                command = bounded_pose(actual, command, cfg)
            # Closing can only occur after the explicit ready stage.
            assert command[6] >= cfg.close_request_below or self.phase == "close"
            PhysicsTrace.command(self, command, step, chunk)
            self.raw = np.asarray(raw).copy()
        self.guard_log.append({"control_step": step, "time_s": now, "phase": self.phase,
            "geometry": value, "raw_policy_reference": np.asarray(raw).tolist(),
            "applied": command.tolist(), "uses_simulator_object_pose": True})
        return command

    def protocol(self):
        return {"config": asdict(self.config), "events": self.events, "retries": self.retries,
            "abort_reason": self.abort_reason, "object_pose_is_privileged": True,
            "axis_reference": "current bottle axis in open align/seat stages; original approach heading retained",
            "model_receives_object_pose": False, "postcontact_lift_matches_depth_probe": True}
