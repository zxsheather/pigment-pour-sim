"""Observe the existing simulator after each physics step without changing it.

The optional contact-latched gripper uses simulator contact identity. It is a
privileged diagnostic, not an autonomous-policy or hardware result. It holds
an already-issued model command; it adds no arm motion or stronger closure.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from grasp_walloss import GraspEnvironment
from feedback_trace import summarize_trace, write_trace


class PhysicsTrace:
    def __init__(self, contact_latched_gripper=False):
        self.rows = []
        self.contact_latched_gripper = contact_latched_gripper
        self.latched_gripper = None
        self.latch_event = None
        self.contact_seconds = 0.

    def command(self, raw, control_step, chunk):
        command = np.asarray(raw, dtype=float).copy()
        if self.latched_gripper is not None:
            command[6] = self.latched_gripper
        assert np.array_equal(command[:6], raw[:6])
        self.raw = np.asarray(raw, dtype=float).copy()
        self.applied = command.copy()
        self.requested_rotation = Rotation.from_euler("xyz", command[3:6])
        self.control_step, self.chunk = control_step, chunk
        return command

    def observe(self, env):
        dt = float(env.m.opt.timestep)
        contacts = env.finger_contacts()
        rz = abs(float(env.d.xmat[env.bottle].reshape(3, 3)[2, 2]))
        bottom = (float(env.d.xpos[env.bottle, 2]) - env.cfg["bottle"]["height"] / 2 * rz
                  - env.cfg["bottle"]["radius"] * math.sqrt(max(0., 1 - rz * rz)))
        lift = bottom - (env.cfg["table"]["height"] + .016)
        feedback = env.gripper_feedback()
        actual = env.arm_pose("right")
        actual_rotation = Rotation.from_euler("xyz", actual[3:6])
        self.contact_seconds = self.contact_seconds + dt if contacts == 2 else 0.
        row = {"time_s": float(env.d.time - env.start_time), "control_step": self.control_step,
            "chunk": self.chunk, "contacts": contacts, "lift_m": lift,
            "held_s": env.held_seconds, "bilateral_s": self.contact_seconds,
            "gripper_raw": float(self.raw[6]), "gripper_applied": float(self.applied[6]),
            "gripper_left": feedback["opening_motor_units"][0],
            "gripper_right": feedback["opening_motor_units"][1],
            "effort_left_n": feedback["effort_n"][0], "effort_right_n": feedback["effort_n"][1],
            "tracking_position_m": float(np.linalg.norm(self.applied[:3] - actual[:3])),
            "tracking_rotation_deg": float(np.degrees((self.requested_rotation * actual_rotation.inv()).magnitude())),
            "wrist_angle_deg": float(np.degrees(actual_rotation.magnitude())),
            "bottle_tilt_deg": float(np.degrees(np.arccos(np.clip(rz, 0., 1.))))}
        assert all(math.isfinite(value) for value in row.values())
        self.rows.append(row)
        # Latch only a closing model request after 80 ms of bilateral contact.
        # Observation is privileged, but its only intervention is to stop later
        # gripper-command changes. Arm coordinates and policy inputs stay intact.
        if (self.contact_latched_gripper and self.latched_gripper is None
                and self.contact_seconds >= .08 - 1e-9
                and self.applied[6] < min(feedback["opening_motor_units"]) - .1):
            self.latched_gripper = float(np.clip(self.applied[6], 0., 4.5))
            self.latch_event = {"time_s": row["time_s"], "gripper_command": self.latched_gripper,
                "bilateral_contact_s": self.contact_seconds,
                "takes_effect_at_next_control_step": True, "uses_simulator_contact_identity": True}

    def finish(self, folder, dt):
        result = summarize_trace(self.rows, dt)
        result["gripper_latch"] = self.latch_event
        write_trace(folder / "physics_trace.jsonl.gz", self.rows)
        return result


class DiagnosticEnvironment(GraspEnvironment):
    def __init__(self, *args, **kwargs):
        self.physics_trace = None
        super().__init__(*args, **kwargs)

    def step(self, duration, evaluate=True):
        if self.physics_trace is None or not evaluate:
            return super().step(duration, evaluate=evaluate)
        # Keep the base implementation's exact number and order of mj_step calls.
        count = max(1, round(duration / self.m.opt.timestep))
        for _ in range(count):
            super().step(float(self.m.opt.timestep), evaluate=True)
            self.physics_trace.observe(self)
