"""A real Wall-OSS inference rollout in the reconstructed MuJoCo scene.

The policy receives RGB cameras and robot proprioception only. Object poses
are used exclusively for evaluation. No object attachment or grasp weld is
used. Optional gripper hold and relative program lift are explicitly labeled
hybrid control. The mobile base, lift, head and left arm are held fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


class GraspEnvironment:
    def __init__(self, scene: Path, config: Path, out: Path, color: str, motor_armature=1e-4, state_origin="arm-base", tcp="flange", controller_label="Wall-OSS policy", ready_pose=(.15, 0., .35)):
        self.cfg = json.loads(config.read_text())
        self.out = out
        self.color = color
        self.controller_label = controller_label
        self.state_origin = state_origin
        self.tcp = tcp
        self.tcp_offset = np.array([.132, 0., 0.]) if tcp == "grasp" else np.zeros(3)
        self.ready_pose = np.asarray(ready_pose, dtype=float)
        out.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(scene)
        root = tree.getroot()
        for parent in root.findall("equality"):
            for entry in list(parent):
                if entry.get("name", "").startswith("grasp_"):
                    parent.remove(entry)
        for parent in root.findall("contact"):
            for entry in list(parent):
                if "bottle_" in entry.get("body1", "") + entry.get("body2", ""):
                    parent.remove(entry)
        # Use the official wrist optical frame instead of an external observer.
        wrist = root.find(".//body[@name='right_arm_gripper_camera_color_frame']")
        if wrist is not None:
            ET.SubElement(wrist, "camera", name="policy_right_wrist", pos="0 0 0", quat="0 1 0 0", fovy="86")
        derived = out / "scene_no_weld.xml"
        tree.write(derived, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_path(str(derived))
        # The URDF motor link is a near-zero-inertia placeholder. Driving its
        # position servo without rotor inertia produces ~1e10 rad/s^2. This
        # numerical regularization affects that motor DOF only; finger/bottle
        # contacts and finger inertias are left unchanged. It is not calibrated.
        if motor_armature < 0:
            raise ValueError("Motor armature must be nonnegative")
        self.motor_armature = float(motor_armature)
        for side in ["left", "right"]:
            dof = self.m.jnt_dofadr[self.m.joint(f"{side}_arm_gripper").id]
            self.m.dof_armature[dof] = max(self.m.dof_armature[dof], self.motor_armature)
        self.d = mujoco.MjData(self.m)
        self.ikd = mujoco.MjData(self.m)
        self.state_offsets = {side: np.zeros(3) for side in ["left", "right"]}
        if state_origin == "zero-joint":
            reference = mujoco.MjData(self.m)
            mujoco.mj_forward(self.m, reference)
            for side in ["left", "right"]:
                base = self.m.body(f"{side}_arm_base_link").id
                ee = self.m.body(f"{side}_arm_gripper_base_link").id
                br = reference.xmat[base].reshape(3, 3)
                er = reference.xmat[ee].reshape(3, 3)
                self.state_offsets[side] = br.T @ (reference.xpos[ee]+er@self.tcp_offset-reference.xpos[base])
        self.acts = {self.m.joint(int(j)).name: i for i, j in enumerate(self.m.actuator_trnid[:, 0])}
        self.ctrl_joints = self.m.actuator_trnid[:, 0]
        self.ctrl_q = self.m.jnt_qposadr[self.ctrl_joints]
        self.control = np.zeros(self.m.nu)
        for name, val in [("base_x", self.cfg["robot"]["base_x"]), ("base_y", self.cfg["robot"]["source_base_y"]), ("lift_joint", .30), ("head_pitch_joint", .55)]:
            self.control[self.acts[name]] = val
        for side in ["left", "right"]:
            for i, val in enumerate([0., 1.4, -1.9, .5, 0., 0.], 1):
                self.control[self.acts[f"{side}_arm_joint{i}"]] = val
            self.set_gripper(side, 4.0)
        self.control[self.acts["left_arm_joint1"]] = 1.2
        self.d.qpos[self.ctrl_q] = self.control
        self.d.ctrl[:] = self.control
        self.frozen_acts = np.array([i for name, i in self.acts.items() if not name.startswith("right_arm_")])
        self.frozen_q = self.ctrl_q[self.frozen_acts]
        self.frozen_v = self.m.jnt_dofadr[self.ctrl_joints[self.frozen_acts]]
        self.joints = np.array([self.m.joint(f"right_arm_joint{i}").id for i in range(1, 7)])
        self.arm_q = self.m.jnt_qposadr[self.joints]
        self.arm_v = self.m.jnt_dofadr[self.joints]
        self.arm_acts = np.array([self.acts[f"right_arm_joint{i}"] for i in range(1, 7)])
        self.ee = self.m.body("right_arm_gripper_base_link").id
        self.arm_base = self.m.body("right_arm_base_link").id
        self.bottle = self.m.body("bottle_" + color).id
        self.fingers = {self.m.body("right_arm_gripper_left_link").id, self.m.body("right_arm_gripper_right_link").id}
        mujoco.mj_forward(self.m, self.d)
        # A generic ready pose relative to each shoulder. This reset uses no
        # bottle coordinates; all approach/grasp/lift motion comes from policy.
        for side in ["left", "right"]:
            joints = np.array([self.m.joint(f"{side}_arm_joint{i}").id for i in range(1, 7)])
            qa = self.m.jnt_qposadr[joints]
            base = self.m.body(f"{side}_arm_base_link").id
            ee = self.m.body(f"{side}_arm_gripper_base_link").id
            br = self.d.xmat[base].reshape(3, 3).copy()
            target = self.d.xpos[base].copy() + br @ self.ready_pose
            def residual(q):
                self.d.qpos[qa] = q
                mujoco.mj_forward(self.m, self.d)
                r = self.d.xmat[ee].reshape(3, 3)
                return np.r_[self.d.xpos[ee]-target, .15*Rotation.from_matrix(br@r.T).as_rotvec()]
            lo = self.m.jnt_range[joints, 0] + 1e-5
            hi = self.m.jnt_range[joints, 1] - 1e-5
            solution = least_squares(residual, np.clip(self.d.qpos[qa], lo, hi), bounds=(lo, hi), max_nfev=300)
            if np.linalg.norm(residual(solution.x)[:3]) > .02:
                raise RuntimeError("Generic ready pose is unreachable")
            for i, value in enumerate(solution.x, 1):
                self.control[self.acts[f"{side}_arm_joint{i}"]] = value
        self.d.qpos[self.ctrl_q] = self.control
        mujoco.mj_forward(self.m, self.d)
        self.renderer = mujoco.Renderer(self.m, height=448, width=448)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0
        self.telemetry = []
        self.frames = []
        self.held_seconds = 0.
        self.success = False
        self.max_lift = 0.
        self.min_grasp_distance = float("inf")
        self.contact_frames = 0
        self.step_time = .05
        self.step(.8, evaluate=False)
        self.initial_z = float(self.d.xpos[self.bottle, 2])
        self.start_time = float(self.d.time)
        self.capture("initial")

    def set_gripper(self, side, motor_angle):
        motor_angle = float(np.clip(motor_angle, 0., 4.5))
        self.control[self.acts[f"{side}_arm_gripper"]] = motor_angle
        # Exact mimic ratio from the official URDF (motor rad -> finger m).
        opening = .00852694 * motor_angle
        self.control[self.acts[f"{side}_arm_gripper_left_joint"]] = opening
        self.control[self.acts[f"{side}_arm_gripper_right_joint"]] = -opening

    def render(self, camera):
        self.renderer.update_scene(self.d, camera=camera, scene_option=self.opt)
        return self.renderer.render().copy()

    def arm_pose(self, side):
        base = self.m.body(f"{side}_arm_base_link").id
        ee = self.m.body(f"{side}_arm_gripper_base_link").id
        br = self.d.xmat[base].reshape(3, 3)
        p = br.T @ (self.d.xpos[ee] - self.d.xpos[base])
        r = br.T @ self.d.xmat[ee].reshape(3, 3)
        grip = self.d.qpos[self.m.joint(f"{side}_arm_gripper_left_joint").qposadr[0]] / .00852694
        p = p + r@self.tcp_offset - self.state_offsets[side]
        return np.r_[p, Rotation.from_matrix(r).as_euler("xyz"), grip]

    def observation(self):
        # Quanta X1 SDK state uses [pitch, yaw], matching training head_rotation.
        head = np.array([float(self.d.qpos[self.m.joint(name).qposadr[0]])
                         for name in ["head_pitch_joint", "head_yaw_joint"]])
        state = {"follow1_pos": self.arm_pose("left"), "follow2_pos": self.arm_pose("right"), "lift": np.array([.30]), "head_pos": head, "velocity_decomposed": np.zeros(3), "velocity_decomposed_odom": np.zeros(3)}
        wrist = "policy_right_wrist" if self.m.camera("policy_right_wrist").id >= 0 else "right_wrist_view"
        views = {"face_view": self.render("face_view")[None], "right_wrist_view": self.render(wrist)[None]}
        return state, views

    def solve_ik(self, local_pose):
        self.ikd.qpos[:] = self.d.qpos
        base_r = self.d.xmat[self.arm_base].reshape(3, 3)
        target_r = base_r @ Rotation.from_euler("xyz", local_pose[3:6]).as_matrix()
        target_p = self.d.xpos[self.arm_base] + base_r @ (local_pose[:3]+self.state_offsets["right"]) - target_r@self.tcp_offset
        initial_q = self.d.qpos[self.arm_q].copy()
        mujoco.mj_forward(self.m, self.ikd)
        # Bound only the command rate; target selection remains entirely model-driven.
        current_p = self.ikd.xpos[self.ee].copy()
        delta = target_p - current_p
        target_p = current_p + delta * min(1., .020 / max(np.linalg.norm(delta), 1e-9))
        cur_r = self.ikd.xmat[self.ee].reshape(3, 3)
        delta_r = Rotation.from_matrix(target_r @ cur_r.T).as_rotvec()
        target_r = Rotation.from_rotvec(delta_r * min(1., .12 / max(np.linalg.norm(delta_r), 1e-9))).as_matrix() @ cur_r
        jp = np.zeros((3, self.m.nv)); jr = np.zeros_like(jp)
        lo = self.m.jnt_range[self.joints, 0] + 1e-5
        hi = self.m.jnt_range[self.joints, 1] - 1e-5
        for _ in range(30):
            mujoco.mj_forward(self.m, self.ikd)
            ep = target_p - self.ikd.xpos[self.ee]
            er = Rotation.from_matrix(target_r @ self.ikd.xmat[self.ee].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(ep) < .001 and np.linalg.norm(er) < .015:
                break
            mujoco.mj_jacBody(self.m, self.ikd, jp, jr, self.ee)
            jac = np.vstack([jp[:, self.arm_v], .15 * jr[:, self.arm_v]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + .0001 * np.eye(6), np.r_[ep, .15 * er])
            self.ikd.qpos[self.arm_q] = np.clip(self.ikd.qpos[self.arm_q] + np.clip(dq, -.10, .10), lo, hi)
        q = initial_q + np.clip(self.ikd.qpos[self.arm_q] - initial_q, -.09, .09)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(delta))

    def finger_contacts(self):
        touched = set()
        for con in self.d.contact:
            a, b = (int(self.m.geom_bodyid[g]) for g in [con.geom1, con.geom2])
            if a == self.bottle and b in self.fingers:
                touched.add(b)
            if b == self.bottle and a in self.fingers:
                touched.add(a)
        return len(touched)

    def gripper_feedback(self):
        joints = [self.m.joint(f"right_arm_gripper_{side}_joint").id for side in ["left", "right"]]
        q = [float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in joints]
        force = [float(self.d.qfrc_actuator[self.m.jnt_dofadr[j]]) for j in joints]
        return {"opening_motor_units": [q[0]/.00852694, -q[1]/.00852694], "effort_n": force}

    def step(self, duration, evaluate=True):
        n = max(1, round(duration / self.m.opt.timestep))
        for _ in range(n):
            self.d.qpos[self.frozen_q] = self.control[self.frozen_acts]
            self.d.qvel[self.frozen_v] = 0
            self.d.ctrl[:] = self.control
            previous_time = float(self.d.time)
            mujoco.mj_step(self.m, self.d)
            if self.d.time <= previous_time:
                raise RuntimeError("MuJoCo reset simulation time after numerical instability; this trial is invalid")
            if not np.isfinite(self.d.qpos).all():
                raise RuntimeError("Non-finite simulator state")
            if evaluate:
                p = self.d.xpos[self.bottle]
                rz = abs(float(self.d.xmat[self.bottle].reshape(3, 3)[2, 2]))
                lowest = p[2] - self.cfg["bottle"]["height"] / 2 * rz - self.cfg["bottle"]["radius"] * np.sqrt(max(0., 1-rz*rz))
                lift = float(lowest - (self.cfg["table"]["height"] + .016))
                self.max_lift = max(self.max_lift, lift)
                contacts = self.finger_contacts()
                self.contact_frames += int(contacts == 2)
                held = lift >= .05 and contacts == 2
                self.held_seconds = self.held_seconds + self.m.opt.timestep if held else 0.
                self.success |= self.held_seconds >= 1.

    def execute(self, pose):
        if np.asarray(pose).shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Invalid model Cartesian action")
        base_r = self.d.xmat[self.arm_base].reshape(3, 3)
        target_r = base_r @ Rotation.from_euler("xyz", pose[3:6]).as_matrix()
        requested_flange = self.d.xpos[self.arm_base] + base_r @ (np.asarray(pose[:3])+self.state_offsets["right"]) - target_r@self.tcp_offset
        requested_tool = requested_flange + target_r@np.array([.132, 0., 0.])
        q, ik_error, requested_distance = self.solve_ik(pose)
        self.control[self.arm_acts] = q
        self.set_gripper("right", pose[6])
        self.step(self.step_time)
        grasp = self.d.site_xpos[self.m.site("right_tool").id]
        distance = float(np.linalg.norm(grasp - self.d.xpos[self.bottle]))
        self.min_grasp_distance = min(self.min_grasp_distance, distance)
        row = {"time_s": float(self.d.time-self.start_time), "bottle_xyz": self.d.xpos[self.bottle].tolist(), "right_ee": self.arm_pose("right").tolist(), "grasp_distance_m": distance, "finger_contacts": self.finger_contacts(), "ik_error_m": ik_error, "requested_step_m": requested_distance, "max_lift_m": self.max_lift, "held_seconds": self.held_seconds, "success": self.success}
        row.update(
            requested_pose=np.asarray(pose).tolist(),
            requested_tool_xyz=requested_tool.tolist(),
            actual_tool_xyz=grasp.tolist(),
            command_tracking_error_m=float(np.linalg.norm(requested_flange-self.d.xpos[self.ee])),
            bottle_tilt_deg=float(np.degrees(np.arccos(np.clip(abs(self.d.xmat[self.bottle].reshape(3, 3)[2, 2]), 0., 1.)))),
            gripper_feedback=self.gripper_feedback(),
        )
        self.telemetry.append(row)
        return row

    def capture(self, name=None):
        state, views = self.observation()
        overview = self.render("overview")
        montage = np.concatenate([overview, views["face_view"][0], views["right_wrist_view"][0]], axis=1)
        image = Image.fromarray(montage)
        draw = ImageDraw.Draw(image)
        title = f"{self.controller_label} | {self.color} bottle | NO GRASP WELD"
        draw.rectangle((0, 0, image.width, 30), fill=(20, 28, 26))
        draw.text((10, 8), title, fill="white")
        for x, text in [(10, "External evaluation camera"), (458, "Head camera"), (906, "Right wrist camera")]:
            draw.text((x, 35), text, fill=(255, 215, 80))
        draw.rectangle((0, image.height-24, image.width, image.height), fill=(20, 28, 26))
        draw.text((10, image.height-18), f"Sim time {self.d.time-self.start_time:.2f}s | max target lift {self.max_lift*100:.1f}cm | finger contacts {self.finger_contacts()} | success {self.success}", fill="white")
        self.frames.append(np.asarray(image))
        if name:
            image.save(self.out / f"{name}.png")
            for key, value in views.items():
                imageio.imwrite(self.out / f"{name}_{key}.png", value[0])
        return state, views

    def close(self):
        self.renderer.close()
        if self.frames:
            imageio.mimsave(self.out / "rollout.mp4", self.frames, fps=20, codec="libx264", macro_block_size=2)
        (self.out / "telemetry.json").write_text(json.dumps(self.telemetry, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT/"models/wall-oss-0.5")
    p.add_argument("--source", type=Path, default=ROOT/"vendor/wall-x")
    p.add_argument("--output", type=Path, default=ROOT/"runs/grasp_blue_seed0")
    p.add_argument("--color", default="blue", choices=["blue", "green", "yellow", "red"])
    p.add_argument("--chunks", type=int, default=32)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm-key", default="ex_normal")
    p.add_argument("--control-period", type=float, default=.05)
    p.add_argument("--scene", type=Path, default=ROOT/"generated/scene.xml")
    p.add_argument("--config", type=Path, default=PROJECT/"simulation/scene_config.json")
    p.add_argument("--state-origin", choices=["arm-base", "zero-joint"], default="arm-base")
    p.add_argument("--tcp", choices=["flange", "grasp"], default="flange")
    p.add_argument("--smooth-actions", action="store_true", help="Apply official runtime Laplacian smoothing, including gripper")
    p.add_argument("--ready-pose", type=float, nargs=3, default=[.15, 0., .35], metavar=("X", "Y", "Z"), help="Fixed reset flange position in each arm base; never relative to object pose")
    p.add_argument("--staged-grasp", action="store_true", help="Hold model close command on proprioceptive resistance and request model lift")
    p.add_argument("--program-lift", action="store_true", help="With staged grasp, use a labeled 12 cm relative program lift after handoff")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    if args.program_lift and not args.staged_grasp:
        p.error("--program-lift requires --staged-grasp")
    sys.path.insert(0, str(args.source))
    env = GraspEnvironment(args.scene, args.config, args.output, args.color, state_origin=args.state_origin, tcp=args.tcp, ready_pose=args.ready_pose)
    if args.control_period <= 0:
        raise ValueError("Control period must be positive")
    env.step_time = args.control_period
    result = {"model": "X-Square-Robot/wall-oss-0.5", "color": args.color, "seed": args.seed, "scene_config": env.cfg, "norm_key": "ex_normal", "policy_executed": False, "trial_valid": False, "success": False, "grasp_weld": False, "object_pose_given_to_policy": False, "fixed_components": ["base", "lift", "head", "left_arm"], "robot_state_frame": "individual arm_base_link", "end_effector_frame": "arm_gripper_base_link", "gripper_mapping": "official URDF 0.00852694 m/rad", "success_criterion": "bottle bottom >= 5 cm above tray, bilateral finger contact for >= 1 second"}
    raw_actions = []
    from grasp_stages import GraspStages
    stages = GraspStages() if args.staged_grasp else None
    result["control_mode"] = ("model_grasp_with_program_lift" if args.program_lift else
                              "model_arm_with_proprioceptive_gripper_hold" if stages else "direct_model")
    result["stage_events"] = []
    started = time.monotonic()
    try:
        if args.probe_only:
            print(json.dumps({"state": {k: v.tolist() for k, v in env.observation()[0].items()}, "output": str(args.output)}), flush=True)
            return
        import torch
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        from walloss_runtime import build_policy
        config = InferConfig(checkpoint_path=str(args.checkpoint), robot_type="ex001", norm_key=args.norm_key, cam_names=["face_view", "right_wrist_view"], model_device="cuda:0", robot_action_end_ratio=1., robot_action_interpolate_multiplier=1, smooth_action=args.smooth_actions, smooth_gripper=True)
        print("Loading official Wall-OSS checkpoint", flush=True)
        policy = build_policy(config)
        result["model_loaded"] = True
        result["norm_key"] = args.norm_key
        result["control_period_s"] = round(env.step_time/env.m.opt.timestep)*env.m.opt.timestep
        result["postprocessing"] = {"official_laplacian_smoothing": args.smooth_actions, "gripper_smoothing": args.smooth_actions}
        result["checkpoint_load_audit"] = policy.model_wrapper.load_audit
        result["coordinate_adapter"] = {"origin": env.state_origin, "tcp": env.tcp, "tcp_offset_m": env.tcp_offset.tolist(), "zero_offsets_m": {k:v.tolist() for k,v in env.state_offsets.items()}, "real_sdk_calibration_confirmed": False}
        result["end_effector_frame"] = "grasp center" if env.tcp == "grasp" else "arm_gripper_base_link"
        result["robot_state_frame"] = env.state_origin
        result["physics"] = {"gripper_motor_armature_kg_m2": env.motor_armature, "timestep_s": float(env.m.opt.timestep), "calibrated": False}
        result["initial_pose"] = {"flange_position_in_arm_base_m": env.ready_pose.tolist(), "rotation": "identity", "uses_object_pose": False}
        result["source_commit"] = "ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f"
        provenance = args.checkpoint/"download_provenance.json"
        result["weight_provenance"] = json.loads(provenance.read_text()) if provenance.exists() else {"repo": "X-Square-Robot/wall-oss-0.5", "revision": None, "verification": "not recorded"}
        result["gpu"] = torch.cuda.get_device_name(0)
        result["instruction"] = f"Use your right hand to pick up the {args.color} bottle from the tray and lift it."
        if stages:
            result["stage_controller"] = {"uses_object_pose": False, "uses_contact_identity": False, "scripted_arm_motion": args.program_lift, "blocked_duration_s": stages.required_seconds, "min_opening_motor_units": stages.min_opening, "min_opposing_effort_n": stages.min_effort_n}
        result["chunk_summaries"] = []
        for chunk_id in range(args.chunks):
            if args.program_lift and stages.phase == "lift":
                command = env.arm_pose("right").copy()
                command[2] += .12
                command[6] = stages.held_gripper
                result["program_lift"] = {"offset_in_arm_base_m": [0., 0., .12], "origin": "measured arm pose at handoff", "target_pose": command.tolist()}
                env.controller_label = "Wall-OSS grasp + PROGRAM LIFT"
                for _ in range(60):
                    row = env.execute(command)
                    row.update(raw_model_pose=None, arm_controller="program_relative_lift", stage="lift")
                    env.capture()
                    if env.success:
                        break
                break
            state, views = env.observation()
            instruction = (f"Lift the {args.color} bottle held in your right gripper upward. Keep holding it."
                           if stages and stages.phase == "lift" else result["instruction"])
            tick = time.monotonic()
            with torch.inference_mode():
                prediction = policy.infer({"state": state, "views": views, "instruction": instruction})
            elapsed = time.monotonic()-tick
            actions = np.asarray(prediction["follow2_pos"], dtype=float)
            raw_actions.append(prediction)
            (args.output/"actions.json").write_text(json.dumps(raw_actions))
            result["policy_executed"] = True
            for action in actions[:args.execute_steps]:
                command = action.copy()
                if stages:
                    command[6] = stages.gripper_command(action[6])
                row = env.execute(command)
                row["raw_model_pose"] = action.tolist()
                row["arm_controller"] = "model"
                changed = False
                if stages:
                    feedback = row["gripper_feedback"]
                    dt = round(env.step_time/env.m.opt.timestep)*env.m.opt.timestep
                    changed = stages.observe(action[6], feedback["opening_motor_units"], feedback["effort_n"], dt)
                    row["stage"] = stages.phase
                    if changed:
                        event = {"time_s": row["time_s"], "phase": stages.phase, "held_gripper": stages.held_gripper, "feedback": feedback}
                        result["stage_events"].append(event)
                        env.controller_label = "Wall-OSS arm + program grip hold"
                env.capture()
                if env.success or changed:
                    break
            summary = {"chunk": chunk_id, "inference_s": round(elapsed, 3), "action_shape": list(actions.shape), "instruction": instruction, **row}
            result["chunk_summaries"].append(summary)
            print(json.dumps(summary), flush=True)
            env.capture(f"chunk_{chunk_id:02d}")
            if env.success:
                break
        result.update(trial_valid=True, success=env.success, model_only_success=bool(env.success and not stages), chunks=len(raw_actions), simulated_seconds=float(env.d.time-env.start_time), max_bottle_bottom_lift_m=env.max_lift, min_grasp_distance_m=env.min_grasp_distance, bilateral_contact_steps=env.contact_frames, final_bottle_xyz=env.d.xpos[env.bottle].tolist(), peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/1e9)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-started
        env.capture("final")
        env.close()
        (args.output/"result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
