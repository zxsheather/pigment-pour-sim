"""Collect synchronized simulator expert demonstrations, not model rollouts.

The privileged expert uses object pose to plan successful trajectories. Object
pose is saved only in episode metadata/evaluation, never in observation.state.
Each row pairs the BEFORE-action RGB/state with the issued Cartesian target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from grasp_walloss import GraspEnvironment


ROOT = Path(__file__).resolve().parent
FEATURE_ORDER = ["left_xyz_rpy_gripper:7", "right_xyz_rpy_gripper:7", "base_velocity:3", "lift:1", "head_pitch_yaw:2"]


def state_vector(state):
    return np.concatenate([state["follow1_pos"], state["follow2_pos"], state["velocity_decomposed_odom"], state["lift"], state["head_pos"]]).astype(np.float32)


def collect(scene, config, output, episodes, seed, radius, fps, include_center=True):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    manifest = {"controller": "privileged scripted expert", "model_executed": False,
                "fps": fps, "observation_width": 20, "action_width": 20,
                "feature_order": FEATURE_ORDER, "rotation": "extrinsic xyz Euler, radians",
                "action_semantics": "absolute arm-base grasp-center Cartesian targets; gripper motor units",
                "object_pose_in_observation": False, "grasp_weld": False,
                "success_criterion": "bottom lift >= 5 cm and bilateral contact continuously >= 1 s",
                "episodes": []}
    for episode in range(episodes):
        folder = output/f"episode_{episode:04d}"
        folder.mkdir()
        offset = np.zeros(2) if episode == 0 and include_center else rng.uniform(-radius, radius, 2)
        tree = ET.parse(scene)
        body = tree.getroot().find("worldbody/body[@name='bottle_blue']")
        pos = np.fromstring(body.get("pos"), sep=" ")
        pos[:2] += offset
        body.set("pos", " ".join(map(str, pos)))
        derived = folder/"input_scene.xml"
        tree.write(derived, encoding="unicode")
        env = GraspEnvironment(derived, config, folder, "blue", tcp="grasp", controller_label="PRIVILEGED EXPERT DEMO - NOT MODEL")
        env.step_time = 1/fps
        steps = round(env.step_time/env.m.opt.timestep)
        if not np.isclose(steps*env.m.opt.timestep, env.step_time, atol=1e-10):
            env.renderer.close()
            raise ValueError("Requested sample period must be an exact multiple of physics timestep")
        states, actions, times, phases = [], [], [], []
        image_dirs = {key: folder/key for key in ["face_view", "right_wrist_view"]}
        for path in image_dirs.values():
            path.mkdir()
        report = {"episode": episode, "seed": seed, "offset_xy_m": offset.tolist(),
                  "accepted": False, "numerically_valid": False, "object_pose_for_expert_only": True}
        try:
            bottle = env.d.xpos[env.bottle].copy()
            plan = [("approach", bottle+[-.10, 0., .025], 4.5, 1.68),
                    ("align", bottle+[0., 0., .005], 4.5, 1.20),
                    ("close", bottle+[0., 0., .005], 2.2, .72),
                    ("lift_hold", bottle+[0., 0., .12], 2.2, 2.40)]
            for phase, target_world, grip, duration in plan:
                br = env.d.xmat[env.arm_base].reshape(3, 3)
                target = br.T@(target_world-env.d.xpos[env.arm_base])
                command = np.r_[target, np.zeros(3), grip]
                for _ in range(round(duration*fps)):
                    state, views = env.observation()
                    before = state_vector(state)
                    action = before.copy()
                    action[7:14] = command
                    frame = len(states)
                    for key, array in views.items():
                        Image.fromarray(array[0]).save(image_dirs[key]/f"{frame:06d}.jpg", quality=90)
                    states.append(before)
                    actions.append(action)
                    times.append(float(env.d.time-env.start_time))
                    phases.append(phase)
                    env.execute(command)
                    env.capture()
            report.update(accepted=bool(env.success), numerically_valid=True,
                          frames=len(states), simulated_seconds=float(env.d.time-env.start_time),
                          max_bottle_bottom_lift_m=env.max_lift, bilateral_contact_steps=env.contact_frames)
            np.savez_compressed(folder/"trajectory.npz", state=np.asarray(states), action=np.asarray(actions),
                                timestamp_s=np.asarray(times), phase=np.asarray(phases))
            assert np.allclose(np.diff(times), 1/fps, atol=1e-8)
            assert np.asarray(states).shape == (len(times), 20)
            assert np.asarray(actions).shape == (len(times), 20)
        except (RuntimeError, AssertionError) as error:
            report["error"] = str(error)
        finally:
            env.capture("final")
            env.close()
        (folder/"episode.json").write_text(json.dumps(report, indent=2))
        manifest["episodes"].append(report)
        manifest["accepted_episodes"] = sum(x["accepted"] for x in manifest["episodes"])
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(report), flush=True)
        del env
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=ROOT/"generated/blue_only/scene.xml")
    p.add_argument("--config", type=Path, default=ROOT/"generated/blue_only/scene_config.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--radius", type=float, default=.02)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--no-center", action="store_true", help="Randomize every episode, including episode zero")
    args = p.parse_args()
    collect(args.scene, args.config, args.output, args.episodes, args.seed, args.radius, args.fps, not args.no_center)


if __name__ == "__main__":
    main()
