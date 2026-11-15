"""Run and optionally record the bottle-transport waypoint teacher."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from carry_env import CAMERAS, CONTROL_DT, PHYSICS_DT, ROOT, CarryEnv, WaypointTeacher, carry_instruction


def run_episode(env: CarryEnv, output: Path, *, color: str, goal: str,
                seed: int, jitter: float, record: bool, teacher_profile: str = "smooth",
                heading_jitter: float = 0.0) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    observation = env.reset(color=color, seed=seed, start_jitter=jitter, goal=goal,
                            heading_jitter=heading_jitter)
    teacher = WaypointTeacher(env, profile=teacher_profile)
    states, actions, poses, bottles, metric_rows = [], [], [], [], []
    metric_keys = ("position_error_m", "bottle_slip_m", "bottle_tilt_rad", "grip_contacts")
    writers = {}
    if record:
        for camera in (*CAMERAS, "overview"):
            writers[camera] = imageio.get_writer(output / f"{camera}.mp4", fps=20,
                                               codec="libx264", quality=7, macro_block_size=1)
    worst_slip = 0.0
    worst_tilt = 0.0
    minimum_contacts = 2
    try:
        while not env.done:
            states.append(observation["state"])
            poses.append(observation["base_pose"])
            bottles.append(observation["bottle_position"])
            current_metrics = env.metrics()
            metric_rows.append([current_metrics[key] for key in metric_keys])
            if record:
                for camera in CAMERAS:
                    writers[camera].append_data(observation["images"][camera])
                writers["overview"].append_data(env.render("carry_overview"))
            observation, info = env.step(teacher.action(observation))
            actions.append(env.action_vector())
            worst_slip = max(worst_slip, info["bottle_slip_m"])
            worst_tilt = max(worst_tilt, info["bottle_tilt_rad"])
            minimum_contacts = min(minimum_contacts, info["grip_contacts"])
    finally:
        for writer in writers.values():
            writer.close()
    np.savez_compressed(output / "trajectory.npz", state=np.asarray(states),
                        action=np.asarray(actions), base_pose=np.asarray(poses),
                        bottle_position=np.asarray(bottles),
                        metrics=np.asarray(metric_rows, dtype=np.float32),
                        timestamp=np.arange(len(states), dtype=np.float64) * CONTROL_DT)
    source_files = (ROOT / "carry_env.py", ROOT / "run_carry.py", ROOT / "scene_config.json")
    metadata = {
        "policy": "waypoint_teacher", "color": color, "goal": goal, "seed": seed,
        "teacher_profile": teacher_profile, "heading_jitter_rad": heading_jitter,
        "start_jitter_m": jitter, "frames": len(states), "fps": 20,
        "physics_timestep_s": PHYSICS_DT, "state_dim": 23, "action_dim": 23,
        "action_representation": "absolute base-frame EE position/rotation6D/gripper targets, body-frame velocity",
        "layout": ["left_position3", "left_rotation_rows6", "left_gripper1",
                   "right_position3", "right_rotation_rows6", "right_gripper1", "base_velocity3"],
        "instruction": carry_instruction(color, goal),
        "initialization": "Bottle placed between fingers at reset; grasp maintained by physical contact, no weld.",
        "base_model": "Planar velocity servos with body-forward/yaw command mapping; wheel traction unvalidated.",
        "recorded_cameras": list(CAMERAS) if record else [],
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "worst_slip_m": worst_slip, "worst_tilt_rad": worst_tilt,
        "minimum_grip_contacts": minimum_contacts, "result": info,
        "metrics_columns": list(metric_keys),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dict(output=str(output), color=color, goal=goal, seed=seed,
                success=info["success"], failure_reason=info["failure_reason"],
                seconds=info["simulated_seconds"], worst_slip_m=worst_slip,
                worst_tilt_rad=worst_tilt, position_error_m=info["position_error_m"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", choices=("blue", "red", "green", "yellow"), default="blue")
    parser.add_argument("--goal", choices=("A", "B"), default="B")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-jitter", type=float, default=0.0)
    parser.add_argument("--heading-jitter-deg", type=float, default=0.0)
    parser.add_argument("--teacher", choices=("smooth", "stop-turn"), default="smooth")
    parser.add_argument("--record", action="store_true", help="Record camera MP4s alongside state/action trajectories.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "generated/carry" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if args.episodes < 1 or not np.isfinite([args.start_jitter, args.heading_jitter_deg]).all() or min(args.start_jitter, args.heading_jitter_deg) < 0:
        parser.error("episodes must be positive; jitter ranges must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    env = CarryEnv(render=args.record, image_size=args.image_size)
    results = []
    try:
        for index in range(args.episodes):
            result = run_episode(env, args.output / f"episode_{index:04d}", color=args.color,
                                 goal=args.goal, seed=args.seed + index,
                                 jitter=args.start_jitter, record=args.record, teacher_profile=args.teacher,
                                 heading_jitter=float(np.deg2rad(args.heading_jitter_deg)))
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    summary = {"episodes": results, "successes": sum(item["success"] for item in results),
               "total": len(results), "policy": "waypoint_teacher", "teacher_profile": args.teacher}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not all(item["success"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
