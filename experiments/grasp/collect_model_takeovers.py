"""Replay training-probe model prefixes, then record teacher corrections only.

No physics state is teleported. Prefix identity is checked against recorded
telemetry before collection. This pilot never reads external evaluation cases.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from collect_expert import FEATURE_ORDER, state_vector
from grasp_walloss import GraspEnvironment
from reactive_expert import apply_setup, reset_recording
from takeover_expert import bottle_tilt_degrees, make_takeover_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=["reactive", "axis", "endon"], default="axis")
    parser.add_argument("--prefix-steps", type=int, nargs="+", default=[8, 24, 48, 96, 192, 384])
    parser.add_argument("--max-seconds", type=float, default=12.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source").mkdir()
    for name in ["collect_model_takeovers.py", "takeover_expert.py", "reactive_expert.py"]:
        (args.output / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report = json.loads((args.evaluation / "report.json").read_text())
    assert report["completed"] and report["control_mode"] == "direct_model"
    probes = [case for case in report["cases"] if case["split"] == "train_episode"]
    assert len(probes) == 2 and all(case["trial_valid"] for case in probes)
    source_hash = hashlib.sha256(Path(__file__).with_name("takeover_expert.py").read_bytes()
                                 + Path(__file__).with_name("reactive_expert.py").read_bytes()).hexdigest()
    manifest = {"controller": "privileged model-state takeover teacher", "teacher": args.teacher,
        "model_executed": False, "model_prefix_before_recording": True,
        "fps": 25, "observation_width": 20, "action_width": 20,
        "feature_order": FEATURE_ORDER, "object_pose_in_observation": False,
        "grasp_weld": False, "setup_commands_in_supervision": False,
        "source_split": "train_episode only", "teacher_source_sha256": source_hash,
        "used_for_xyz_normalization_training": False, "episodes": [], "accepted_episodes": 0}
    for case in probes:
        source = args.runs / case["batch"] / f"episode_{case['episode']:04d}"
        recorded = args.evaluation / case["directory"]
        chunks = np.asarray(json.loads((recorded / "actions.json").read_text()))
        commands = chunks[:, :8].reshape(-1, 7)
        expected_rows = json.loads((recorded / "telemetry.json").read_text())
        metadata = json.loads((source / "episode.json").read_text())
        for prefix in args.prefix_steps:
            assert 1 <= prefix <= len(commands)
            index = len(manifest["episodes"])
            folder = args.output / f"episode_{index:04d}"
            folder.mkdir(parents=True)
            (folder / "input_scene.xml").write_bytes((source / "input_scene.xml").read_bytes())
            env = GraspEnvironment(folder / "input_scene.xml", args.config, folder, "blue", tcp="grasp",
                controller_label="RECORDED MODEL PREFIX - NOT SUPERVISION")
            env.step_time = .04
            entry = {"episode": index, "kind": "model_state_takeover", "accepted": False,
                "numerically_valid": False, "source_batch": case["batch"],
                "source_episode": case["episode"], "prefix_steps": prefix,
                "source_evaluation": str(args.evaluation.relative_to(args.runs)),
                "offset_xy_m": metadata["offset_xy_m"], "source_split": "train_episode"}
            states, labels, times, phases = [], [], [], []
            try:
                apply_setup(env, metadata["setup"])
                for command in commands[:prefix]:
                    env.execute(command)
                    env.capture()
                expected = expected_rows[prefix - 1]
                state_error = float(np.max(np.abs(env.arm_pose("right") - expected["right_ee"])))
                bottle_error = float(np.max(np.abs(env.d.xpos[env.bottle] - expected["bottle_xyz"])))
                assert max(state_error, bottle_error) < 1e-6, "Model prefix did not reproduce its recorded state"
                imageio.mimsave(folder / "student_prefix.mp4", env.frames, fps=20, codec="libx264", macro_block_size=2)
                entry.update(prefix_state_error=state_error, prefix_bottle_error=bottle_error,
                             initial_bottle_tilt_deg=bottle_tilt_degrees(env),
                             initial_robot_state=state_vector(env.observation()[0]).tolist())
                env.controller_label = "TAKEOVER TEACHER - NOT MODEL"
                reset_recording(env)
                teacher = make_takeover_teacher(env, args.teacher)
                entry["teacher_class"] = type(teacher).__name__
                for camera in ["face_view", "right_wrist_view"]:
                    (folder / camera).mkdir()
                for frame in range(round(args.max_seconds * 25)):
                    state, views = env.observation()
                    before = state_vector(state)
                    command, phase = teacher.command()
                    label = before.copy()
                    label[7:14] = command
                    for camera, array in views.items():
                        Image.fromarray(array[0]).save(folder / camera / f"{frame:06d}.jpg", quality=90)
                    states.append(before); labels.append(label)
                    times.append(float(env.d.time - env.start_time)); phases.append(phase)
                    row = env.execute(command); row["expert_phase"] = phase
                    env.capture()
                    if env.success:
                        break
                entry.update(accepted=bool(env.success), numerically_valid=True, frames=len(states),
                    simulated_seconds=float(env.d.time - env.start_time),
                    max_bottle_bottom_lift_m=env.max_lift,
                    max_held_seconds=max((r["held_seconds"] for r in env.telemetry), default=0.),
                    bilateral_contact_steps=env.contact_frames,
                    corrective_realignments=teacher.recoveries,
                    phase_counts={phase: phases.count(phase) for phase in sorted(set(phases))})
            except (RuntimeError, AssertionError) as error:
                entry["error"] = str(error)
            finally:
                if states:
                    np.savez_compressed(folder / "trajectory.npz", state=np.asarray(states),
                        action=np.asarray(labels), timestamp_s=np.asarray(times), phase=np.asarray(phases))
                env.capture("final")
                env.close()
            (folder / "episode.json").write_text(json.dumps(entry, indent=2))
            manifest["episodes"].append(entry)
            manifest["accepted_episodes"] = sum(e["accepted"] for e in manifest["episodes"])
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
