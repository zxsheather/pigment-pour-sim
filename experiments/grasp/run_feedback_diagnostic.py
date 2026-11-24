"""Matched 8/2-step feedback and isolated contact-latched gripper diagnostics.

Physics is paused during inference, as in the preceding evaluation. These
are ideal synchronous simulation diagnostics, not real-time robot results.
"""
from pathlib import Path
import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "sdpa")

import numpy as np

from diagnostic_environment import DiagnosticEnvironment, PhysicsTrace
from reactive_expert import apply_setup


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("experiments/grasp/runs/takeover_training_eval_v6/augmented_closedloop_internal"))
    parser.add_argument("--execute-steps", type=int, choices=[2, 8], default=8)
    parser.add_argument("--max-control-steps", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contact-latched-gripper", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    assert args.max_control_steps > 0
    if not args.replay_only:
        assert args.source is not None and args.checkpoint is not None
    else:
        assert not args.contact_latched_gripper
    args.output.mkdir(parents=True, exist_ok=False)
    probes = read(args.cases)
    reference = read(args.reference / "report.json")
    assert len(probes) == 4 and reference["completed"]
    mode = ("recorded_action_replay" if args.replay_only else "privileged_contact_latched_gripper"
            if args.contact_latched_gripper else "direct_model")
    report = {"completed": False, "model": "Wall-OSS-0.5", "control_mode": mode,
        "execute_steps": args.execute_steps, "max_control_steps": args.max_control_steps,
        "max_simulated_seconds": args.max_control_steps * .04, "control_period_s": .04,
        "inference_seed": args.seed, "development_cases_only": True,
        "physics_paused_during_inference": True,
        "hardware_realtime_performance_tested": False,
        "policy_receives_object_pose_or_contacts": False,
        "assistance_uses_simulator_contact_identity": args.contact_latched_gripper,
        "grasp_weld": False, "cases": []}
    if not args.replay_only:
        sys.path.insert(0, str(args.source.resolve()))
        import torch
        from walloss_runtime import build_policy
        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        torch.manual_seed(args.seed)
        config = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
            norm_key="local/pigment_grasp_reactive_v4", cam_names=["face_view", "right_wrist_view"],
            model_device="cuda:0", robot_action_end_ratio=1.,
            robot_action_interpolate_multiplier=1, smooth_action=False)
        policy = build_policy(config)
        report["load_audit"] = policy.model_wrapper.load_audit
    for probe in probes:
        source = args.runs / probe["batch"] / f"episode_{probe['episode']:04d}"
        name = f"{probe['split']}_{probe['batch']}_e{probe['episode']:04d}"
        output = args.output / name
        label = ("RECORDED ACTION REPLAY" if args.replay_only else "PRIVILEGED GRIPPER HOLD DIAGNOSTIC"
                 if args.contact_latched_gripper else f"WALL-OSS MODEL | EXECUTE {args.execute_steps}")
        env = DiagnosticEnvironment(source / "input_scene.xml", args.config, output, "blue",
                                    tcp="grasp", controller_label=label)
        env.step_time = .04
        meta = read(source / "episode.json")
        apply_setup(env, meta["setup"])
        trace = PhysicsTrace(args.contact_latched_gripper)
        env.physics_trace = trace
        case = {**probe, "directory": name, "trial_valid": False, "success": False,
                "control_mode": mode, "control_steps": 0}
        actions_log, inference_log = [], []
        started = time.monotonic()
        try:
            if args.replay_only:
                original_case = next(c for c in reference["cases"] if c["batch"] == probe["batch"]
                                     and c["episode"] == probe["episode"] and c["split"] == probe["split"])
                recorded = args.reference / original_case["directory"]
                commands = np.asarray(read(recorded / "actions.json"))[:, :8].reshape(-1, 7)
                expected = read(recorded / "telemetry.json")
                assert len(commands) == len(expected) == args.max_control_steps
                errors = {key: 0. for key in ["right_ee", "bottle_xyz", "held_seconds"]}
                for step, command in enumerate(commands):
                    applied = trace.command(command, step, step // 8)
                    row = env.execute(applied)
                    for key in errors:
                        error = float(np.max(np.abs(np.asarray(row[key]) - expected[step][key])))
                        errors[key] = max(errors[key], error)
                assert max(errors.values()) < 1e-6, "Observation changed the original physics trajectory"
                assert bool(env.success) == original_case["success"]
                case["replay_max_abs_errors"] = errors
                case["control_steps"] = len(commands)
            else:
                policy.reset()
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                for chunk in range(math.ceil(args.max_control_steps / args.execute_steps)):
                    state, views = env.observation()
                    observed_at = float(env.d.time - env.start_time)
                    torch.cuda.synchronize()
                    inference_start = time.monotonic()
                    with torch.inference_mode():
                        prediction = policy.infer({"state": state, "views": views,
                            "instruction": "Use your right hand to pick up the blue bottle from the tray and lift it."})
                    torch.cuda.synchronize()
                    latency = time.monotonic() - inference_start
                    actions = np.asarray(prediction["follow2_pos"], dtype=float)
                    assert actions.shape == (32, 7) and np.isfinite(actions).all()
                    actions_log.append(actions.tolist())
                    count = min(args.execute_steps, args.max_control_steps - case["control_steps"])
                    inference_log.append({"chunk": chunk, "observation_time_s": observed_at,
                        "inference_wall_seconds": latency, "control_steps": count,
                        "nominal_action_window_s": count * .04})
                    for raw in actions[:count]:
                        applied = trace.command(raw, case["control_steps"], chunk)
                        row = env.execute(applied)
                        row.update(raw_model_pose=raw.tolist(), arm_controller=mode, inference_chunk=chunk)
                        case["control_steps"] += 1
                        env.capture()
                        if env.success:
                            break
                    if env.success:
                        break
                assert env.success or case["control_steps"] == args.max_control_steps
            case.update(trial_valid=True, success=bool(env.success),
                simulated_seconds=float(env.d.time - env.start_time),
                min_grasp_distance_m=env.min_grasp_distance,
                max_bottle_bottom_lift_m=env.max_lift,
                sampled_max_held_seconds=max(r["held_seconds"] for r in env.telemetry),
                bilateral_contact_physics_steps=env.contact_frames)
            case["diagnostics"] = trace.finish(output, float(env.m.opt.timestep))
            if inference_log:
                latencies = [r["inference_wall_seconds"] for r in inference_log]
                case["inference"] = {"calls": len(latencies), "median_seconds": statistics.median(latencies),
                    "max_seconds": max(latencies),
                    "calls_slower_than_action_window": sum(r["inference_wall_seconds"] > r["nominal_action_window_s"] for r in inference_log)}
        except Exception as error:
            case["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        finally:
            case["wall_seconds"] = time.monotonic() - started
            env.capture("final")
            env.close()
            (output / "actions.json").write_text(json.dumps(actions_log))
            (output / "inferences.json").write_text(json.dumps(inference_log))
            (output / "result.json").write_text(json.dumps(case, indent=2))
        report["cases"].append(case)
        report["valid_trials"] = sum(c["trial_valid"] for c in report["cases"])
        report["successes"] = sum(c["success"] for c in report["cases"])
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print("FEEDBACK_CASE " + json.dumps({k: v for k, v in case.items() if k != "diagnostics"}), flush=True)
    report["completed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    assert report["valid_trials"] == len(probes), "Some diagnostics were invalid"


if __name__ == "__main__":
    main()
