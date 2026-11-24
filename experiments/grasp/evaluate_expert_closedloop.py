"""Evaluate one checkpoint across fixed episode positions, loading it only once.

Default probes are two training-pool positions and three unused internal
positions. Use --external-holdout only after selecting a checkpoint. No object
state is passed to the policy. Explicit recovery cases replay their recorded
physical setup first; every subsequent action comes from the model.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from evaluate_expert_openloop import PROBES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    p.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-key", default="local/pigment_grasp_train")
    p.add_argument("--chunks", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--external-holdout", action="store_true")
    p.add_argument("--cases",type=Path,help="JSON list of split/batch/episode cases; replay recorded setup before policy control")
    p.add_argument("--fixed-tool-orientation", action="store_true",
                   help="Separate assisted diagnostic: replace model RPY with zero in the arm-base frame")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from grasp_walloss import GraspEnvironment
    from walloss_runtime import build_policy
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    config = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
                         norm_key=args.norm_key, cam_names=["face_view", "right_wrist_view"],
                         model_device="cuda:0", robot_action_end_ratio=1.,
                         robot_action_interpolate_multiplier=1, smooth_action=False)
    torch.manual_seed(args.seed)
    policy = build_policy(config)
    probes = ([(c["split"],c["batch"],c["episode"]) for c in json.loads(args.cases.read_text())]
              if args.cases else [("external_holdout", "expert_heldout_v1", i) for i in range(8)]
              if args.external_holdout else PROBES)
    cases = []
    mode = "model_position_gripper_fixed_orientation" if args.fixed_tool_orientation else "direct_model"
    report = {"model": "X-Square-Robot/wall-oss-0.5", "control_mode": mode,
              "fixed_tool_orientation_rpy": [0., 0., 0.] if args.fixed_tool_orientation else None,
              "checkpoint_name": args.checkpoint.parent.parent.name + "/" + args.checkpoint.name,
              "external_holdout_used": args.external_holdout, "control_period_s": .04,
              "execute_steps": 8, "seed": args.seed, "load_audit": policy.model_wrapper.load_audit,
              "object_pose_given_to_policy": False, "grasp_weld": False, "cases": cases}
    for split, batch, episode in probes:
        source = args.runs/batch/f"episode_{episode:04d}"
        case_name = f"{split}_{batch}_e{episode:04d}"
        output = args.output/case_name
        env = GraspEnvironment(source/"input_scene.xml", args.config, output, "blue", tcp="grasp",
            controller_label="MODEL + FIXED ORIENTATION" if args.fixed_tool_orientation else "WALL-OSS MODEL")
        env.step_time = .04
        episode_meta=json.loads((source/"episode.json").read_text())
        setup=episode_meta.get("setup")
        prefix_initialization = None
        if episode_meta.get("kind") == "model_state_takeover":
            from replay_takeover_start import replay_takeover_start
            assert not args.fixed_tool_orientation, "Prefix recovery probes use unmodified model actions"
            env.controller_label = "MODEL FROM SEEN CORRECTION START"
            prefix_initialization = replay_takeover_start(env, episode_meta, args.runs)
        elif setup:
            from reactive_expert import apply_setup
            apply_setup(env,setup)
        policy.reset()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        started = time.monotonic()
        actions_log = []
        case = {"split": split, "batch": batch, "episode": episode, "directory": case_name,
                "trial_valid": False, "success": False,
                "initialization_kind": "recorded_model_prefix" if prefix_initialization else setup.get("kind") if setup else "clean",
                "scripted_initialization_unrecorded":bool(prefix_initialization or (setup and setup.get("kind")!="clean")),
                "recorded_prefix_initialization": prefix_initialization,
                "task": "in_sample_prefix_recovery" if prefix_initialization else "slip_recovery" if setup and setup.get("kind")=="slip" else "tabletop_pickup"}
        try:
            for chunk in range(args.chunks):
                state, views = env.observation()
                with torch.inference_mode():
                    prediction = policy.infer({"state": state, "views": views,
                        "instruction": "Use your right hand to pick up the blue bottle from the tray and lift it."})
                actions = np.asarray(prediction["follow2_pos"], dtype=float)
                if actions.shape != (32, 7) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid action chunk: {actions.shape}")
                actions_log.append(actions.tolist())
                for action in actions[:8]:
                    command = action.copy()
                    if args.fixed_tool_orientation:
                        command[3:6] = 0.
                    row = env.execute(command)
                    row.update(raw_model_pose=action.tolist(),
                               arm_controller=mode if args.fixed_tool_orientation else "model")
                    env.capture()
                    if env.success:
                        break
                if env.success:
                    break
            case.update(trial_valid=True, success=env.success, chunks=len(actions_log),
                        simulated_seconds=float(env.d.time-env.start_time),
                        min_grasp_distance_m=env.min_grasp_distance,
                        max_bottle_bottom_lift_m=env.max_lift,
                        max_held_seconds=max(r["held_seconds"] for r in env.telemetry),
                        bilateral_contact_steps=env.contact_frames)
        except Exception as error:
            case["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        finally:
            case["wall_seconds"] = time.monotonic() - started
            env.capture("final")
            env.close()
            (output/"actions.json").write_text(json.dumps(actions_log))
            (output/"result.json").write_text(json.dumps(case, indent=2))
        cases.append(case)
        report["aggregates"] = {split_name: {
            "trials": sum(c["split"] == split_name for c in cases),
            "valid": sum(c["trial_valid"] for c in cases if c["split"] == split_name),
            "successes": sum(c["success"] for c in cases if c["split"] == split_name)}
            for split_name in sorted({c["split"] for c in cases})}
        (args.output/"report.json").write_text(json.dumps(report, indent=2))
        print("CLOSED_LOOP_CASE " + json.dumps(case), flush=True)
    report["completed"] = True
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregates"]), flush=True)


if __name__ == "__main__":
    main()
