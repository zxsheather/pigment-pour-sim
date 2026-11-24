"""Re-infer approaches at one fixed start with matched insertion and lift control.

The policy is frozen and sees robot state/RGB only. Grasp confirmation still
uses simulator contact identity; these are hybrid-system development tests,
not model-only or real-robot results. Physics pauses during inference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from diagnostic_environment import DiagnosticEnvironment
from grasp_depth_probe import SeatingTrace, hand_geometry, closeup, read
from reactive_expert import apply_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/grasp_depth_v8"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--depths-mm", type=int, nargs="+",
                        help="Explicit additional development depths; default is baseline and replay-selected depth")
    parser.add_argument("--tag", default="live", choices=["live", "live_extended"])
    args = parser.parse_args()
    evidence = read(args.output / "lift_report.json")
    assert evidence["completed"] and len(evidence["cases"]) == 12
    scores = {depth: sum(c["end_bilateral_at_least_3s"] for c in evidence["cases"] if c["depth_mm"] == depth)
              for depth in [0, 10, 20, 30]}
    candidates = [d for d in [10, 20, 30] if scores[d] == max(scores.values())]
    assert candidates and max(scores.values()) > scores[0], "No replay improvement to validate"
    selected = min(candidates)
    depths = args.depths_mm if args.depths_mm is not None else [0, selected]
    assert len(set(depths)) == len(depths) and all(0 <= d <= 30 for d in depths)
    protocol = {"selected_depth_mm": selected, "selection": "smallest positive depth with best end-3s replay count",
        "development_replay_scores": scores, "seeds": args.seeds, "depths_mm": depths,
        "additional_development_screen": args.depths_mm is not None,
        "fixed_scene_only": True, "new_model_inference": True, "training": False,
        "same_program_controller_all_groups": True, "program_uses_simulator_contact_identity": True,
        "policy_observation_object_pose_or_contacts": False,
        "execute_steps": 8, "max_control_steps": 384, "physics_paused_during_inference": True,
        "warmup_per_trial": 1, "warmup_discarded_then_policy_and_seed_reset": True,
        "after_grasp": "no more policy inference; settle .2s, minimum-jerk world-up 12cm over 2s, hold pose/gripper2.2",
        "raw_reference_after_grasp": "last cached proposal only, not newly inferred model output",
        "not_model_only_successes": True, "not_new_position_generalization": True}
    (args.output / (args.tag + "_protocol.json")).write_text(json.dumps(protocol, indent=2))
    root = args.output / args.tag
    root.mkdir(exist_ok=False)
    import sys
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from walloss_runtime import build_policy
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    torch.manual_seed(0)
    config = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
        norm_key="local/pigment_grasp_reactive_v4", cam_names=["face_view", "right_wrist_view"],
        model_device="cuda:0", robot_action_end_ratio=1., robot_action_interpolate_multiplier=1, smooth_action=False)
    policy = build_policy(config)
    report = {"completed": False, "protocol": protocol, "load_audit": policy.model_wrapper.load_audit, "cases": []}
    source = args.runs / "reactive_train_v4/episode_0000"
    instruction = "Use your right hand to pick up the blue bottle from the tray and lift it."
    for seed in args.seeds:
        order = depths if seed % 2 == 0 else list(reversed(depths))
        for depth_mm in order:
            folder = root / f"seed_{seed}_depth_{depth_mm:02d}mm"
            env = DiagnosticEnvironment(source / "input_scene.xml", args.config, folder, "blue", tcp="grasp",
                controller_label=f"MODEL + {depth_mm}mm INSERTION + PRIVILEGED PROGRAM LIFT")
            env.step_time = .04
            apply_setup(env, read(source / "episode.json")["setup"])
            geometry, bodies = hand_geometry(env)
            lift_axis = env.d.xmat[env.arm_base].reshape(3, 3).T @ np.array([0., 0., 1.])
            trace = SeatingTrace(depth_mm / 1000, geometry, bodies, lift_axis)
            state, views = env.observation()
            initial = {"state": {k: np.asarray(v).tolist() for k, v in state.items()},
                "camera_sha256": {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in views.items()}}
            policy.reset(); torch.manual_seed(seed); np.random.seed(seed)
            with torch.inference_mode():
                policy.infer({"state": state, "views": views, "instruction": instruction})
            torch.cuda.synchronize()
            policy.reset(); torch.manual_seed(seed); np.random.seed(seed)
            env.physics_trace = trace
            actions, inferences, frames = [], [], []
            current = None
            chunk_start = 0
            case = {"seed": seed, "depth_mm": depth_mm, "directory": str(folder.relative_to(args.output)),
                    "trial_valid": False, "initial_observation": initial}
            try:
                for step in range(384):
                    active = trace.handoff is None
                    if active and step % 8 == 0:
                        state, views = env.observation()
                        torch.cuda.synchronize(); started = time.monotonic()
                        with torch.inference_mode():
                            result = policy.infer({"state": state, "views": views, "instruction": instruction})
                        torch.cuda.synchronize()
                        current = np.asarray(result["follow2_pos"], dtype=float)
                        assert current.shape == (32, 7) and np.isfinite(current).all()
                        actions.append(current.tolist()); chunk_start = step
                        inferences.append({"control_step": step, "wall_seconds": time.monotonic() - started})
                    raw = current[min(step - chunk_start, 31)]
                    command = trace.command(raw, step, len(actions) - 1)
                    row = env.execute(command)
                    row.update(controller_phase="model_approach" if active else "program_lift_and_hold",
                               raw_model_pose=raw.tolist() if active else None)
                    env.capture()
                    image = Image.fromarray(closeup(env, depth_mm))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 28), fill=(20, 28, 26))
                    draw.text((8, 8), f"MODEL + {depth_mm}mm + PRIVILEGED PROGRAM | SEED {seed}", fill="white")
                    frames.append(np.asarray(image))
                diagnostic = trace.finish(folder, float(env.m.opt.timestep))
                final = trace.rows[-1]
                case.update(trial_valid=True, original_criterion_reached=bool(env.success),
                    max_bilateral_lifted_s=diagnostic["max_bilateral_lifted_s"],
                    end_bilateral_lifted_s=final["held_s"], end_retained_s=final["retained_s"],
                    end_bilateral_at_least_3s=final["held_s"] >= 3.,
                    end_retention_at_least_3s=final["retained_s"] >= 3.,
                    final_lift_m=final["lift_m"], final_finger_contacts=final["contacts"],
                    handoff=trace.handoff, inference_calls=len(inferences), diagnostics=diagnostic)
            finally:
                env.capture("final"); env.close()
                if frames:
                    imageio.mimsave(folder / "closeup.mp4", frames, fps=25, codec="libx264", macro_block_size=2)
            (folder / "actions.json").write_text(json.dumps(actions))
            (folder / "inferences.json").write_text(json.dumps(inferences))
            (folder / "result.json").write_text(json.dumps(case, indent=2))
            report["cases"].append(case)
            (args.output / (args.tag + "_report.json")).write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in case.items() if k not in ["initial_observation", "diagnostics", "handoff"]}), flush=True)
    report["completed"] = True
    (args.output / (args.tag + "_report.json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
