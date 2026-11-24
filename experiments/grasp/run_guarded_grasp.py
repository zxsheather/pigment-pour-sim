"""Paired guarded-preclosure versus unchanged contact/lift baseline.

Recorded replays are development diagnostics. Fresh inference stops when the
local geometric controller takes over. The controller uses simulator object
poses/contact identity; policy inputs remain only RGB and proprioception.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from diagnostic_environment import DiagnosticEnvironment
from grasp_depth_probe import APPROACHES, CASE, SeatingTrace, closeup, hand_geometry, read
from guarded_grasp import GuardConfig, GuardedTrace, measurements
from reactive_expert import apply_setup, reset_recording


def integration_state(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.m, spec))
    mujoco.mj_getState(env.m, env.d, state, spec)
    return state


def paired_observation(env, cache):
    """Common sensor sample only when the full integration state is identical."""
    key = hashlib.sha256(integration_state(env).tobytes()).hexdigest()
    state, rendered = env.observation()
    used_cached = key in cache
    if used_cached:
        old_state, views = cache[key]
        for name in state:
            np.testing.assert_array_equal(state[name], old_state[name])
    else:
        views = rendered
        cache[key] = ({k: np.asarray(v).copy() for k, v in state.items()},
            {k: np.asarray(v).copy() for k, v in views.items()})
    render_diff = {name: int(np.count_nonzero(np.asarray(rendered[name]) != views[name])) for name in views}
    return state, views, key, used_cached, render_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/guarded_grasp_v9"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--suite", choices=["normal", "external"], default="normal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--modes", nargs="+", choices=["baseline", "guarded"], default=["baseline", "guarded"])
    args = parser.parse_args()
    tag = "replay" if args.replay else args.suite
    root = args.output / tag
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "source"
    snapshots.mkdir()
    for name in ["run_guarded_grasp.py", "guarded_grasp.py", "grasp_depth_probe.py", "diagnostic_environment.py", "grasp_walloss.py",
                 "walloss_runtime.py", "runtime_compat.py", "test_guarded_grasp.py", "audit_policy_repeatability.py"]:
        shutil.copy2(Path(__file__).parent / name, snapshots / name)
    protocol = {"tag": tag, "guard_config": asdict(GuardConfig()), "modes": args.modes,
        "max_control_steps": 384, "control_period_s": .04, "simulation_budget_s": 15.36,
        "original_criterion": "anytime bilateral physical contact and bottom lift >=5cm for >=1s",
        "end_criterion": "at the end: continuous bilateral physical contact and bottom lift >=5cm for >=3s",
        "uses_simulator_object_pose_in_guard": True, "uses_simulator_contact_identity_in_both": True,
        "object_pose_given_to_model": False, "physics_paused_during_inference": True,
        "program_lift_identical_to_v8": True, "frozen_model_no_training": True,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "tf32_disabled": True, "python_numpy_torch_seed_reset_per_trial": True,
        "inference_seed_schedule": "trial_seed * 10000 + control_step // 8",
        "paired_common_observation_and_prediction": True,
        "shared_prefix_not_independent_inferences": True,
        "shared_observation_requires_identical_full_mjSTATE_INTEGRATION": True,
        "shared_model_sample_requires_same_physics_state_and_inference_seed": True,
        "external_cases_previously_evaluated_in_earlier_rounds": True,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots.iterdir()}}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    policy = None
    if not args.replay:
        assert args.source and args.checkpoint
        protocol["checkpoint_sha256"] = {name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
            for name in ["action_expert.safetensors", "normalizer_action.pth", "normalizer_propri.pth"]}
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
        sys.path.insert(0, str(args.source.resolve()))
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        from walloss_runtime import build_policy
        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        config = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
            norm_key="local/pigment_grasp_reactive_v4", cam_names=["face_view", "right_wrist_view"],
            model_device="cuda:0", robot_action_end_ratio=1., robot_action_interpolate_multiplier=1, smooth_action=False)
        policy = build_policy(config)
    report = {"completed": False, "protocol": protocol, "cases": [],
        "load_audit": policy.model_wrapper.load_audit if policy else None}
    if args.replay:
        cases = [(name, args.runs / "reactive_train_v4/episode_0000", 0,
            np.asarray(read(args.runs / path / CASE / "actions.json"))[:, :8].reshape(-1, 7))
            for name, path in APPROACHES.items()]
    elif args.suite == "normal":
        cases = [(f"seed_{seed}", args.runs / "reactive_train_v4/episode_0000", seed, None) for seed in args.seeds]
    else:
        cases = [(f"episode_{index:04d}_seed_{seed}", args.runs / f"reactive_validation_v4/episode_{index:04d}", seed, None)
                 for index in args.episodes for seed in args.seeds]
    for index, (name, source, seed, recorded) in enumerate(cases):
        modes = args.modes if index % 2 == 0 else list(reversed(args.modes))
        observation_cache, prediction_cache = {}, {}
        initial_physics, initial_control = None, None
        for mode in modes:
            folder = root / f"{name}_{mode}"
            env = DiagnosticEnvironment(source / "input_scene.xml", args.config, folder, "blue", tcp="grasp",
                controller_label=f"{'RECORDED APPROACH' if args.replay else 'MODEL APPROACH'} + {mode.upper()} PRIVILEGED CONTROL")
            env.step_time = .04
            episode = read(source / "episode.json")
            apply_setup(env, episode["setup"])
            if policy:
                if initial_physics is None:
                    initial_physics, initial_control = integration_state(env), env.control.copy()
                mujoco.mj_setState(env.m, env.d, initial_physics.copy(), mujoco.mjtState.mjSTATE_INTEGRATION)
                env.control[:] = initial_control
                mujoco.mj_forward(env.m, env.d)
                reset_recording(env)
            np.savez_compressed(folder / "initial_physics.npz", integration_state=integration_state(env), control=env.control)
            geometry, bodies = hand_geometry(env)
            lift_axis = env.d.xmat[env.arm_base].reshape(3, 3).T @ np.array([0., 0., 1.])
            trace = GuardedTrace(geometry, bodies, lift_axis) if mode == "guarded" else SeatingTrace(0., geometry, bodies, lift_axis)
            state, views, _, _, _ = paired_observation(env, observation_cache)
            initial = {"state": {k: np.asarray(v).tolist() for k, v in state.items()},
                "camera_sha256": {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in views.items()}}
            if policy:
                policy.reset(); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                with torch.inference_mode():
                    policy.infer({"state": state, "views": views,
                        "instruction": "Use your right hand to pick up the blue bottle from the tray and lift it."})
                torch.cuda.synchronize()
                policy.reset(); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            env.physics_trace = trace
            actions, inferences, frames, geometry_log = [], [], [], []
            current, chunk_start, program_started = None, 0, None
            case = {"name": name, "seed": seed, "mode": mode, "directory": folder.name,
                "source_case": str(source.relative_to(args.runs)), "kind": episode["kind"],
                "offset_xy_m": episode["offset_xy_m"], "trial_valid": False, "initial_observation": initial}
            try:
                for step in range(384):
                    active = trace.phase == "policy" if mode == "guarded" else trace.handoff is None
                    if recorded is not None:
                        raw = recorded[step]
                    else:
                        if active and step % 8 == 0:
                            state, views, physics_key, shared_observation, render_diff = paired_observation(env, observation_cache)
                            inference_seed = seed * 10000 + step // 8
                            torch.manual_seed(inference_seed); np.random.seed(inference_seed); random.seed(inference_seed)
                            input_digest = hashlib.sha256()
                            for key, data in sorted({**state, **views}.items()):
                                input_digest.update(key.encode()); input_digest.update(np.asarray(data).tobytes())
                            sample_key = (physics_key, inference_seed)
                            shared_prediction = sample_key in prediction_cache
                            torch.cuda.synchronize(); started = time.monotonic()
                            if shared_prediction:
                                current = prediction_cache[sample_key].copy()
                            else:
                                with torch.inference_mode():
                                    output = policy.infer({"state": state, "views": views,
                                        "instruction": "Use your right hand to pick up the blue bottle from the tray and lift it."})
                                torch.cuda.synchronize()
                                current = np.asarray(output["follow2_pos"], dtype=float)
                                prediction_cache[sample_key] = current.copy()
                            assert current.shape == (32, 7) and np.isfinite(current).all()
                            actions.append(current.tolist()); chunk_start = step
                            inferences.append({"control_step": step, "wall_seconds": time.monotonic() - started,
                                "rng_seed": inference_seed, "input_sha256": input_digest.hexdigest(),
                                "physics_state_sha256": physics_key, "shared_observation": shared_observation,
                                "shared_prediction": shared_prediction, "rendered_pixel_differences": render_diff})
                        raw = current[min(step - chunk_start, 31)]
                    before = measurements(env)
                    command = trace.guarded_command(env, raw, step, max(0, len(actions) - 1)) if mode == "guarded" else trace.command(raw, step, max(0, len(actions) - 1))
                    local_active = trace.phase != "policy" if mode == "guarded" else trace.handoff is not None
                    if local_active and program_started is None:
                        program_started = step
                    row = env.execute(command)
                    phase = trace.phase if mode == "guarded" else "lift_hold" if trace.handoff else "policy"
                    row.update(controller_phase=phase, raw_policy_reference=raw.tolist(),
                        raw_reference_is_new_inference=active and step % 8 == 0 and recorded is None)
                    geometry_log.append({"control_step": step, "before": before, "after": measurements(env), "phase": phase})
                    env.capture()
                    image = Image.fromarray(closeup(env, 0))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 28), fill=(20, 28, 26))
                    draw.text((6, 8), f"{mode.upper()} | PRIVILEGED {phase.upper()} | {name}", fill="white")
                    frames.append(np.asarray(image))
                diagnostic = trace.finish(folder, float(env.m.opt.timestep))
                final = trace.rows[-1]
                case.update(trial_valid=True, original_criterion_reached=bool(env.success),
                    max_bilateral_lifted_s=diagnostic["max_bilateral_lifted_s"],
                    end_bilateral_lifted_s=final["held_s"], end_bilateral_at_least_3s=final["held_s"] >= 3.,
                    final_lift_m=final["lift_m"], final_finger_contacts=final["contacts"],
                    handoff=trace.handoff, inference_queries=len(inferences),
                    inference_calls=sum(not v["shared_prediction"] for v in inferences), program_started_step=program_started,
                    guard=trace.protocol() if mode == "guarded" else None, diagnostics=diagnostic)
            finally:
                env.capture("final"); env.close()
                # All videos in this round use the actual 25Hz control rate.
                imageio.mimsave(folder / "rollout.mp4", env.frames, fps=25, codec="libx264", macro_block_size=2)
                if frames:
                    imageio.mimsave(folder / "closeup.mp4", frames, fps=25, codec="libx264", macro_block_size=2)
            for filename, data in [("actions.json", recorded.tolist() if recorded is not None else actions),
                ("inferences.json", inferences), ("geometry.json", geometry_log), ("result.json", case)]:
                (folder / filename).write_text(json.dumps(data, indent=2))
            if mode == "guarded":
                (folder / "guard_commands.json").write_text(json.dumps(trace.guard_log))
            report["cases"].append(case)
            (root / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in case.items() if k not in ["initial_observation", "diagnostics", "guard"]}), flush=True)
    report["completed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
