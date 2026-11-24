"""Separate closure strength from numerical slow slip on one recorded grasp.

No model inference or training. All Cartesian commands are the saved applied
commands. Interventions begin only after the original grasp handoff. The
solver variant is a simulator diagnostic, not a hardware improvement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np

from diagnostic_environment import DiagnosticEnvironment, PhysicsTrace
from reactive_expert import apply_setup


def read(path):
    return json.loads(path.read_text())


class SlipTrace(PhysicsTrace):
    def observe(self, env):
        super().observe(env)
        hand_rotation = env.d.xmat[env.ee].reshape(3, 3)
        local = hand_rotation.T @ (env.d.xpos[env.bottle] - env.d.xpos[env.ee])
        tool = env.d.xpos[env.ee] + hand_rotation @ env.tcp_offset
        normal = {body: 0. for body in sorted(env.fingers)}
        for index, contact in enumerate(env.d.contact):
            a, b = [int(env.m.geom_bodyid[g]) for g in [contact.geom1, contact.geom2]]
            if env.bottle not in [a, b]:
                continue
            finger = b if a == env.bottle else a
            if finger in normal:
                force = np.zeros(6)
                mujoco.mj_contactForce(env.m, env.d, index, force)
                normal[finger] += max(0., float(force[0]))
        self.rows[-1].update(bottle_local_xyz_m=local.tolist(),
            relative_world_z_m=float(env.d.xpos[env.bottle, 2] - tool[2]),
            finger_normal_sums_n=list(normal.values()))


def summarize(trace):
    start = min(trace.rows, key=lambda r: abs(r["time_s"] - 7.))
    final = trace.rows[-1]
    late = min(trace.rows, key=lambda r: abs(r["time_s"] - 14.))
    rows = [r for r in trace.rows if r["time_s"] >= start["time_s"]]
    return {"relative_drop_7s_to_end_mm": (start["relative_world_z_m"] - final["relative_world_z_m"]) * 1000,
        "relative_drop_14s_to_end_mm": (late["relative_world_z_m"] - final["relative_world_z_m"]) * 1000,
        "average_relative_down_mm_s": (start["relative_world_z_m"] - final["relative_world_z_m"]) * 1000 / (final["time_s"] - start["time_s"]),
        "final_lift_m": final["lift_m"], "final_tilt_deg": final["bottle_tilt_deg"],
        "end_bilateral_lifted_s": final["held_s"],
        "finger_actuator_abs_median_n": [statistics.median(abs(r[k]) for r in rows) for k in ["effort_left_n", "effort_right_n"]],
        "finger_normal_sum_median_n": [statistics.median(r["finger_normal_sums_n"][i] for r in rows) for i in [0, 1]],
        "normal_sums_are_contact_magnitudes_not_net_vertical_support": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("experiments/grasp/runs/guarded_grasp_v9/paired/normal/seed_1_guarded"))
    parser.add_argument("--source", type=Path, default=Path("experiments/grasp/runs/reactive_train_v4/episode_0000"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/slip_diagnostic_v10"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = read(args.reference / "telemetry.json")
    assert len(original) == 384
    intervention_step = next(i for i, row in enumerate(original) if row["controller_phase"] == "lift_hold")
    report = {"completed": False, "reference": str(args.reference), "intervention_step": intervention_step,
        "intervention_time_s": intervention_step * .04, "recorded_applied_actions_only": True,
        "model_inference": False, "training": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": []}
    baseline = None
    for name, grip, noslip in [("baseline", None, None), ("tighter_1p7", 1.7, None), ("noslip3", None, 3)]:
        folder = args.output / name
        env = DiagnosticEnvironment(args.source / "input_scene.xml", args.config, folder, "blue", tcp="grasp",
            controller_label="RECORDED ACTION SLIP DIAGNOSTIC")
        env.step_time = .04
        apply_setup(env, read(args.source / "episode.json")["setup"])
        with np.load(args.reference / "initial_physics.npz") as state:
            mujoco.mj_setState(env.m, env.d, state["integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
            env.control[:] = state["control"]
        mujoco.mj_forward(env.m, env.d)
        env.start_time = float(env.d.time)
        env.telemetry, env.frames = [], []
        env.capture("initial")
        trace = SlipTrace()
        env.physics_trace = trace
        reference_error, prefix_error = 0., 0.
        baseline_solver = {key: getattr(env.m.opt, key) for key in ["noslip_iterations", "impratio", "timestep"]}
        try:
            for step, recorded in enumerate(original):
                command = np.asarray(recorded["requested_pose"], dtype=float)
                if step >= intervention_step:
                    if grip is not None:
                        command[6] = grip
                    if noslip is not None:
                        env.m.opt.noslip_iterations = noslip
                applied = trace.command(command, step, step // 8)
                row = env.execute(applied)
                for key in ["right_ee", "bottle_xyz", "actual_tool_xyz", "held_seconds"]:
                    reference_error = max(reference_error, float(np.max(np.abs(np.asarray(row[key]) - recorded[key]))))
                    if baseline is not None and step < intervention_step:
                        prefix_error = max(prefix_error, float(np.max(np.abs(np.asarray(row[key]) - baseline[step][key]))))
            if name == "baseline":
                assert reference_error < 1e-7, f"Replay differs from original: {reference_error}"
                baseline = env.telemetry.copy()
            assert prefix_error == 0.
            diagnostic = trace.finish(folder, float(env.m.opt.timestep))
            case = {"name": name, "trial_valid": True, "hold_gripper": grip if grip is not None else 2.2,
                "baseline_solver": baseline_solver, "noslip_iterations_after_handoff": int(env.m.opt.noslip_iterations),
                "baseline_replay_max_abs_error": reference_error if name == "baseline" else None,
                "preintervention_max_abs_error": prefix_error, "diagnostic": diagnostic, **summarize(trace)}
            (folder / "result.json").write_text(json.dumps(case, indent=2))
            report["cases"].append(case)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in case.items() if k != "diagnostic"}), flush=True)
        finally:
            env.capture("final")
            env.frames.clear()  # This numerical diagnostic produces stills, not a one-frame rollout video.
            env.close()
    report["completed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "executed_source.py").write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
