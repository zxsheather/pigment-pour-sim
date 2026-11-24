"""Counterfactual replays from one recorded model grasp, with isolated changes.

The recorded model does not adapt to changed states. All interventions use
privileged contact detection and are diagnostic replays, not model successes.
The 1.7 motor-unit close target is a predeclared tighter target than the
teacher's 2.2. Physics, friction, actuator limits and attachments are unchanged.
"""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.spatial.transform import Rotation

from diagnostic_environment import DiagnosticEnvironment, PhysicsTrace
from reactive_expert import apply_setup


def read(path):
    return json.loads(path.read_text())


class CounterfactualTrace(PhysicsTrace):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.intervention = None

    def command(self, raw, control_step, chunk):
        command = super().command(raw, control_step, chunk)
        if self.intervention is not None:
            if self.mode in ["pose_hold", "pose_and_close"]:
                command[:6] = self.intervention["actual_tool_pose"][:6]
            if self.mode == "grip_latch":
                command[6] = self.intervention["issued_gripper"]
            elif self.mode in ["close_1p7", "pose_and_close"]:
                command[6] = 1.7
        self.applied = command.copy()
        self.requested_rotation = Rotation.from_euler("xyz", command[3:6])
        return command

    def observe(self, env):
        super().observe(env)
        if (self.intervention is None and self.contact_seconds >= .08 - 1e-9
                and self.applied[6] < min(env.gripper_feedback()["opening_motor_units"]) - .1):
            self.intervention = {"time_s": float(env.d.time - env.start_time),
                "actual_tool_pose": env.arm_pose("right").tolist(),
                "issued_gripper": float(self.applied[6]), "applied_from_next_control_step": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/grasp/runs/feedback_diagnostic_v7"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    args = parser.parse_args()
    output = args.root / "replay_probes"
    output.mkdir(parents=True, exist_ok=False)
    probe = read(args.root / "cases.json")[0]
    source = args.runs / probe["batch"] / f"episode_{probe['episode']:04d}"
    name = f"{probe['split']}_{probe['batch']}_e{probe['episode']:04d}"
    recorded = args.runs / "takeover_training_eval_v6/augmented_closedloop_internal" / name
    recorded_chunks = read(recorded / "actions.json")
    commands = np.asarray(recorded_chunks)[:, :8].reshape(-1, 7)
    assert len(commands) == 384
    report = {"completed": False, "case": probe, "model_inference_executed": False,
        "model_adapts_to_intervention": False, "uses_privileged_contact_detection": True,
        "same_recorded_commands_outside_modified_channels": True,
        "probe_close_target": 1.7, "reference_teacher_close_target": 2.2,
        "not_model_successes": True, "cases": []}
    for mode in ["grip_latch", "close_1p7", "pose_hold", "pose_and_close"]:
        folder = output / mode
        env = DiagnosticEnvironment(source / "input_scene.xml", args.config, folder, "blue", tcp="grasp",
            controller_label="RECORDED POLICY + " + mode.upper() + " | PRIVILEGED DIAGNOSTIC")
        env.step_time = .04
        apply_setup(env, read(source / "episode.json")["setup"])
        trace = CounterfactualTrace(mode)
        env.physics_trace = trace
        try:
            for step, raw in enumerate(commands):
                applied = trace.command(raw, step, step // 8)
                if mode in ["grip_latch", "close_1p7"]:
                    assert np.array_equal(applied[:6], raw[:6])
                elif mode == "pose_hold":
                    assert applied[6] == raw[6]
                env.execute(applied)
                env.capture()
            result = {"mode": mode, "trial_valid": True, "success_criterion_reached": bool(env.success),
                "intervention": trace.intervention, "simulated_seconds": float(env.d.time - env.start_time),
                "diagnostics": trace.finish(folder, float(env.m.opt.timestep))}
            assert trace.intervention is not None
        finally:
            env.capture("final")
            env.close()
        (folder / "result.json").write_text(json.dumps(result, indent=2))
        report["cases"].append(result)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}), flush=True)
    report["completed"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
