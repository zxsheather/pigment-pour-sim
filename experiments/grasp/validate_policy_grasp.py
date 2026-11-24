"""Diagnostic: replay model grasp, then lift relative to measured arm pose.

This is NOT a fresh model rollout or model-only success. The approach comes
from recorded model commands; a proprioceptive handoff triggers a scripted
vertical lift. No object coordinates are used for selecting either motion.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from grasp_stages import GraspStages
from grasp_walloss import GraspEnvironment


ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=ROOT/"generated/blue_only/scene.xml")
    p.add_argument("--config", type=Path, default=ROOT/"generated/blue_only/scene_config.json")
    p.add_argument("--trace", type=Path, default=ROOT/"runs/blue_only_lower_ready_seed0/actions.json")
    p.add_argument("--output", type=Path, default=ROOT/"runs/policy_grasp_program_lift_check")
    args = p.parse_args()
    recorded = json.loads(args.trace.read_text())
    env = GraspEnvironment(args.scene, args.config, args.output, "blue", tcp="grasp", ready_pose=(.10, 0., .25), controller_label="REPLAYED MODEL GRASP + PROGRAM LIFT CHECK")
    stage = GraspStages()
    report = {"model_executed": False, "model_action_replay": True, "program_arm_lift": True, "uses_object_pose_for_control": False, "grasp_weld": False, "success": False, "handoff": None}
    try:
        for chunk in recorded:
            for raw in chunk["follow2_pos"][:8]:
                row = env.execute(np.asarray(raw))
                feedback = row["gripper_feedback"]
                dt = round(env.step_time/env.m.opt.timestep)*env.m.opt.timestep
                changed = stage.observe(raw[6], feedback["opening_motor_units"], feedback["effort_n"], dt)
                env.capture()
                if changed:
                    report["handoff"] = {"time_s": row["time_s"], "held_gripper": stage.held_gripper, "feedback": feedback}
                    break
            if stage.phase == "lift":
                break
        if stage.phase == "lift":
            env.capture("pinch")
            command = env.arm_pose("right").copy()
            command[2] += .12
            command[6] = stage.held_gripper
            for _ in range(60):
                env.execute(command)
                env.capture()
            report["program_lift_offset_in_arm_base_m"] = [0., 0., .12]
        report.update(success=bool(env.success), max_bottle_bottom_lift_m=env.max_lift, bilateral_contact_steps=env.contact_frames, simulated_seconds=float(env.d.time-env.start_time))
    finally:
        env.capture("final")
        env.close()
        (args.output/"validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
