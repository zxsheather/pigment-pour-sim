"""Replay recorded policy actions to check simulator stability without a model."""
from pathlib import Path
import argparse
import json

import mujoco
import numpy as np

from grasp_walloss import GraspEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--motor-armature", type=float, default=1e-4)
    args = p.parse_args()
    actions = np.asarray(json.loads(args.trace.read_text())["follow2_pos"])
    env = GraspEnvironment(args.scene, args.config, args.output, "blue", motor_armature=args.motor_armature)
    result = {"stable": False, "steps": 0}
    try:
        for pose in actions[:args.steps]:
            q, _, _ = env.solve_ik(pose)
            env.control[env.arm_acts] = q
            env.set_gripper("right", pose[6])
            env.d.ctrl[:] = env.control
            mujoco.mj_forward(env.m, env.d)
            worst = int(np.argmax(abs(env.d.qacc)))
            joint = int(env.m.dof_jntid[worst])
            if result["steps"] == 0:
                print(json.dumps({"stage": "before_first_step", "joint": env.m.joint(joint).name, "qacc": float(env.d.qacc[worst]), "actuator_force": float(env.d.qfrc_actuator[worst]), "armature": float(env.m.dof_armature[worst]), "gripper_command": float(pose[6])}), flush=True)
            before = float(env.d.time)
            env.execute(pose)
            assert env.d.time > before
            result["steps"] += 1
        result["stable"] = True
        result["simulated_seconds"] = float(env.d.time-env.start_time)
    except (RuntimeError, AssertionError) as e:
        result["error"] = str(e)
    finally:
        env.renderer.close()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"replay_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["stable"] else 1)


if __name__ == "__main__":
    main()
