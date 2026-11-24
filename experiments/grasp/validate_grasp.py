"""Privileged scripted grasp used only to validate physical contact behavior."""
from pathlib import Path
import argparse
import json
import numpy as np

from grasp_walloss import GraspEnvironment


def main():
    root=Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=root/"generated/scene.xml")
    p.add_argument("--config", type=Path, default=root.parents[1]/"simulation/scene_config.json")
    p.add_argument("--output", type=Path, default=root/"runs/scripted_physics_check")
    args = p.parse_args()
    env = GraspEnvironment(args.scene, args.config, args.output, "blue", controller_label="SCRIPTED PHYSICS CHECK - NOT MODEL")
    report = {"controller": "scripted privileged-pose validation", "model_executed": False, "grasp_weld": False, "success": False}
    def move(tcp_position, grip, n=35):
        br = env.d.xmat[env.arm_base].reshape(3, 3)
        flange = np.asarray(tcp_position)-np.array([.132, 0., 0.])
        local = br.T@(flange-env.d.xpos[env.arm_base])
        command = np.r_[local, np.zeros(3), grip]
        for step in range(n):
            env.execute(command)
            if step % 4 == 0:
                env.capture()
    try:
        bottle = env.d.xpos[env.bottle].copy()
        move(bottle+[-.10, 0., .025], 4.5)
        move(bottle+[0., 0., .005], 4.5, 25)
        move(bottle+[0., 0., .005], 2.2, 15)
        env.capture("closed")
        move(bottle+[0., 0., .12], 2.2, 50)
        report.update(success=bool(env.success), max_bottle_bottom_lift_m=env.max_lift, bilateral_contact_steps=env.contact_frames, final_bottle_xyz=env.d.xpos[env.bottle].tolist())
    finally:
        env.capture("final")
        env.close()
        (args.output/"validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
