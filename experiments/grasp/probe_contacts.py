"""Diagnostic FK placement: measure jaw contacts at an aligned bottle pose."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco

from grasp_walloss import GraspEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    env = GraspEnvironment(args.scene, args.config, args.output, "blue", controller_label="COLLISION DIAGNOSTIC - NOT POLICY")
    try:
        bottle = env.d.xpos[env.bottle].copy()
        br = env.d.xmat[env.arm_base].reshape(3, 3)
        local = br.T@(bottle-np.array([.132, 0., 0.])-env.d.xpos[env.arm_base])
        command = np.r_[local, np.zeros(3), 4.5]
        for _ in range(50):
            q, _, _ = env.solve_ik(command)
            env.d.qpos[env.arm_q] = q
            mujoco.mj_forward(env.m, env.d)
        for grip in [4.5, 4., 3.5, 3., 2.5, 2., 1.]:
            env.set_gripper("right", grip)
            for name in ["right_arm_gripper", "right_arm_gripper_left_joint", "right_arm_gripper_right_joint"]:
                env.d.qpos[env.m.joint(name).qposadr[0]] = env.control[env.acts[name]]
            mujoco.mj_forward(env.m, env.d)
            rows=[]
            for con in env.d.contact:
                a,b=(int(env.m.geom_bodyid[g]) for g in [con.geom1,con.geom2])
                if env.bottle in [a,b]:
                    other=b if a==env.bottle else a
                    rows.append({"body": env.m.body(other).name,"distance_m":float(con.dist)})
            print(json.dumps({"grip":grip,"tcp_error_m":float(np.linalg.norm(env.d.site_xpos[env.m.site('right_tool').id]-bottle)),"contacts":rows}),flush=True)
        env.capture("aligned")
    finally:
        env.renderer.close()


if __name__ == "__main__":
    main()
