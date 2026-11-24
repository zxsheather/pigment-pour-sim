"""Regression: a 58 mm bottle must fit into the fully opened jaws."""
from pathlib import Path
import argparse
import tempfile
import unittest
import numpy as np
import mujoco

from grasp_walloss import GraspEnvironment

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--scene",type=Path,default=ROOT/"generated/scene.xml")
p.add_argument("--config",type=Path,default=ROOT.parents[1]/"simulation/scene_config.json")
args,extra=p.parse_known_args()


class OpenJawGeometry(unittest.TestCase):
    def test_bottle_fits_without_finger_penetration(self):
        with tempfile.TemporaryDirectory() as folder:
            env=GraspEnvironment(args.scene,args.config,Path(folder),"blue")
            try:
                bottle=env.d.xpos[env.bottle].copy()
                br=env.d.xmat[env.arm_base].reshape(3,3)
                local=br.T@(bottle-np.array([.132,0,0])-env.d.xpos[env.arm_base])
                pose=np.r_[local,np.zeros(3),4.5]
                for _ in range(50):
                    q,_,_=env.solve_ik(pose)
                    env.d.qpos[env.arm_q]=q
                    mujoco.mj_forward(env.m,env.d)
                env.set_gripper("right",4.5)
                for name in ["right_arm_gripper","right_arm_gripper_left_joint","right_arm_gripper_right_joint"]:
                    env.d.qpos[env.m.joint(name).qposadr[0]]=env.control[env.acts[name]]
                mujoco.mj_forward(env.m,env.d)
                self.assertLess(np.linalg.norm(env.d.site_xpos[env.m.site("right_tool").id]-bottle),.001)
                penetrations=[]
                for con in env.d.contact:
                    a,b=(int(env.m.geom_bodyid[g]) for g in [con.geom1,con.geom2])
                    if (a==env.bottle and b in env.fingers) or (b==env.bottle and a in env.fingers):
                        penetrations.append(float(con.dist))
                self.assertTrue(all(d>=-.001 for d in penetrations),f"Open jaw penetrates bottle: {penetrations}")
            finally:
                env.renderer.close()


if __name__=="__main__":
    unittest.main(argv=[__file__,*extra])
