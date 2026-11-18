"""Shared robot pose for scene export and scenario initialization."""

import mujoco


def set_display_pose(m, d, cfg):
    def setq(name,value):d.qpos[m.joint(name).qposadr[0]]=value
    setq('base_x',cfg['robot']['base_x']);setq('base_y',cfg['robot']['source_base_y'])
    setq('lift_joint',.60);setq('head_pitch_joint',.15)
    for side in ['left','right']:
        for i,value in enumerate([0.,3.0,-2.6,-.4,0.,0.],1):setq(f'{side}_arm_joint{i}',value)
        setq(f'{side}_arm_gripper_left_joint',.035);setq(f'{side}_arm_gripper_right_joint',-.035)
    mujoco.mj_forward(m,d)
