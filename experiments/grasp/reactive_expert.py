"""Privileged reactive teacher and reproducible, unrecorded episode setup.

Object pose/contact identity are teacher-only signals. Neither enters the
20-dimensional policy observation. Setup motions are never action labels.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def local_command(env, world, grip, rotation=None):
    br=env.d.xmat[env.arm_base].reshape(3,3)
    position=br.T@(np.asarray(world)-env.d.xpos[env.arm_base])
    return np.r_[position,np.zeros(3) if rotation is None else rotation,grip]


def tool_world(env):
    pose=env.arm_pose("right")
    br=env.d.xmat[env.arm_base].reshape(3,3)
    return env.d.xpos[env.arm_base]+br@pose[:3]


def alignment_target(bottle,tool,opening,angle):
    """Continuous pregrasp-to-grasp field; closed/misaligned jaws retreat first."""
    grasp=np.asarray(bottle)+[0.,0.,.005]
    yz_error=np.linalg.norm(np.asarray(tool)[1:]-grasp[1:])
    aligned=np.clip(1-max(yz_error-.006,0.)/.03,0.,1.)
    opened=np.clip((opening-3.6)/.5,0.,1.)
    oriented=np.clip(1-max(angle-.04,0.)/.14,0.,1.)
    fraction=min(float(aligned),float(opened),float(oriented))
    return grasp+[-.10*(1-fraction),0.,.02*(1-fraction)]


def reset_recording(env):
    env.telemetry=[]
    env.frames=[]
    env.start_time=float(env.d.time)
    env.held_seconds=0.
    env.success=False
    env.max_lift=0.
    env.min_grasp_distance=float("inf")
    env.contact_frames=0
    env.capture("initial")


def apply_setup(env, spec):
    """Create a start state through physical commands, without teleport/weld."""
    kind=spec.get("kind","clean")
    if kind=="hand_offset":
        command=env.arm_pose("right").copy()
        command[:3]+=np.asarray(spec["position_offset_m"])
        command[3:6]=np.asarray(spec["rotation_rpy_rad"])
        command[6]=spec["gripper"]
        for _ in range(25):
            env.execute(command)
    elif kind=="grasp_miss":
        bottle=env.d.xpos[env.bottle].copy()
        for offset,grip,count in [([-.08,0.,.025],4.5,30),
                                  (spec["grasp_offset_m"],4.5,25),
                                  (spec["grasp_offset_m"],2.2,6)]:
            command=local_command(env,bottle+np.asarray(offset),grip)
            for _ in range(count):
                env.execute(command)
    elif kind=="slip":
        bottle=env.d.xpos[env.bottle].copy()
        for offset,grip,count in [([-.10,0.,.025],4.5,42),
                                  ([0.,0.,.005],4.5,30),
                                  ([0.,0.,.005],2.2,18),
                                  ([0.,0.,.065],2.2,20)]:
            command=local_command(env,bottle+np.asarray(offset),grip)
            for _ in range(count):
                env.execute(command)
        if env.finger_contacts()!=2:
            raise RuntimeError("Slip setup did not establish a physical bilateral grasp")
        before_release_z=float(env.d.xpos[env.bottle,2])
        command=env.arm_pose("right").copy()
        command[:2]+=np.asarray(spec.get("lateral_offset_m",[.005,0.]))
        command[6]=spec.get("release_gripper",3.5)
        for _ in range(spec.get("release_steps",2)):
            env.execute(command)
        for _ in range(3):
            if env.finger_contacts()<2 and env.d.xpos[env.bottle,2]<before_release_z-.003:
                break
            env.execute(command)
        if env.finger_contacts()==2 or env.d.xpos[env.bottle,2]>=before_release_z-.003:
            raise RuntimeError("Setup did not create an actual loss of grasp and downward slip")
    elif kind!="clean":
        raise ValueError(f"Unknown episode setup: {kind}")
    rz=abs(float(env.d.xmat[env.bottle].reshape(3,3)[2,2]))
    if rz<np.cos(np.radians(35)):
        raise RuntimeError("Setup tipped the bottle beyond the upright-recovery scope")
    reset_recording(env)
    return {"kind":kind,"setup_unrecorded":True,
            "initial_tool_pose":env.arm_pose("right").tolist(),
            "initial_bottle_xyz":env.d.xpos[env.bottle].tolist(),
            "initial_finger_contacts":env.finger_contacts()}


class ReactiveExpert:
    def __init__(self, env):
        self.phase="align"
        self.contact_seconds=0.
        self.recoveries=0
        self.lift_goal=None
        self.env=env

    def command(self):
        env=self.env
        bottle=env.d.xpos[env.bottle].copy()
        tool=tool_world(env)
        distance=float(np.linalg.norm(tool-(bottle+[0.,0.,.005])))
        pose=env.arm_pose("right")
        angle=float(Rotation.from_euler("xyz",pose[3:6]).magnitude())
        contacts=env.finger_contacts()
        if self.phase=="lift_hold" and np.linalg.norm(tool-bottle)>.045:
            self.phase="recover_align"
            self.recoveries+=1
            self.contact_seconds=0.
        if self.phase in ("align","recover_align"):
            self.contact_seconds=0.
            # Alignment must be observable in tool position and orientation.
            if distance<.008 and angle<.04 and pose[6]>=4.1:
                self.phase="close"
            else:
                return local_command(env,alignment_target(bottle,tool,pose[6],angle),4.5),self.phase
        if self.phase=="close":
            # If closing displaces the bottle, reopen rather than chase closed.
            if distance>.022 or angle>.10 or (pose[6]<2.25 and contacts<2):
                self.phase="recover_align"
                self.recoveries+=1
                self.contact_seconds=0.
                return local_command(env,alignment_target(bottle,tool,pose[6],angle),4.5),self.phase
            self.contact_seconds=self.contact_seconds+env.step_time if contacts==2 else 0.
            if self.contact_seconds>=.08:
                self.phase="lift_hold"
                self.lift_goal=bottle+[0.,0.,.115]
            else:
                return local_command(env,bottle+[0.,0.,.005],2.2),self.phase
        return local_command(env,self.lift_goal,2.2),"lift_hold"


def make_setup(kind,rng):
    if kind=="hand_offset":
        return {"kind":kind,"position_offset_m":rng.uniform([-.035,-.035,-.035],[.025,.035,.025]).tolist(),
                "rotation_rpy_rad":rng.uniform(-.18,.18,3).tolist(),
                "gripper":float(rng.uniform(2.1,3.5))}
    if kind=="grasp_miss":
        side=float(rng.choice([-1.,1.]))
        return {"kind":kind,"grasp_offset_m":[float(rng.uniform(-.018,.018)),side*float(rng.uniform(.012,.022)),.008]}
    if kind=="slip":
        return {"kind":kind,"lateral_offset_m":rng.uniform(-.006,.006,2).tolist(),
                "release_gripper":3.5,"release_steps":2}
    return {"kind":"clean"}
