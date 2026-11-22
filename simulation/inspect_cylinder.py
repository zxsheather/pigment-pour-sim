"""Contact-grasp a cylinder from the tray and move it in front of the camera."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares

from carry_env import CarryEnv, CARRY_JOINTS, CONTROL_DT, PHYSICS_DT, GRASP_OFFSET, ROOT, CAMERAS, angle_difference
from build_scene import camera, element, fmt
from cylinder_reader import read_volume, annotate
from cylinder_actions import commanded_cartesian_action

SCALE_ZERO = 0.012
SCALE_SPAN = 0.220
CAPACITY_ML = 50.0
GRIP_HEIGHT = 0.225
READ_DISTANCE = 0.35
CYLINDER_INSTRUCTION = ("Pick up the measuring cylinder, hold it upright, bring it close to the head camera, "
                        "and align the liquid surface with the image center for reading. Keep holding it.")
INSPECTION_SOURCE_HASHES = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__), ROOT/"cylinder_actions.py", ROOT/"cylinder_reader.py",
                 ROOT/"carry_env.py", ROOT/"audit_gripper.py", ROOT/"scene_config.json")
}


def add_digit(parent, digit: str, x: float, z: float):
    segments = {"0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd",
                "4": "fgbc", "5": "afgcd", "6": "afgecd", "7": "abc",
                "8": "abcdefg", "9": "abfgcd"}[digit]
    points = {"a": (-.0012, .003, .0012, .003), "b": (.0012, .003, .0012, 0),
              "c": (.0012, 0, .0012, -.003), "d": (-.0012, -.003, .0012, -.003),
              "e": (-.0012, 0, -.0012, -.003), "f": (-.0012, .003, -.0012, 0),
              "g": (-.0012, 0, .0012, 0)}
    for segment in segments:
        x1, z1, x2, z2 = points[segment]
        element(parent, "geom", type="capsule", fromto=fmt([x+x1, -.013, z+z1, x+x2, -.013, z+z2]),
                size=".00022", rgba=".08 .10 .12 1", mass=0, contype=0, conaffinity=0)


class CylinderEnv(CarryEnv):
    def __init__(self, volume_ml=25.0, *, render=False, image_size=640, grip_profile="official",
                 friction_scale=1.0, mass_scale=1.0,grasp_height=GRIP_HEIGHT,grasp_x=GRASP_OFFSET[0],
                 solver_impratio=1.,physics_dt=PHYSICS_DT,grip_force_target=None):
        if not 0 < volume_ml <= CAPACITY_ML:
            raise ValueError("Volume must be within (0,50] mL")
        self.volume_ml = float(volume_ml)
        if grip_profile not in ("official","flat","v-pad"):raise ValueError("Unknown grip profile")
        self.grip_profile=grip_profile
        if not all(np.isfinite(v) and v>0 for v in (friction_scale,mass_scale)):
            raise ValueError("Friction and mass scales must be finite and positive")
        self.friction_scale=float(friction_scale)
        self.mass_scale=float(mass_scale)
        self.grasp_height=float(grasp_height)
        self.grasp_offset=np.array([grasp_x,0.,0.])
        self.solver_impratio=float(solver_impratio)
        self.grip_force_target=grip_force_target
        if grip_force_target is not None and not 0<grip_force_target<=20:raise ValueError("Invalid grip force target")
        if not np.isfinite(self.solver_impratio) or self.solver_impratio<=0:raise ValueError("Invalid impratio")
        if not 0<physics_dt<=CONTROL_DT:raise ValueError("Invalid physics timestep")
        self.finger_force_limits={finger:10.0 for finger in ("left","right")}
        if grip_profile=="official":
            spec=json.loads((ROOT.parent/"sources.json").read_text())["sdk_robot"]
            urdf=ET.parse(ROOT.parent/spec["destination"]/"urdf/quanta_x1/ex001_robot.urdf").getroot()
            self.finger_force_limits={finger:float(urdf.find(
                f"joint[@name='right_arm_gripper_{finger}_joint']/limit").get("effort"))
                for finger in ("left","right")}
        super().__init__(render=render, image_size=image_size, model_transform=self._configure_model,
                         finger_collisions="official" if grip_profile=="official" else "flat-proxy")
        self.model.opt.timestep=physics_dt
        self.cylinder_body = self.model.body("measuring_cylinder").id
        self.arm_addresses = np.array([self.model.joint(f"right_arm_joint{i}").qposadr[0]
                                       for i in range(1, 7)], dtype=int)
        self.arm_dofs = np.array([self.model.joint(f"right_arm_joint{i}").dofadr[0]
                                  for i in range(1, 7)], dtype=int)
        self.ik_data = mujoco.MjData(self.model)

    def _configure_model(self, scene, config):
        official=self.grip_profile=="official"
        scene.find("option").set("noslip_iterations","0" if official else "8")
        if official:scene.find("option").set("impratio",str(self.solver_impratio))
        body = scene.find(".//body[@name='measuring_cylinder']")
        for child in list(body):
            body.remove(child)
        element(body, "freejoint", name="cylinder_free")
        mass = (0.09 + self.volume_ml / 1000)*self.mass_scale
        friction=fmt(np.array([1.,.005,.0001] if official else [1.2,.01,.008])*self.friction_scale)
        element(body, "inertial", pos="0 0 .105", mass=mass,
                diaginertia=fmt([mass*.245**2/12, mass*.245**2/12, mass*.012**2/2]))
        element(body, "geom", name="cylinder_base", type="box", pos="0 0 .003", size=".043 .043 .003",
                rgba=".82 .88 .88 .8", mass=0, contype=2, conaffinity=3)
        inner = np.sqrt(CAPACITY_ML * 1e-6 / (np.pi * SCALE_SPAN))
        outer = config["cylinder"]["radius"]
        middle = (inner + outer) / 2
        for index in range(32):
            angle = 2 * np.pi * index / 32
            element(body, "geom", name=f"cylinder_wall_{index}", type="box",
                    pos=fmt([middle*np.cos(angle), middle*np.sin(angle), .1255]),
                    size=fmt([(outer-inner)/2, middle*np.pi/32, .1195]), euler=fmt([0, 0, angle]),
                    rgba=".80 .90 .93 .12", mass=0, contype=2, conaffinity=3,
                    friction=friction, condim=3 if official else 6, solref=".02 1" if official else ".006 1")
        height = self.volume_ml / CAPACITY_ML * SCALE_SPAN
        element(body, "geom", name="cylinder_liquid", type="cylinder",
                size=fmt([inner*.98, height/2]), pos=fmt([0,0,SCALE_ZERO+height/2]),
                rgba=".10 .32 .85 .92", mass=0, contype=0, conaffinity=0)
        # Visible ruler: conventional black ticks, with two calibrated colored
        # anchors for the first image-only reader (20 and 35 mL).
        for ml in range(51):
            z = SCALE_ZERO + ml / CAPACITY_ML * SCALE_SPAN
            major = ml % 5 == 0
            rgba = ".08 .10 .12 1"
            if ml == 20: rgba = ".9 .05 .8 1"
            if ml == 35: rgba = "1 .35 .03 1"
            element(body, "geom", name=f"cylinder_tick_{ml}", type="box",
                    pos=fmt([-.006, -.0126, z]), size=fmt([.006 if major else .003, .00018, .00035 if major else .00022]),
                    rgba=rgba, mass=0, contype=0, conaffinity=0)
            if ml % 10 == 0 or ml == 35:
                for digit_index, digit in enumerate(str(ml)):
                    add_digit(body, digit, .004+digit_index*.0034, z)
        for finger in ("left", "right"):
            actuator=scene.find(f".//actuator/position[@name='right_arm_gripper_{finger}_joint_servo']")
            limit=self.finger_force_limits[finger]
            actuator.set("forcerange",fmt([-limit,limit]))
            if official:
                # The URDF supplies geometry, but no contact material. Use
                # nominal MuJoCo contact defaults, explicitly uncalibrated.
                for geom in scene.findall(f".//body[@name='right_arm_gripper_{finger}_link']/geom"):
                    if geom.get("contype")!="0":
                        geom.set("friction",friction);geom.set("condim","3")
                        geom.set("solref",".02 1")
                continue
            offset=.018 if finger=="left" else -.018
            for level,height in enumerate((-.014,.014)):
                pad=scene.find(f".//geom[@name='right_{finger}_contact_pad_{level}']")
                pad.set("pos",fmt([.055,0,offset+height]))
                pad.set("size",".016 .005 .012")
                pad.set("condim","6");pad.set("friction",friction)
                if self.grip_profile=="v-pad":
                    parent=scene.find(f".//body[@name='right_arm_gripper_{finger}_link']")
                    parent.remove(pad)
                    side=1 if finger=="left" else -1
                    for face in (-1,1):
                        # Two angled faces per height make a V-groove around
                        # the cylinder instead of letting it roll on a flat pad.
                        element(parent,"geom",name=f"right_{finger}_vpad_{level}_{face}",type="box",
                                pos=fmt([.055+face*.009,-side*.003,offset+height]),
                                size=".008 .003 .010",euler=fmt([0,0,-side*face*np.pi/6]),
                                rgba=".16 .18 .20 1",mass=0,contype=1,conaffinity=2,
                                condim=6,friction=friction,solref=".006 1",solimp=".98 .999 .001")
        camera(scene.find("worldbody"), "cylinder_overview", [2.1,1.0,1.8], [.65,-.05,.9], 48)

    def reset_inspection(self, *, seed=0, jitter=0.0, yaw_jitter=0.0):
        if not np.isfinite(jitter) or jitter<0:raise ValueError("jitter must be finite and nonnegative")
        if not np.isfinite(yaw_jitter) or yaw_jitter<0:raise ValueError("yaw_jitter must be finite and nonnegative")
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        for name,value in (("base_x",.614),("base_y",-.38),("base_yaw",np.pi/2),
                           ("lift_joint",0.2),("head_pitch_joint",0.0)):
            self._set_joint(name,value)
        for side in ("left","right"):
            for index,value in enumerate(CARRY_JOINTS,1):self._set_joint(f"{side}_arm_joint{index}",value)
            self._set_joint(f"{side}_arm_gripper_left_joint",.034)
            self._set_joint(f"{side}_arm_gripper_right_joint",-.034)
        self.hold_targets={a:float(self.data.qpos[q]) for a,q in self.position_actuators.items()}
        qadr=self.model.joint("cylinder_free").qposadr[0]
        self.data.qpos[qadr:qadr+2]+=rng.uniform(-jitter,jitter,2)
        self.initial_yaw=float(rng.uniform(-yaw_jitter,yaw_jitter))
        mujoco.mju_axisAngle2Quat(self.data.qpos[qadr+3:qadr+7],np.array([0.,0.,1.]),self.initial_yaw)
        self.bottle_body=self.cylinder_body
        self.goal_pose=self.base_pose().copy()
        self.applied_action[:]=0
        self.grasp_reference=np.zeros(3)
        self.grasp_point_reference=None
        self.phase="approach";self.done=False
        for _ in range(150):self._apply_servos(0,0);mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.data.time=0
        self.grasp_reference=self._bottle_relative_position()
        return self.observe()

    def grip_point_relative(self):
        point=self.data.xpos[self.cylinder_body]+self.data.xmat[self.cylinder_body].reshape(3,3)@np.array([0,0,self.grasp_height])
        return self.data.xmat[self.gripper_body].reshape(3,3).T@(point-self.data.xpos[self.gripper_body])

    def metrics(self):
        values=super().metrics()
        loaded_fingers=set();finger_forces={body:0.0 for body in self.finger_bodies}
        for index,contact in enumerate(self.data.contact[:self.data.ncon]):
            a,b=self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            if self.cylinder_body not in (a,b):continue
            other=b if a==self.cylinder_body else a
            if other in self.finger_bodies:
                force=np.empty(6);mujoco.mj_contactForce(self.model,self.data,index,force)
                finger_forces[int(other)]+=max(0.0,float(force[0]))
                if force[0]>.05:loaded_fingers.add(int(other))
        values["grip_contacts"]=len(loaded_fingers)
        values["finger_normal_forces_n"]=list(finger_forces.values())
        if getattr(self,"grasp_point_reference",None) is not None:
            values["bottle_slip_m"]=float(np.linalg.norm(self.grip_point_relative()-self.grasp_point_reference))
        return values

    def ik_joint_bounds(self):
        ranges=np.array([self.model.joint(f"right_arm_joint{i}").range for i in range(1,7)])
        return ranges[:,0]+.005,ranges[:,1]-.005

    def inverse_kinematics(self, point: np.ndarray, rotation: np.ndarray, *, initial=None):
        d=self.ik_data
        d.qpos[:]=self.data.qpos
        def residual(q):
            d.qpos[self.arm_addresses]=q
            mujoco.mj_kinematics(self.model,d)
            current=d.xpos[self.gripper_body]+d.xmat[self.gripper_body].reshape(3,3)@self.grasp_offset
            delta_rotation=rotation@d.xmat[self.gripper_body].reshape(3,3).T
            delta_quat=np.empty(4);mujoco.mju_mat2Quat(delta_quat,delta_rotation.reshape(-1))
            orientation=np.empty(3);mujoco.mju_quat2Vel(orientation,delta_quat,1.0)
            return np.r_[point-current,.25*orientation]
        low,high=self.ik_joint_bounds()
        seeds=[initial if initial is not None else self.data.qpos[self.arm_addresses],
               [0,1.2,-.5,-.7,0,0],[-.8,.6,-1.4,.8,.8,0]]
        best=None
        for seed in seeds:
            result=least_squares(residual,np.clip(seed,low,high),bounds=(low,high),
                                 max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            if best is None or np.linalg.norm(result.fun)<np.linalg.norm(best.fun):best=result
            if np.linalg.norm(result.fun)<.0001:break
        return best.x.copy(),float(np.linalg.norm(best.fun))

    def step_joints(self, target, *, opening=.034, forward=0.0,yaw=0.0):
        for index,value in enumerate(target,1):
            actuator=self.model.actuator(f"right_arm_joint{index}_servo").id
            previous=self.hold_targets[actuator]
            self.hold_targets[actuator]=float(previous+np.clip(value-previous,-.65*CONTROL_DT,.65*CONTROL_DT))
        for finger,sign in (("left",1),("right",-1)):
            actuator=self.model.actuator(f"right_arm_gripper_{finger}_joint_servo").id
            previous=self.hold_targets[actuator]
            self.hold_targets[actuator]=float(previous+np.clip(sign*opening-previous,-.03*CONTROL_DT,.03*CONTROL_DT))
        self.applied_action[0]+=np.clip(forward-self.applied_action[0],-.3*CONTROL_DT,.3*CONTROL_DT)
        self.applied_action[1]+=np.clip(yaw-self.applied_action[1],-.3*CONTROL_DT,.3*CONTROL_DT)
        for _ in range(round(CONTROL_DT/self.model.opt.timestep)):
            if self.grip_force_target is not None and opening<=.015:
                # Force-limited closure using robot joint position/velocity,
                # not object pose or privileged contact-force feedback.
                for finger,direction in (("left",-1.),("right",1.)):
                    name=f"right_arm_gripper_{finger}_joint"
                    joint=self.model.joint(name);actuator=self.model.actuator(name+"_servo").id
                    position=float(self.data.qpos[joint.qposadr[0]])
                    velocity=float(self.data.qvel[joint.dofadr[0]])
                    speed_limit=.03 if abs(position)>.015 else .12
                    force=direction*min(self.grip_force_target,200*(speed_limit-direction*velocity))
                    limit=self.finger_force_limits[finger];force=float(np.clip(force,-limit,limit))
                    kp=self.model.actuator_gainprm[actuator,0]
                    kv=-self.model.actuator_biasprm[actuator,2]
                    command=position+(force+kv*velocity)/kp
                    self.hold_targets[actuator]=float(np.clip(command,*joint.range))
            self._apply_servos(*self.applied_action);mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        return self.observe(),self.metrics()


class InspectionFailure(RuntimeError):
    pass


class ReadingFailure(InspectionFailure):
    pass


def run_inspection(env: CylinderEnv, output: Path, *, record=False, seed=0, jitter=0.0,yaw_jitter=0.0,
                   grip_opening=.010,motion_scale=1.0,read_lateral=.14,read_distance=READ_DISTANCE,
                   grasp_roll=0.,upright_gain=0.,record_training=False,training_image_size=224,
                   reset_scene=True,initial_state_provenance=None,start_holding=False,skip_retreat=False,
                   held_initial_lift_m=0.,eye_alignment_target_px=2.,approach_motion_scale=None,
                   release_single_contact=False,release_retreat_m=0.,release_retreat_seconds=1.5,
                   release_retreat_along_contact_normal=False,
                   release_lift_m=0.,teacher_grasp_height_m=None,
                   stable_read_samples=3,approach_base_reference=None,align_from_clearance=False,
                   wrist_clearance_rear_distance_m=None,approach_heading_reference=None,
                   wrist_clearance_open_after_retreat=False,wrist_clearance_align_before_lift=False,
                   wrist_clearance_retreat_from_reset=False,read_from_current_view=False,
                   retry_reading_from_current_view=False,wrist_clearance_extra_lift_m=None,
                   align_near_grasp=False,hold_wrist_on_heading_return=False,
                   align_near_grasp_forward=False,
                   wrist_clearance_align_at_grasp_height=False,release_retreat_before_open=False,
                   align_heading_before_base_advance=False,align_near_grasp_at_current_depth=False,
                   lateral_alignment_min_depth_m=None,read_lateral_retry_offsets_m=(),near_grasp_open_ticks=12,
                   near_grasp_align_seconds=1.5,near_grasp_settle_ticks=10,
                   grasp_close_ticks=35,initial_lift_seconds=2.,held_presentation_waypoints=False):
    if not 0<=grip_opening<.02 or motion_scale<=0:raise ValueError("Invalid controller parameters")
    if type(grasp_close_ticks) is not int or grasp_close_ticks < 1:
        raise ValueError("Grasp closing must last a positive number of control ticks")
    if not np.isfinite(initial_lift_seconds) or initial_lift_seconds <= 0.:
        raise ValueError("Initial lift duration must be finite and positive")
    if approach_motion_scale is not None and (not np.isfinite(approach_motion_scale) or approach_motion_scale <= 0):
        raise ValueError("Approach motion scale must be finite and positive")
    if not np.isfinite(held_initial_lift_m) or not 0<=held_initial_lift_m<=.10:
        raise ValueError("Held recovery lift must be within [0,0.10] m")
    if held_initial_lift_m and (not start_holding or skip_retreat):
        raise ValueError("Initial lift requires a held transport recovery")
    if read_from_current_view and (reset_scene or not start_holding or not skip_retreat):
        raise ValueError("Reading from the current view requires a saved held reading state")
    if held_presentation_waypoints and (not start_holding or read_from_current_view):
        raise ValueError("Held presentation waypoints require a preserved grasp and a move to the camera")
    if release_single_contact and (reset_scene or start_holding):
        raise ValueError("Single-contact release requires a saved pre-grasp recovery state")
    if align_from_clearance and (reset_scene or start_holding):
        raise ValueError("Wrist clearance alignment requires a saved unheld approach")
    if align_near_grasp and (reset_scene or start_holding or align_from_clearance or release_single_contact):
        raise ValueError("Near grasp alignment requires a saved unheld approach with the fingers clear")
    if align_near_grasp_forward and not align_near_grasp:
        raise ValueError("Forward alignment requires near grasp alignment")
    if align_near_grasp_at_current_depth and (not align_near_grasp or align_near_grasp_forward):
        raise ValueError("Current-depth alignment requires its own near grasp mode")
    if not isinstance(near_grasp_open_ticks, (int, np.integer)) or near_grasp_open_ticks < 0:
        raise ValueError("Near grasp opening ticks must be a nonnegative integer")
    if not np.isfinite(near_grasp_align_seconds) or near_grasp_align_seconds < CONTROL_DT:
        raise ValueError("Near grasp alignment must last at least one control tick")
    if not isinstance(near_grasp_settle_ticks, (int, np.integer)) or near_grasp_settle_ticks < 0:
        raise ValueError("Near grasp settling ticks must be a nonnegative integer")
    if lateral_alignment_min_depth_m is not None and (
            not align_near_grasp_at_current_depth or not np.isfinite(lateral_alignment_min_depth_m)
            or lateral_alignment_min_depth_m <= 0.):
        raise ValueError("Lateral clearance requires current-depth alignment and a positive depth")
    if wrist_clearance_rear_distance_m is not None and (
            not align_from_clearance or not .07 <= wrist_clearance_rear_distance_m <= .18):
        raise ValueError("Rear clearance requires wrist alignment and a distance in [0.07,0.18] m")
    if wrist_clearance_extra_lift_m is not None and (
            not align_from_clearance or not np.isfinite(wrist_clearance_extra_lift_m) or wrist_clearance_extra_lift_m < 0.):
        raise ValueError("Wrist clearance lift requires alignment and a nonnegative height")
    if wrist_clearance_align_at_grasp_height and (not align_from_clearance or wrist_clearance_extra_lift_m != 0.):
        raise ValueError("Alignment at grasp height requires wrist clearance with zero additional lift")
    if wrist_clearance_open_after_retreat and not align_from_clearance:
        raise ValueError("Delayed opening requires wrist clearance alignment")
    if wrist_clearance_align_before_lift and not align_from_clearance:
        raise ValueError("Early wrist rotation requires wrist clearance alignment")
    if wrist_clearance_retreat_from_reset and not (
            align_from_clearance or align_near_grasp_at_current_depth or (start_holding and not skip_retreat)):
        raise ValueError("Reset-relative retreat requires wrist clearance alignment or held transport recovery")
    if not np.isfinite(release_retreat_m) or not 0. <= release_retreat_m <= .10:
        raise ValueError("Release retreat must be within [0,0.10] m")
    if not np.isfinite(release_retreat_seconds) or release_retreat_seconds <= 0.:
        raise ValueError("Release retreat duration must be finite and positive")
    if release_retreat_m and not release_single_contact:
        raise ValueError("Release retreat requires single-contact recovery")
    if release_retreat_before_open and not release_retreat_m:
        raise ValueError("Retreat before opening requires a single-contact retreat distance")
    if release_retreat_along_contact_normal and not release_retreat_before_open:
        raise ValueError("Retreat along the contact normal requires retreating before opening")
    if not np.isfinite(release_lift_m) or not 0. <= release_lift_m <= .06:
        raise ValueError("Release lift must be within [0,0.06] m")
    if release_lift_m and (not release_single_contact or release_retreat_m):
        raise ValueError("Release lift requires single-contact recovery without a planar release retreat")
    if ((release_lift_m or wrist_clearance_retreat_from_reset)
            and not np.isfinite((initial_state_provenance or {}).get('reset_base_y_m', np.nan))):
        raise ValueError("Reset-relative retreat requires the source episode's reset base coordinate")
    if not np.isfinite(eye_alignment_target_px) or not .5<=eye_alignment_target_px<=2.:
        raise ValueError("Eye alignment controller target must be within [0.5,2] pixels")
    if not isinstance(stable_read_samples, int) or stable_read_samples < 3:
        raise ValueError("Teacher readout requires at least three stable samples")
    read_lateral_retry_offsets_m=tuple(read_lateral_retry_offsets_m)
    if not all(np.isfinite(offset) for offset in read_lateral_retry_offsets_m):
        raise ValueError("Reading retry offsets must be finite")
    if approach_base_reference is not None:
        approach_base_reference = np.asarray(approach_base_reference, dtype=float)
        if (reset_scene or start_holding or release_single_contact
                or approach_base_reference.shape != (3,) or not np.isfinite(approach_base_reference).all()):
            raise ValueError("Approach base recovery requires finite reset odometry and a saved unheld state")
    if align_heading_before_base_advance and approach_base_reference is None:
        raise ValueError("Aligning before base advance requires a base recovery reference")
    if approach_heading_reference is not None and (
            not align_from_clearance or approach_base_reference is not None
            or release_single_contact or not np.isfinite(approach_heading_reference)):
        raise ValueError("Heading recovery requires a finite yaw reference and a saved clearance approach")
    if hold_wrist_on_heading_return and approach_heading_reference is None:
        raise ValueError("Holding the wrist during heading recovery requires a heading reference")
    approach_height = env.grasp_height
    if teacher_grasp_height_m is not None:
        # The existing cylinder collision wall spans 6–245 mm above the base.
        if start_holding or not np.isfinite(teacher_grasp_height_m) or not .006 <= teacher_grasp_height_m <= .245:
            raise ValueError("Teacher grasp height requires a new grasp at the cylinder wall")
        approach_height = float(teacher_grasp_height_m)
    geometry_report=None
    if env.grip_profile=="official":
        from audit_gripper import audit
        geometry_report=audit(env)
        (output/"geometry_audit.json").write_text(json.dumps(geometry_report,indent=2)+"\n")
    if reset_scene:env.reset_inspection(seed=seed,jitter=jitter,yaw_jitter=yaw_jitter)
    elif not initial_state_provenance or abs(env.data.time)>1e-9:
        raise ValueError("Recovery demonstrations require provenance and an episode-relative clock")
    if start_holding and reset_scene:raise ValueError("Held continuation requires preserved physics")
    if skip_retreat and not start_holding:raise ValueError("Reading recovery requires an existing grasp")
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]],dtype=float)
    c,s=np.cos(grasp_roll),np.sin(grasp_roll)
    rotation=rotation@np.array([[1.,0,0],[0,c,-s],[0,s,c]])
    if start_holding:rotation=env.data.xmat[env.gripper_body].reshape(3,3).copy()
    cylinder=env.data.xpos[env.cylinder_body].copy()
    original_z=float(cylinder[2])
    initial_eye=env.data.cam_xpos[env.model.camera("face_view").id].copy()
    initial_distance=float(np.linalg.norm(cylinder+[0,0,.125]-initial_eye))
    rows=[];writers={};phase_log=[];held=False;lost_seconds=0.0;max_slip=0.0
    max_tilt=0.0;max_contact_loss=0.0
    readings=[];view_search=[]
    reading_retry_reason=None
    lateral_read_attempts=[]
    extra_stillness_samples = 0
    if record:
        for cam in ("cylinder_overview","face_view"):
            writers[(cam,env.image_size)]=imageio.get_writer(output/f"{cam}.mp4",fps=20,codec="libx264",quality=8,macro_block_size=1)
    if record_training:
        for cam in CAMERAS:
            writers[(cam,training_image_size)]=imageio.get_writer(output/f"training_{cam}.mp4",fps=20,codec="libx264",quality=9,macro_block_size=1,ffmpeg_params=["-threads","1"])
    def tick(phase,q,opening,forward=0,yaw=0):
        nonlocal lost_seconds,max_slip,max_tilt,max_contact_loss
        before=env.state_vector();t=float(env.data.time)
        before_qpos=env.data.qpos.copy()
        before_metrics=env.metrics();before_cylinder=env.data.xpos[env.cylinder_body].copy()
        before_relative=env.grip_point_relative();before_base=env.base_pose().copy()
        before_quaternion=env.data.xquat[env.cylinder_body].copy()
        for (cam,size),writer in writers.items():writer.append_data(env.render(cam,width=size))
        if held and upright_gain>0:
            # Privileged demonstration teacher only. The learned policy must
            # predict this correction from its RGB/proprioception inputs.
            axis=env.data.xmat[env.cylinder_body].reshape(3,3)[:,2]
            tilt=float(np.arccos(np.clip(axis[2],-1,1)))
            if tilt>.003:
                direction=np.cross(axis,[0.,0.,1.]);direction/=np.linalg.norm(direction)
                quat=np.empty(4);mujoco.mju_axisAngle2Quat(quat,direction,min(.18,upright_gain*tilt))
                correction=np.empty(9);mujoco.mju_quat2Mat(correction,quat)
                planned=env.ik_data;planned.qpos[:]=env.data.qpos;planned.qpos[env.arm_addresses]=q
                mujoco.mj_kinematics(env.model,planned)
                planned_rotation=planned.xmat[env.gripper_body].reshape(3,3).copy()
                point=planned.xpos[env.gripper_body]+planned_rotation@env.grasp_offset
                q,error=env.inverse_kinematics(point,correction.reshape(3,3)@planned_rotation,
                                                initial=env.data.qpos[env.arm_addresses])
                if error>.002:raise InspectionFailure("upright_feedback_unreachable")
        _,metrics=env.step_joints(q,opening=opening,forward=forward,yaw=yaw)
        commanded=np.array([env.hold_targets[env.model.actuator(f"right_arm_joint{i}_servo").id] for i in range(1,7)])
        commanded_opening=env.hold_targets[env.model.actuator("right_arm_gripper_left_joint_servo").id]
        cartesian_action=commanded_cartesian_action(env,before_qpos)
        cartesian_action[19]=2*opening/.0975
        rows.append(dict(time=t,state=before,joints=np.r_[commanded,commanded_opening,env.applied_action],
                         cylinder=before_cylinder,phase=phase,
                         action=cartesian_action,
                         relative_point=before_relative,base_pose=before_base,
                         cylinder_quaternion=before_quaternion,
                         finger_forces=before_metrics["finger_normal_forces_n"],
                         slip=before_metrics["bottle_slip_m"],tilt=before_metrics["bottle_tilt_rad"],contacts=before_metrics["grip_contacts"]))
        if not np.isfinite(env.data.qpos).all():raise InspectionFailure("nonfinite_state")
        if metrics["forbidden_contacts"]:raise InspectionFailure(f"collision:{metrics['forbidden_contacts']}")
        if (align_from_clearance or align_near_grasp) and (
                phase.startswith("wrist_clearance_") or phase in ("approach", "advance")):
            shift = np.linalg.norm(env.data.xpos[env.cylinder_body]-cylinder)
            if metrics["grip_contacts"] or shift > .005:
                raise InspectionFailure("wrist_alignment_disturbed_cylinder")
        if held:
            max_slip=max(max_slip,metrics["bottle_slip_m"])
            max_tilt=max(max_tilt,metrics["bottle_tilt_rad"])
            lost_seconds=lost_seconds+CONTROL_DT if metrics["grip_contacts"]<2 else 0
            max_contact_loss=max(max_contact_loss,lost_seconds)
            # A round cylinder may roll/recenter inside the 32 mm pads while
            # both fingers remain loaded. Bound that transient motion, then
            # require the cylinder and image reading to be still at the end.
            if lost_seconds>.2 or metrics["bottle_slip_m"]>.025:raise InspectionFailure("lost_grasp")
            if metrics["bottle_tilt_rad"]>.12:raise InspectionFailure("cylinder_tilt")
    def move(phase,point,seconds=2.0,opening=None,*,scale=None,straight=False,settle_ticks=10):
        if opening is None:opening=grip_opening
        q0=env.data.qpos[env.arm_addresses].copy()
        p0=env.data.xpos[env.gripper_body]+env.data.xmat[env.gripper_body].reshape(3,3)@env.grasp_offset
        rotation0=env.data.xmat[env.gripper_body].reshape(3,3).copy()
        delta_quat=np.empty(4);mujoco.mju_mat2Quat(delta_quat,(rotation@rotation0.T).reshape(-1))
        rotvec=np.empty(3);mujoco.mju_quat2Vel(rotvec,delta_quat,1.0)
        angle=float(np.linalg.norm(rotvec))
        q1,error=env.inverse_kinematics(np.asarray(point),rotation,initial=q0)
        if error>.002:raise InspectionFailure(f"unreachable:{phase}:{error:.4f}")
        duration=max(seconds,1.875*np.max(abs(q1-q0))/.45)*(motion_scale if scale is None else scale)
        count=int(np.ceil(duration/CONTROL_DT))
        phase_log.append(dict(phase=phase,start_seconds=float(env.data.time),target=list(point),ik_error=error))
        print(json.dumps(phase_log[-1]),flush=True)
        for step in range(1,count+1):
            s=step/count;s=10*s**3-15*s**4+6*s**5
            if held or straight:
                target=p0+(np.asarray(point)-p0)*s
                if angle>1e-8:
                    quat=np.empty(4);mujoco.mju_axisAngle2Quat(quat,rotvec/angle,angle*s)
                    delta=np.empty(9);mujoco.mju_quat2Mat(delta,quat)
                    current_rotation=delta.reshape(3,3)@rotation0
                else:current_rotation=rotation
                q,error=env.inverse_kinematics(target,current_rotation,initial=q0)
                if error>.002:raise InspectionFailure(f"path_unreachable:{phase}:{error:.4f}")
                q0=q
            else:q=q0+(q1-q0)*s
            tick(phase,q,opening)
        if held or straight:q1=q
        for _ in range(settle_ticks):tick(phase,q1,opening)
        return q1

    def align_wrist_at_clearance():
        nonlocal rotation
        if env.metrics()["grip_contacts"]:
            raise InspectionFailure("wrist_alignment_requires_contact_free_start")
        nominal = rotation
        rotation = env.data.xmat[env.gripper_body].reshape(3,3).copy()
        tip = env.data.xpos[env.gripper_body]+rotation@env.grasp_offset
        clear = tip.copy()
        rear_distance = .14 if wrist_clearance_rear_distance_m is None else wrist_clearance_rear_distance_m
        clear[1] = min(tip[1], cylinder[1]-rear_distance)
        if tip[1]-clear[1] > .002:
            opening = .034
            if wrist_clearance_open_after_retreat:
                # Keep the previous servo target until the fingers have cleared the cylinder.
                gripper_servo = env.model.actuator("right_arm_gripper_left_joint_servo").id
                opening = env.hold_targets[gripper_servo]
            move("wrist_clearance_back",clear,1.5,opening=opening,straight=True)
        if wrist_clearance_align_before_lift:
            rotation = nominal
            move("wrist_clearance_rotate",clear,2,opening=.034,straight=True)
        lift = .045 if wrist_clearance_extra_lift_m is None else wrist_clearance_extra_lift_m
        safe_height = cylinder[2]+approach_height+lift
        if clear[2] < safe_height-.002:
            clear[2] = safe_height
            move("wrist_clearance_up",clear,1.5,opening=.034,straight=True)
        elif wrist_clearance_align_at_grasp_height:
            clear[2] = safe_height
        rotation = nominal
        clear[0] = cylinder[0]
        move("wrist_clearance_align",clear,2,opening=.034,straight=True)
        return clear

    def tuck_for_base_motion(phase, check_clearance):
        # A measured joint can briefly lie outside the allowed target range.
        q0 = np.clip(env.data.qpos[env.arm_addresses], *env.ik_joint_bounds())
        q = np.asarray(CARRY_JOINTS)
        count = max(1,int(np.ceil(1.875*np.max(np.abs(q-q0))/(.45*CONTROL_DT))))
        phase_log.append(dict(phase=phase,start_seconds=float(env.data.time),target=q.tolist()))
        for step in range(1,count+1):
            fraction = step/count
            fraction = 10*fraction**3-15*fraction**4+6*fraction**5
            tick(phase,q0+(q-q0)*fraction,.034)
            check_clearance()
        return q

    def restore_approach_heading(reference):
        nonlocal rotation
        phase = "restore_approach_heading"
        if env.metrics()["grip_contacts"]:
            raise InspectionFailure("heading_recovery_requires_contact_free_start")

        def require_clearance():
            if (env.metrics()["grip_contacts"]
                    or np.linalg.norm(env.data.xpos[env.cylinder_body]-cylinder) > .005):
                raise InspectionFailure("heading_recovery_disturbed_cylinder")

        if hold_wrist_on_heading_return:
            nominal = rotation
            rotation = env.data.xmat[env.gripper_body].reshape(3,3).copy()
            tip = env.data.xpos[env.gripper_body] + rotation @ env.grasp_offset
            clearance = tip.copy()
            clearance[1] = min(tip[1], cylinder[1] - .14)
            if tip[1] - clearance[1] > .002:
                gripper = env.model.actuator("right_arm_gripper_left_joint_servo").id
                move("wrist_clearance_heading_retreat",clearance,1.5,
                     opening=env.hold_targets[gripper],straight=True)
            rotation = nominal
            q = env.data.qpos[env.arm_addresses].copy()
            wrist_rotation = env.data.xmat[env.gripper_body].reshape(3,3).copy()
            wrist_tip = env.data.xpos[env.gripper_body] + wrist_rotation @ env.grasp_offset
        else:
            q = tuck_for_base_motion("tuck_for_heading_return", require_clearance)
        start = env.base_pose()
        attempt = dict(phase=phase,start_seconds=float(env.data.time),start_pose=start.tolist(),
                       target_yaw_rad=float(reference))
        phase_log.append(attempt)
        stable = 0
        for _ in range(120):
            error = angle_difference(reference, env.base_pose()[2])
            yaw = float(np.clip(2.2*error,-.16,.16)) if abs(error) > .006 else 0.
            if hold_wrist_on_heading_return:
                q, residual = env.inverse_kinematics(wrist_tip, wrist_rotation, initial=q)
                if residual > .002:
                    raise InspectionFailure(f"wrist_hold_unreachable_during_heading_return:{residual:.4f}")
            tick(phase,q,.034,yaw=yaw)
            require_clearance()
            pose = env.base_pose()
            if np.linalg.norm(pose[:2]-start[:2]) > .008:
                raise InspectionFailure("heading_recovery_base_translation")
            settled = abs(angle_difference(reference, pose[2])) <= .006 and abs(env.base_velocity()[2]) < .02
            stable = stable + 1 if settled else 0
            if stable >= 5:
                attempt.update(end_seconds=float(env.data.time),end_pose=pose.tolist())
                return
        raise InspectionFailure("heading_recovery_timeout")

    def restore_approach_base(reference):
        phase = "restore_approach_base"
        direction = np.array([np.cos(reference[2]), np.sin(reference[2])])
        left = np.array([-direction[1], direction[0]])
        initial_pose = env.base_pose()
        initial_distance = float((reference[:2]-initial_pose[:2]) @ direction)
        if initial_distance <= .02 or env.metrics()["grip_contacts"]:
            raise InspectionFailure("approach_base_recovery_requires_contact_free_retreat")
        def require_clearance():
            if (env.metrics()["grip_contacts"]
                    or np.linalg.norm(env.data.xpos[env.cylinder_body]-cylinder) > .005):
                raise InspectionFailure("approach_base_recovery_disturbed_cylinder")

        q = tuck_for_base_motion("tuck_for_base_return", require_clearance)
        attempt = dict(phase=phase,start_seconds=float(env.data.time),start_pose=env.base_pose().tolist(),
                       target=reference.tolist())
        phase_log.append(attempt)
        for _ in range(200):
            pose = env.base_pose()
            displacement = reference[:2]-pose[:2]
            distance, lateral = float(displacement@direction), float(displacement@left)
            heading_offset = np.arctan2(lateral,max(.09,distance)) * np.clip((distance-.015)/.06,0,1)
            heading_error = reference[2]+heading_offset-pose[2]
            heading_error = np.arctan2(np.sin(heading_error),np.cos(heading_error))
            yaw = float(np.clip(1.8*heading_error,-.15,.15))
            speed = min(.09, max(0., 1.6*(distance-.008))) if distance > .012 else 0.
            if align_heading_before_base_advance:
                # Avoid accumulating lateral error while the base is still turning.
                speed *= float(np.clip((.12-abs(heading_error))/.09,0.,1.))
            tick(phase,q,.034,forward=speed,yaw=yaw)
            require_clearance()
            velocity = env.base_velocity()
            if abs(distance) <= .015 and abs(lateral) <= .025 and abs(yaw) < .04 and np.max(np.abs(velocity)) < .01:
                attempt["end_seconds"] = float(env.data.time)
                attempt["end_pose"] = env.base_pose().tolist()
                break
        else:
            raise InspectionFailure("approach_base_recovery_timeout")

    def finger_contact_force():
        total = 0.
        outward = np.zeros(3)
        force = np.empty(6)
        for i, contact in enumerate(env.data.contact):
            first, second = env.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if first == env.cylinder_body and second in env.finger_bodies:
                sign = 1.
            elif second == env.cylinder_body and first in env.finger_bodies:
                sign = -1.
            else:
                continue
            mujoco.mj_contactForce(env.model, env.data, i, force)
            if force[0] > .05:
                total += force[0]
                outward += sign * force[0] * contact.frame[:3]
        return total, outward

    def withdraw_from_single_contact(opening):
        nonlocal rotation
        nominal_rotation = rotation
        rotation = env.data.xmat[env.gripper_body].reshape(3, 3).copy()
        tip = env.data.xpos[env.gripper_body] + rotation @ env.grasp_offset
        if release_retreat_along_contact_normal:
            _, away = finger_contact_force()
        else:
            away = -(rotation @ (env.grip_point_relative() - env.grasp_offset))
            away[2] = 0.
        distance = np.linalg.norm(away)
        if distance <= .001:
            raise InspectionFailure("single_contact_retreat_direction_undefined")
        target = tip + release_retreat_m * away / distance
        joints = move("withdraw_from_contact", target, seconds=release_retreat_seconds, opening=opening, straight=True)
        rotation = nominal_rotation
        return joints

    reason=None
    try:
        if start_holding:
            if env.metrics()["grip_contacts"]!=2:raise InspectionFailure("held_continuation_requires_two_contacts")
            env.grasp_reference=env._bottle_relative_position();held=True
            if not read_from_current_view or env.grasp_point_reference is None:
                env.grasp_point_reference=env.grip_point_relative()
            point=env.data.xpos[env.gripper_body]+rotation@env.grasp_offset
            if held_initial_lift_m:
                point=point+np.array([0.,0.,held_initial_lift_m])
                move("lift_from_partial_grasp",point,2)
            q=env.data.qpos[env.arm_addresses].copy() if skip_retreat else move("retract",point+[0,-.15,0],3)
        else:
            if approach_base_reference is not None:
                restore_approach_base(approach_base_reference)
            if release_single_contact:
                if env.metrics()["grip_contacts"] != 1:
                    raise InspectionFailure("release_requires_one_loaded_finger")
                q = env.data.qpos[env.arm_addresses].copy()
                cylinder_dof = env.model.joint("cylinder_free").dofadr[0]
                if release_lift_m:
                    total_force, outward = finger_contact_force()
                    if total_force <= 0. or outward[2] < .8*total_force:
                        raise InspectionFailure("release_lift_requires_upward_contact_normal")
                    nominal_rotation = rotation
                    rotation = env.data.xmat[env.gripper_body].reshape(3,3).copy()
                    tip = env.data.xpos[env.gripper_body] + rotation @ env.grasp_offset
                    q = move("withdraw_from_contact", tip+[0,0,release_lift_m],1.2,opening=.034,straight=True)
                    rotation = nominal_rotation
                if release_retreat_before_open:
                    gripper_servo = env.model.actuator("right_arm_gripper_left_joint_servo").id
                    q = withdraw_from_single_contact(env.hold_targets[gripper_servo])
                stationary_ticks = 0
                for tick_index in range(30):
                    if (tick_index == 12 and release_retreat_m and not release_retreat_before_open
                            and env.metrics()["grip_contacts"]):
                        q = withdraw_from_single_contact(.034)
                    tick("release_contact", q, .034)
                    metrics = env.metrics()
                    settled = (tick_index >= 11 and metrics["grip_contacts"] == 0
                               and metrics["bottle_tilt_rad"] < .10
                               and np.linalg.norm(env.data.qvel[cylinder_dof:cylinder_dof + 3]) < .01
                               and np.linalg.norm(env.data.qvel[cylinder_dof + 3:cylinder_dof + 6]) < .2)
                    stationary_ticks = stationary_ticks + 1 if settled else 0
                    if stationary_ticks == 3:
                        break
                else:
                    raise InspectionFailure("single_contact_release_did_not_settle")
                cylinder = env.data.xpos[env.cylinder_body].copy()
            if approach_heading_reference is not None:
                restore_approach_heading(approach_heading_reference)
            if align_near_grasp:
                error = env.grip_point_relative() - env.grasp_offset
                forward = (align_near_grasp_forward and 0. < error[0] <= .065
                           and np.linalg.norm(error[1:]) <= .012)
                if align_near_grasp_at_current_depth:
                    measured_rotation = env.data.xmat[env.gripper_body].reshape(3,3)
                    measured_tip = env.data.xpos[env.gripper_body] + measured_rotation @ env.grasp_offset
                    axis = env.data.xmat[env.cylinder_body].reshape(3,3)[:,2]
                    delta = env.data.xpos[env.cylinder_body] + approach_height * axis - measured_tip
                    gap = (env.data.qpos[env.model.joint("right_arm_gripper_left_joint").qposadr[0]]
                           - env.data.qpos[env.model.joint("right_arm_gripper_right_joint").qposadr[0]])
                    near = .025 <= delta[1] <= .075 and np.linalg.norm(delta[[0, 2]]) <= .065
                    # A higher start needs more clearance before the wrist can lower.
                    high = .040 <= delta[1] <= .075 and abs(delta[0]) <= .025 and .065 < -delta[2] <= .170
                    if (not (near or high) or gap < .035
                            or env.metrics()["grip_contacts"]):
                        raise InspectionFailure("current_depth_alignment_requires_open_fingers_and_forward_clearance")
                    alignment_depth = measured_tip[1]
                elif (np.linalg.norm(error) > .03 and not forward) or env.metrics()["grip_contacts"]:
                    reason = ("near_grasp_alignment_requires_clear_fingers_in_approach_range" if align_near_grasp_forward
                              else "near_grasp_alignment_requires_clear_fingers_within_30mm")
                    raise InspectionFailure(reason)
                q = np.array([env.hold_targets[env.model.actuator(f"right_arm_joint{i}_servo").id]
                              for i in range(1,7)])
                if lateral_alignment_min_depth_m is not None and abs(delta[0]) >= .030:
                    clearance = lateral_alignment_min_depth_m - delta[1]
                    if clearance > 0.:
                        # Withdraw at the measured aperture so opening cannot sweep a pad into the wall.
                        nominal = rotation
                        rotation = measured_rotation.copy()
                        rear_tip = measured_tip.copy()
                        rear_tip[1] -= clearance
                        q = move("wrist_clearance_near_rear",rear_tip,1.,opening=gap*.5,straight=True)
                        rotation = nominal
                        alignment_depth = rear_tip[1]
                for _ in range(near_grasp_open_ticks):tick("wrist_clearance_near_open",q,.034)
                axis = env.data.xmat[env.cylinder_body].reshape(3,3)[:,2]
                target = env.data.xpos[env.cylinder_body] + approach_height * axis
                if align_near_grasp_at_current_depth:
                    alignment = target.copy()
                    alignment[1] = alignment_depth
                    # Align behind the cylinder before advancing toward it.
                    move("wrist_clearance_near_plane",alignment,1.5,opening=.034,straight=True)
                q = move("wrist_clearance_near_align",target,near_grasp_align_seconds,opening=.034,straight=True,
                         settle_ticks=near_grasp_settle_ticks)
            else:
                aligned = align_wrist_at_clearance() if align_from_clearance else None
                approach_distance = .10 if wrist_clearance_rear_distance_m is None else min(.10, wrist_clearance_rear_distance_m)
                approach=cylinder+[0,-approach_distance,approach_height]
                if approach_heading_reference is not None:
                    wrist_rotation = env.data.xmat[env.gripper_body].reshape(3,3)
                    tip = env.data.xpos[env.gripper_body] + wrist_rotation@env.grasp_offset
                    if tip[2] > approach[2] + .002:
                        # Lower while the wrist is still behind the tray after the tucked turn.
                        tip[2] = approach[2]
                        move("wrist_clearance_lower",tip,1.5,opening=.034,straight=True)
                if not wrist_clearance_align_at_grasp_height or not np.array_equal(aligned, approach):
                    move("approach",approach,opening=.034,scale=approach_motion_scale,straight=align_from_clearance)
                q=move("advance",cylinder+[0,0,approach_height],2,opening=.034,straight=align_from_clearance)
            for _ in range(grasp_close_ticks):tick("close",q,grip_opening)
            if env.metrics()["grip_contacts"]!=2:raise InspectionFailure("grasp_not_established")
            env.grasp_reference=env._bottle_relative_position();held=True
            env.grasp_point_reference=env.grip_point_relative()
            move("lift",cylinder+[0,0,approach_height+.04],initial_lift_seconds)
            q=move("retract",cylinder+[0,-.15,approach_height+.12],3)
        if not skip_retreat:
            target_y=env.base_pose()[1]-.24
            if release_lift_m or wrist_clearance_retreat_from_reset:
                target_y = min(target_y, initial_state_provenance['reset_base_y_m']-.24)
            phase_log.append(dict(phase="back_away", start_seconds=float(env.data.time),
                                  target_base_y_m=float(target_y)))
            for _ in range(150):
                distance=env.base_pose()[1]-target_y
                if distance<.008:break
                tick("back_away",q,grip_opening,-min(.10,1.5*distance))
            else:raise InspectionFailure("base_retreat_timeout")
            for _ in range(15):tick("back_away",q,grip_opening)
        camera_id=env.model.camera("face_view").id
        use_current_view=read_from_current_view
        view_right=lateral_moves=None

        def settle_at_current_view():
            nonlocal grip_opening, rotation, point, view_right, lateral_moves
            open_servo=env.model.actuator("right_arm_gripper_left_joint_servo").id
            grip_opening=env.hold_targets[open_servo]
            hold_joints=np.array([env.hold_targets[env.model.actuator(f"right_arm_joint{i}_servo").id]
                                  for i in range(1,7)])
            for _ in range(24):tick("hold_for_reading",hold_joints,grip_opening)
            rotation=env.data.xmat[env.gripper_body].reshape(3,3).copy()
            point=env.data.xpos[env.gripper_body]+rotation@env.grasp_offset
            view_right=env.data.cam_xmat[camera_id].reshape(3,3)[:,0].copy()
            view_right[2]=0.
            view_right/=np.linalg.norm(view_right)
            lateral_moves=iter((.012,-.024,.037,-.05))
            return hold_joints

        if use_current_view:
            q=settle_at_current_view()
        else:
            eye=env.data.cam_xpos[camera_id].copy()
            point=eye+[read_lateral,read_distance,approach_height-(SCALE_ZERO+SCALE_SPAN/2)]
            if held_presentation_waypoints:
                # Correct the preserved grasp before translating it toward the camera.
                measured_rotation=env.data.xmat[env.gripper_body].reshape(3,3).copy()
                wrist_point=env.data.xpos[env.gripper_body]+measured_rotation@env.grasp_offset
                cylinder_axis=env.data.xmat[env.cylinder_body].reshape(3,3)[:,2]
                tilt=float(np.arccos(np.clip(cylinder_axis[2],-1.,1.)))
                rotation=measured_rotation
                if tilt>.003:
                    correction_axis=np.cross(cylinder_axis,[0.,0.,1.])
                    correction_axis/=np.linalg.norm(correction_axis)
                    quat=np.empty(4)
                    mujoco.mju_axisAngle2Quat(quat,correction_axis,min(.07,.8*tilt))
                    correction=np.empty(9)
                    mujoco.mju_quat2Mat(correction,quat)
                    rotation=correction.reshape(3,3)@measured_rotation
                    move("upright_before_present",wrist_point,1.5)
                wrist_point=env.data.xpos[env.gripper_body]+env.data.xmat[env.gripper_body].reshape(3,3)@env.grasp_offset
                move("present_sideways",np.array([point[0],wrist_point[1],wrist_point[2]]),1.6)
                move("present_depth",np.array([point[0],point[1],wrist_point[2]]),2.)
                q=move("present_height",point,1.6)
            else:
                q=move("present",point,3)
            for _ in range(20):tick("hold_for_reading",q,grip_opening)

        def move_camera_offset(phase, offset):
            nonlocal rotation, point
            rotation=env.data.xmat[env.gripper_body].reshape(3,3).copy()
            tip=env.data.xpos[env.gripper_body]+rotation@env.grasp_offset
            point=tip+offset
            return move(phase,point,1.,straight=True)
        def straighten_for_reading(q):
            nonlocal rotation
            # The teacher uses the object pose to correct tilt through joint commands.
            for _ in range(4):
                axis=env.data.xmat[env.cylinder_body].reshape(3,3)[:,2]
                tilt=float(np.arccos(np.clip(axis[2],-1,1)))
                if tilt<np.deg2rad(3 if use_current_view else 1):break
                correction_axis=np.cross(axis,[0.,0.,1.])
                correction_axis/=np.linalg.norm(correction_axis)
                quat=np.empty(4);mujoco.mju_axisAngle2Quat(quat,correction_axis,.8*tilt)
                correction=np.empty(9);mujoco.mju_quat2Mat(correction,quat)
                rotation=correction.reshape(3,3)@env.data.xmat[env.gripper_body].reshape(3,3)
                q=move("straighten",point,1.5)
                for _ in range(10):tick("straighten",q,grip_opening)
            return q
        def needs_view(reading):
            if not reading["valid"]:return True
            # Coarse height alignment may observe a curved projected cap.
            # At eye level, require a majority of the visible liquid width
            # to support the boundary, rather than a few unoccluded pixels.
            return (abs(reading["eye_alignment_error_px"])<=2
                    and reading["boundary_support_fraction"]<.5)
        def align_camera_reading(q, attempt):
            nonlocal rotation, point
            for iteration in range(11 if use_current_view else 6):
                suffix = f"{attempt}_{iteration:02d}" if attempt else f"{iteration:02d}"
                rgb=env.render("face_view")
                reading=read_volume(rgb)
                if needs_view(reading):
                    view_search.append(dict(time=float(env.data.time),iteration=iteration,reading=reading))
                    imageio.imwrite(output/f"view_rejected_{suffix}.png",annotate(rgb,reading))
                    if use_current_view:
                        for lateral in lateral_moves:
                            q=move_camera_offset("find_scale",lateral*view_right)
                            rgb=env.render("face_view");reading=read_volume(rgb)
                            if not needs_view(reading):break
                    else:
                        rotation_seed=rotation.copy()
                        for yaw in (.14,-.14,.28,-.28):
                            c,s=np.cos(yaw),np.sin(yaw)
                            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])@rotation_seed
                            q=move("find_scale",point,1.2)
                            for _ in range(10):tick("find_scale",q,grip_opening)
                            rgb=env.render("face_view");reading=read_volume(rgb)
                            if not needs_view(reading):break
                readings.append(dict(time=float(env.data.time),**reading))
                imageio.imwrite(output/f"reading_{suffix}.png",annotate(rgb,reading))
                print(json.dumps(readings[-1]),flush=True)
                if not reading["valid"]:raise ReadingFailure(f"unreadable:{reading['reason']}")
                if needs_view(reading):raise ReadingFailure("liquid_boundary_occluded")
                if abs(reading["eye_alignment_error_px"])<=eye_alignment_target_px:return q
                focal=env.image_size/(2*np.tan(np.deg2rad(env.model.cam_fovy[camera_id])/2))
                limit=.012 if use_current_view else .025
                dz=np.clip(reading["eye_alignment_error_px"]*read_distance/focal,-limit,limit)
                if use_current_view:
                    q=move_camera_offset("align_eye_level",np.array([0.,0.,dz]))
                else:
                    point[2]+=dz
                    q=move("align_eye_level",point,1.2)
                    for _ in range(10):tick("align_eye_level",q,grip_opening)
            raise ReadingFailure("eye_alignment_timeout")

        def hold_for_stable_reading(q, sample_count=stable_read_samples):
            # Only consecutive observations after the latest alignment count.
            for _ in range(sample_count):
                for _ in range(5):tick("read_stable",q,grip_opening)
                reading=read_volume(env.render("face_view"))
                readings.append(dict(time=float(env.data.time),**reading))
                if not reading["valid"]:return f"unstable_reading:{reading['reason']}"
                if needs_view(reading):return "unstable_boundary_visibility"
                if abs(reading["eye_alignment_error_px"])>2:return "eye_alignment_drift"
            return None

        def acquire_reading(q, first_attempt):
            nonlocal use_current_view, reading_retry_reason
            for attempt in range(3 if use_current_view or retry_reading_from_current_view else 1):
                q=align_camera_reading(q,first_attempt+attempt)
                read_failure=hold_for_stable_reading(q)
                if read_failure is None:
                    stable=[r["volume_ml"] for r in readings[-3:]]
                    if np.ptp(stable)>.4:raise ReadingFailure("reading_not_stable")
                    return
                if not use_current_view and retry_reading_from_current_view:
                    reading_retry_reason=read_failure
                    q=settle_at_current_view()
                    use_current_view=True
            raise ReadingFailure(read_failure)

        offsets=(0.,*read_lateral_retry_offsets_m)
        for index,offset in enumerate(offsets):
            lateral=read_lateral+offset
            if index:
                use_current_view=False
                # Translate sideways while holding the measured wrist angle and height.
                rotation=env.data.xmat[env.gripper_body].reshape(3,3).copy()
                point=env.data.xpos[env.gripper_body]+rotation@env.grasp_offset
                point[0]=env.data.cam_xpos[camera_id,0]+lateral
                q=move("reposition_for_reading",point,1.5,straight=True)
                for _ in range(20):tick("hold_for_reading",q,grip_opening)
            q=straighten_for_reading(q)
            attempt_start=float(env.data.time)
            try:
                acquire_reading(q,index*3)
            except ReadingFailure as failure:
                lateral_read_attempts.append(dict(lateral_m=lateral,start_seconds=attempt_start,failure=str(failure)))
                if index==len(offsets)-1:raise
            else:
                lateral_read_attempts.append(dict(lateral_m=lateral,start_seconds=attempt_start,failure=None))
                break
        if env.metrics()["bottle_tilt_rad"]>np.deg2rad(3):raise InspectionFailure("not_upright_for_reading")
        dof=env.model.joint("cylinder_free").dofadr[0]

        def cylinder_is_still():
            velocity = env.data.qvel[dof:dof+6]
            return np.linalg.norm(velocity[:3]) <= .005 and np.linalg.norm(velocity[3:]) <= .03

        if not cylinder_is_still():
            # Keep the command used for the last successful camera alignment.
            hold_joints = np.array([env.hold_targets[env.model.actuator(f"right_arm_joint{i}_servo").id]
                                    for i in range(1, 7)])
            consecutive = 0
            for extra_stillness_samples in range(1, 9):
                read_failure = hold_for_stable_reading(hold_joints, sample_count=1)
                metrics = env.metrics()
                if (read_failure is not None or not cylinder_is_still() or metrics["grip_contacts"] != 2
                        or metrics["bottle_tilt_rad"] > np.deg2rad(3)):
                    consecutive = 0
                    continue
                consecutive += 1
                if consecutive >= 3 and np.ptp([reading["volume_ml"] for reading in readings[-3:]]) <= .4:
                    break
            else:
                raise InspectionFailure("cylinder_not_still_for_reading")
    except InspectionFailure as failure:
        reason=str(failure)
    finally:
        for writer in writers.values():writer.close()
    imageio.imwrite(output/"final_face.png",env.render("face_view"))
    final_reading=read_volume(env.render("face_view"))
    imageio.imwrite(output/"reading_annotated.png",annotate(env.render("face_view"),final_reading))
    imageio.imwrite(output/"final_overview.png",env.render("cylinder_overview"))
    metrics=env.metrics()
    summary=dict(success=reason is None,failure_reason=reason,simulated_seconds=float(env.data.time),
                 initialization="scene_reset" if reset_scene else "preserved_policy_visited_physics",
                 initial_state_provenance=initial_state_provenance,
                 starts_holding=start_holding,
                 recovery_skips_existing_retreat=skip_retreat,
                 recovery_initial_lift_m=held_initial_lift_m,
                 lifted_m=float(env.data.xpos[env.cylinder_body,2]-original_z),
                 final_tilt_rad=metrics["bottle_tilt_rad"],max_slip_m=max_slip,
                 final_contacts=metrics["grip_contacts"],phases=phase_log,
                 last_control_phase=rows[-1]["phase"] if rows else None,
                 expected_volume_ml=env.volume_ml,reading=final_reading,reading_history=readings,
                 view_search_history=view_search,neq=env.model.neq,
                 policy="privileged_pose_teacher",perception="RGB only; 20/35 mL color calibration anchors",
                 grip_profile=env.grip_profile,
                 geometry_audit_passed=bool(geometry_report and geometry_report["passed"]),
                 controller_parameters=dict(grip_opening_m=grip_opening,motion_scale=motion_scale,
                                            grasp_height_m=env.grasp_height,grasp_x_m=float(env.grasp_offset[0]),
                                            read_lateral_m=read_lateral,read_distance_m=read_distance,
                                            eye_alignment_target_px=eye_alignment_target_px,
                                            grasp_roll_rad=grasp_roll,upright_feedback_gain=upright_gain),
                 grip_force_target_n=env.grip_force_target,
                 grasp_reference_point_m=None if env.grasp_point_reference is None else env.grasp_point_reference.tolist(),
                 hardware_model=("Pinned official parallel-finger collision meshes; no custom pads"
                                 if env.grip_profile=="official" else "Historical custom finger-pad proxy"),
                 contact_calibration="Uncalibrated simulation assumptions; no measured real-finger friction",
                 friction_scale=env.friction_scale,mass_scale=env.mass_scale,
                 initial_yaw_rad=env.initial_yaw,yaw_jitter_rad=yaw_jitter,
                 max_held_tilt_rad=max_tilt,max_contact_loss_seconds=max_contact_loss,
                 final_finger_normal_forces_n=metrics["finger_normal_forces_n"],
                 gripper_actuator_force_limit_n=max(env.finger_force_limits.values()),
                 gripper_force_limit_source=("Official URDF prismatic-joint effort limits"
                                             if env.grip_profile=="official" else "Historical experiment assumption"),
                 motion_teacher_inputs="Simulator cylinder pose for grasp and upright correction; robot/camera calibration",
                 collision_model="32 hollow wall segments and base; "+env.grip_profile,
                 contact_parameters=dict(noslip_iterations=int(env.model.opt.noslip_iterations),
                                         impratio=float(env.model.opt.impratio),physics_timestep=float(env.model.opt.timestep),
                                         finger_geoms=[dict(type=int(env.model.geom_type[g]),
                                             condim=int(env.model.geom_condim[g]),friction=env.model.geom_friction[g].tolist())
                                             for g in range(env.model.ngeom)
                                             if env.model.geom_bodyid[g] in env.finger_bodies and env.model.geom_contype[g]]),
                 max_lifted_m=max((float(row['cylinder'][2]-original_z) for row in rows),default=0.0),
                 initial_camera_distance_m=initial_distance,
                 final_camera_distance_m=float(np.linalg.norm(env.data.xpos[env.cylinder_body]
                    +env.data.xmat[env.cylinder_body].reshape(3,3)@np.array([0,0,.125])
                    -env.data.cam_xpos[env.model.camera("face_view").id])),
                 seed=seed,position_jitter_m=jitter,fps=20,frames=len(rows),
                 joint_command_columns=["right_joint1","right_joint2","right_joint3","right_joint4",
                                        "right_joint5","right_joint6","half_gripper_gap_m","base_vx","base_yaw_rate"],
                 source_sha256=INSPECTION_SOURCE_HASHES,
                 liquid_model="Rigid upright flat-surface rendering; no sloshing, refraction or meniscus dynamics")
    if approach_motion_scale is not None:
        summary["controller_parameters"]["approach_motion_scale"]=approach_motion_scale
    if not start_holding and grasp_close_ticks != 35:
        summary["controller_parameters"]["grasp_close_ticks"] = grasp_close_ticks
    if not start_holding and initial_lift_seconds != 2.:
        summary["controller_parameters"]["initial_lift_seconds"] = float(initial_lift_seconds)
    if held_presentation_waypoints:
        summary["controller_parameters"]["held_presentation_waypoints"] = True
    if wrist_clearance_rear_distance_m is not None:
        summary["controller_parameters"]["wrist_clearance_rear_distance_m"]=wrist_clearance_rear_distance_m
    if wrist_clearance_extra_lift_m is not None:
        summary["controller_parameters"]["wrist_clearance_extra_lift_m"]=wrist_clearance_extra_lift_m
    if wrist_clearance_align_at_grasp_height:
        summary["controller_parameters"]["wrist_clearance_align_at_grasp_height"]=True
    if align_near_grasp:
        summary["controller_parameters"]["align_near_grasp"]=True
        if near_grasp_open_ticks != 12:
            summary["controller_parameters"]["near_grasp_open_ticks"]=int(near_grasp_open_ticks)
        if near_grasp_align_seconds != 1.5:
            summary["controller_parameters"]["near_grasp_align_seconds"]=float(near_grasp_align_seconds)
        if near_grasp_settle_ticks != 10:
            summary["controller_parameters"]["near_grasp_settle_ticks"]=int(near_grasp_settle_ticks)
    if align_near_grasp_forward:
        summary["controller_parameters"]["align_near_grasp_forward"]=True
    if align_near_grasp_at_current_depth:
        summary["controller_parameters"]["align_near_grasp_at_current_depth"]=True
    if lateral_alignment_min_depth_m is not None:
        summary["controller_parameters"]["lateral_alignment_min_depth_m"]=lateral_alignment_min_depth_m
    if wrist_clearance_open_after_retreat:
        summary["controller_parameters"]["wrist_clearance_open_after_retreat"]=True
    if wrist_clearance_align_before_lift:
        summary["controller_parameters"]["wrist_clearance_align_before_lift"]=True
    if wrist_clearance_retreat_from_reset:
        summary["controller_parameters"]["wrist_clearance_retreat_from_reset"]=True
    if read_from_current_view:
        summary["controller_parameters"]["read_from_current_view"]=True
    if retry_reading_from_current_view:
        summary["controller_parameters"]["retry_reading_from_current_view"]=True
        summary["reading_retry_reason"]=reading_retry_reason
    if teacher_grasp_height_m is not None:
        summary["controller_parameters"]["teacher_grasp_height_m"]=approach_height
    if stable_read_samples != 3:
        summary["controller_parameters"]["stable_read_samples"]=stable_read_samples
    if read_lateral_retry_offsets_m:
        summary["controller_parameters"]["read_lateral_retry_offsets_m"]=list(read_lateral_retry_offsets_m)
        summary["lateral_read_attempts"]=lateral_read_attempts
    if release_single_contact:
        summary["controller_parameters"]["release_single_contact"]=True
        summary["controller_parameters"]["release_retreat_m"]=release_retreat_m
        if release_retreat_m and release_retreat_seconds != 1.5:
            summary["controller_parameters"]["release_retreat_seconds"]=release_retreat_seconds
        if release_retreat_before_open:
            summary["controller_parameters"]["release_retreat_before_open"]=True
        if release_retreat_along_contact_normal:
            summary["controller_parameters"]["release_retreat_along_contact_normal"]=True
        if release_lift_m:
            summary["controller_parameters"]["release_lift_m"]=release_lift_m
    if align_from_clearance:
        summary["controller_parameters"]["align_from_clearance"]=True
    if align_heading_before_base_advance:
        summary["controller_parameters"]["align_heading_before_base_advance"]=True
    if approach_heading_reference is not None:
        summary["controller_parameters"]["approach_heading_reference_rad"]=float(approach_heading_reference)
    if hold_wrist_on_heading_return:
        summary["controller_parameters"]["hold_wrist_on_heading_return"]=True
    if final_reading["valid"]:
        summary["absolute_reading_error_ml"]=abs(final_reading["volume_ml"]-env.volume_ml)
        if reason is None and summary["absolute_reading_error_ml"]>.5:
            summary["success"]=False;summary["failure_reason"]="reading_inaccurate"
    dof=env.model.joint("cylinder_free").dofadr[0]
    if extra_stillness_samples:
        summary["extra_stillness_samples"] = extra_stillness_samples
    summary["final_linear_speed_m_s"]=float(np.linalg.norm(env.data.qvel[dof:dof+3]))
    summary["final_angular_speed_rad_s"]=float(np.linalg.norm(env.data.qvel[dof+3:dof+6]))
    np.savez_compressed(output/"final_physics.npz",qpos=env.data.qpos,qvel=env.data.qvel,
                        ctrl=env.data.ctrl,time=env.data.time)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    metadata=dict(result=summary,instruction=CYLINDER_INSTRUCTION,frames=len(rows),fps=20,
                  recorded_cameras=list(CAMERAS) if record_training else [],video_prefix="training_",
                  robot_type="quanta_x1_official_cylinder",action_space="23D absolute Cartesian targets and body velocities",
                  action_label="FK of commanded joints, including the requested gripper preload; not measured future pose",
                  training_observations="RGB camera images and robot proprioception only; no object pose or fill value")
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    np.savez_compressed(output/"trajectory.npz",timestamp=np.array([r["time"] for r in rows]),
                        action=np.array([r["action"] for r in rows]),
                        state=np.array([r["state"] for r in rows]),joint_command=np.array([r["joints"] for r in rows]),
                        cylinder_position=np.array([r["cylinder"] for r in rows]),phase=np.array([r["phase"] for r in rows]),
                        grip_point_relative=np.array([r["relative_point"] for r in rows]),
                        cylinder_quaternion=np.array([r["cylinder_quaternion"] for r in rows]),
                        base_pose=np.array([r["base_pose"] for r in rows]),
                        slip=np.array([r["slip"] for r in rows]),tilt=np.array([r["tilt"] for r in rows]),
                        finger_normal_forces=np.array([r["finger_forces"] for r in rows]),
                        contacts=np.array([r["contacts"] for r in rows]))
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--volume",type=float,default=25)
    parser.add_argument("--probe-ik",action="store_true")
    parser.add_argument("--record",action="store_true")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--jitter",type=float,default=0.0)
    parser.add_argument("--grip-profile",choices=("official","flat","v-pad"),default="official",
                        help="official: pinned URDF collision meshes; flat/v-pad: historical custom proxies")
    parser.add_argument("--friction-scale",type=float,default=1.0)
    parser.add_argument("--mass-scale",type=float,default=1.0)
    parser.add_argument("--yaw-jitter",type=float,default=0.0,help="Maximum initial scale yaw perturbation in radians")
    parser.add_argument("--grip-opening",type=float,default=.010,help="Position-servo closure target; geometry and force limit unchanged")
    parser.add_argument("--motion-scale",type=float,default=1.,help="Trajectory duration multiplier")
    parser.add_argument("--grasp-height",type=float,default=GRIP_HEIGHT,help="Grasp point above cylinder base, m")
    parser.add_argument("--grasp-x",type=float,default=float(GRASP_OFFSET[0]),help="Tool point along finger, m")
    parser.add_argument("--solver-impratio",type=float,default=1.,help="Friction/normal impedance ratio; no change to friction coefficient")
    parser.add_argument("--physics-dt",type=float,default=PHYSICS_DT)
    parser.add_argument("--read-lateral",type=float,default=.14)
    parser.add_argument("--read-distance",type=float,default=READ_DISTANCE)
    parser.add_argument("--grasp-roll-deg",type=float,default=0.)
    parser.add_argument("--upright-gain",type=float,default=0.,help="Program-teacher object-pose feedback, never used by Wall execution")
    parser.add_argument("--record-training",action="store_true",help="Record three raw 224px training cameras aligned with Cartesian actions")
    parser.add_argument("--training-image-size",type=int,default=224,choices=(224,448))
    parser.add_argument("--grip-force-target",type=float,help="Robot-feedback force-limited closure; URDF force cap stays unchanged")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    env=CylinderEnv(args.volume,render=False,image_size=768,grip_profile=args.grip_profile,
                    friction_scale=args.friction_scale,mass_scale=args.mass_scale,
                    grasp_height=args.grasp_height,grasp_x=args.grasp_x,
                    solver_impratio=args.solver_impratio,physics_dt=args.physics_dt,grip_force_target=args.grip_force_target)
    try:
        if not args.probe_ik:
            result=run_inspection(env,args.output,record=args.record,seed=args.seed,jitter=args.jitter,
                                  yaw_jitter=args.yaw_jitter,grip_opening=args.grip_opening,motion_scale=args.motion_scale,
                                  read_lateral=args.read_lateral,read_distance=args.read_distance,
                                  grasp_roll=np.deg2rad(args.grasp_roll_deg),upright_gain=args.upright_gain,
                                  record_training=args.record_training,training_image_size=args.training_image_size)
            if not result["success"]:raise SystemExit(1)
            return
        env.reset_inspection()
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        cylinder=env.data.xpos[env.cylinder_body].copy()
        camera_id=env.model.camera("face_view").id
        eye=env.data.cam_xpos[camera_id].copy()
        points={"approach":cylinder+[0,-.10,GRIP_HEIGHT],"grasp":cylinder+[0,0,GRIP_HEIGHT],
                "lift":cylinder+[0,0,GRIP_HEIGHT+.04],
                "retract":cylinder+[0,-.15,GRIP_HEIGHT+.12],
                "read":eye+[.14,READ_DISTANCE,GRIP_HEIGHT-(SCALE_ZERO+SCALE_SPAN/2)]}
        initial=env.data.qpos[env.arm_addresses].copy()
        report={}
        for name,point in points.items():
            options=[env.inverse_kinematics(point,rotation,initial=seed) for seed in (initial,np.array(CARRY_JOINTS))]
            q,error=min(options,key=lambda item:item[1])
            report[name]={"point":point.tolist(),"joints":q.tolist(),"position_error_m":error};initial=q
        (args.output/"ik.json").write_text(json.dumps(report,indent=2))
        imageio.imwrite(args.output/"initial.png",env.render("cylinder_overview"))
        print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=="__main__":main()
