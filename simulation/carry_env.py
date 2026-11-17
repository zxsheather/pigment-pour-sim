"""Contact-based bottle transport on the reconstructed planar robot model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")
CARRY_JOINTS = (0.0, 1.0, -2.2, 1.2, 0.0, 0.0)
GRASP_OFFSET = np.array([0.135, 0.0, 0.0])
PHYSICS_DT = 0.002
CONTROL_DT = 0.05


def angle_difference(target: float, current: float) -> float:
    return float((target - current + np.pi) % (2 * np.pi) - np.pi)


def carry_instruction(color: str, goal: str) -> str:
    destination = ("the tray beside the measuring cylinder" if goal == "B"
                   else "the tray with the other colored bottles")
    return (f"While holding the {color} bottle upright, drive around the outside of the table "
            f"and stop beside {destination}. Keep the bottle in the gripper.")


def preserve_official_finger_geometry(scene, scene_path: Path):
    """Import each convex solid in the official STL as one contact geom.

    The pinned collision STLs contain multiple disconnected convex solids.
    Loading each whole STL as one MuJoCo mesh would fill empty space between
    those solids with a single convex hull. Splitting connected components
    preserves the supplied union without designing a new pad or moving vertices.
    """
    import trimesh
    assets=scene.find("asset")
    cache=ROOT/"generated/official_finger_collision"
    cache.mkdir(parents=True,exist_ok=True)
    converted={}
    for side in ("left","right"):
        for finger in ("left","right"):
            body=scene.find(f".//body[@name='{side}_arm_gripper_{finger}_link']")
            for index,geom in enumerate(list(body.findall("geom"))):
                if geom.get("contype")=="0":continue
                name=geom.get("mesh")
                if not name:raise ValueError("Official finger collision must reference a mesh")
                if name not in converted:
                    asset=assets.find(f"mesh[@name='{name}']")
                    path=Path(asset.get("file"))
                    if not path.is_absolute():path=scene_path.parent/path
                    digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    mesh=trimesh.load_mesh(path)
                    parts=mesh.split(only_watertight=False)
                    if not parts or not all(part.is_convex for part in parts):
                        raise ValueError("Official finger STL has a nonconvex solid; do not silently replace its shape")
                    converted[name]=[]
                    for component,part in enumerate(parts):
                        target=cache/f"{digest[:16]}_{component}.stl"
                        if not target.exists():
                            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                                stage=Path(temporary)/"part.stl"
                                part.export(stage);stage.replace(target)
                        component_name=f"{name}_solid_{component}"
                        attrs=dict(asset.attrib,name=component_name,file=str(target.resolve()))
                        ET.SubElement(assets,"mesh",attrs)
                        converted[name].append(component_name)
                body.remove(geom)
                for component,component_name in enumerate(converted[name]):
                    attrs=dict(geom.attrib,mesh=component_name,
                               name=f"{side}_{finger}_official_collision_{index}_{component}")
                    ET.SubElement(body,"geom",attrs)


def build_control_model(scene_path: Path, config: dict, *, scene_transform=None,
                        finger_collisions="flat-proxy") -> mujoco.MjModel:
    """Add servos and simple finger collision pads to the static scene."""
    if finger_collisions not in ("flat-proxy", "official"):
        raise ValueError("Unknown finger collision model")
    tree = ET.parse(scene_path)
    scene = tree.getroot()
    if finger_collisions=="official":preserve_official_finger_geometry(scene,scene_path)
    option = scene.find("option")
    option.set("timestep", str(PHYSICS_DT))
    option.set("impratio", "5")
    option.set("noslip_iterations", "3")
    # Heading is continuous during navigation; the display model's +/- pi
    # slider limits would otherwise block valid turns across that boundary.
    scene.find(".//joint[@name='base_yaw']").set("limited", "false")
    # Planar joints support the base. Ground friction from the imported wheel
    # meshes would add an incompatible second support model; bottles still
    # collide with the floor, while the robot collides with tables and trays.
    ground = scene.find(".//geom[@name='ground']")
    ground.set("contype", "4")
    ground.set("conaffinity", "2")
    # This camera includes the entire transport route; the static overview
    # intentionally concentrates on tabletop geometry instead.
    from build_scene import camera
    camera(scene.find("worldbody"), "carry_overview", [2.2, -2.7, 2.6], [0.0, 0.2, 0.8], 45)
    actuators = ET.SubElement(scene, "actuator")
    for name, gain, force in (("base_x", 1000, 150), ("base_y", 1000, 150),
                              ("base_yaw", 180, 100)):
        ET.SubElement(actuators, "velocity", name=name + "_servo", joint=name,
                      kv=str(gain), forcelimited="true", forcerange=f"-{force} {force}")
    for joint in scene.findall(".//worldbody//joint"):
        name = joint.get("name", "")
        if name.startswith("base_"):
            continue
        if "wheel_joint" in name:
            ET.SubElement(actuators, "velocity", name=name + "_servo", joint=name,
                          kv="2", forcelimited="true", forcerange="-10 10")
            continue
        finger = name.endswith(("gripper_left_joint", "gripper_right_joint"))
        gain, damping, force = (2000, 30, 20) if finger else (800, 60, 100)
        ET.SubElement(actuators, "position", name=name + "_servo", joint=name,
                      kp=str(gain), kv=str(damping), forcelimited="true",
                      forcerange=f"-{force} {force}")
    # The CAD finger meshes are single convex collision hulls. Use localized
    # contact pads so the open gripper has a physical gap around the bottle.
    for side in ("left", "right"):
        for finger, offset in (("left", 0.018), ("right", -0.018)):
            body = scene.find(f".//body[@name='{side}_arm_gripper_{finger}_link']")
            if finger_collisions == "official":
                # Preserve the collision geoms imported from the pinned URDF.
                # No custom contact pad is inserted into this baseline.
                continue
            for geom in body.findall("geom"):
                if geom.get("contype") != "0":
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
            # Separate upper/lower pads provide a distributed grasp wrench.
            for level, height in enumerate((-0.024, 0.024)):
                ET.SubElement(body, "geom", name=f"{side}_{finger}_contact_pad_{level}",
                              type="box", pos=f"0.055 0 {offset + height}",
                              size="0.025 0.005 0.012", mass="0", rgba="0.18 0.18 0.18 1",
                              contype="1", conaffinity="2", condim="4",
                              friction="1.2 0.01 0.001", solref="0.006 1",
                              solimp="0.98 0.999 0.001")
    # Treat the initial water volume as rigid payload mass for transport.
    for geom in scene.findall(".//geom"):
        if geom.get("name", "").startswith("bottle_") and geom.get("name", "").endswith("_shell"):
            geom.set("mass", str(0.11 + config["bottle"]["initial_ml"] * 0.001))
    if scene_transform is not None:
        scene_transform(scene, config)
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))


class CarryEnv:
    """Move a physically held bottle using body-forward and yaw velocities.

    The base uses planar velocity servos. Wheel traction, sloshing and a grasp
    acquisition policy are outside this environment's current scope.
    """

    def __init__(self, scene_path: Path | None = None, *, render: bool = False,
                 image_size: int = 256, max_seconds: float = 75.0, model_transform=None,
                 finger_collisions="flat-proxy"):
        self.config = json.loads((ROOT / "scene_config.json").read_text())
        self.model = build_control_model(scene_path or ROOT / "generated/scene.xml", self.config,
                                         scene_transform=model_transform,finger_collisions=finger_collisions)
        self.data = mujoco.MjData(self.model)
        self.render_enabled = render
        self.image_size = image_size
        self.max_seconds = max_seconds
        self.renderer = None
        self.robot_root = self.model.body("robot_planar_root").id
        self.gripper_body = self.model.body("right_arm_gripper_base_link").id
        self.finger_bodies = {
            self.model.body(f"right_arm_gripper_{finger}_link").id for finger in ("left", "right")
        }
        self.robot_bodies = set()
        for index in range(1, self.model.nbody):
            parent = index
            while parent and parent != self.robot_root:
                parent = int(self.model.body_parentid[parent])
            if parent == self.robot_root:
                self.robot_bodies.add(index)
        self.position_actuators = {}
        for index in range(self.model.nu):
            joint = int(self.model.actuator_trnid[index, 0])
            name = self.model.joint(joint).name
            if not name.startswith("base_") and "wheel_joint" not in name:
                self.position_actuators[index] = int(self.model.jnt_qposadr[joint])
        self.applied_action = np.zeros(2)
        self.done = False

    def _set_joint(self, name: str, value: float):
        self.data.qpos[self.model.joint(name).qposadr[0]] = value

    def _joint_value(self, name: str) -> float:
        return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def base_pose(self) -> np.ndarray:
        return np.array([self._joint_value(name) for name in ("base_x", "base_y", "base_yaw")])

    def base_velocity(self) -> np.ndarray:
        velocity = np.array([self.data.qvel[self.model.joint(name).dofadr[0]]
                             for name in ("base_x", "base_y", "base_yaw")])
        yaw = self.base_pose()[2]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        return np.array([cosine * velocity[0] + sine * velocity[1],
                         -sine * velocity[0] + cosine * velocity[1], velocity[2]])

    def _arm_state(self, side: str) -> np.ndarray:
        body = self.model.body(f"{side}_arm_gripper_base_link").id
        yaw = self.base_pose()[2]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        base_rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        position = base_rotation.T @ (self.data.xpos[body] - self.data.xpos[self.robot_root])
        rotation = base_rotation.T @ self.data.xmat[body].reshape(3, 3)
        opening = (self._joint_value(f"{side}_arm_gripper_left_joint")
                   - self._joint_value(f"{side}_arm_gripper_right_joint")) / 0.0975
        # Wall-X uses the first two ROWS for its 6D rotation representation.
        return np.concatenate([position, rotation[:2].reshape(6), [opening]])

    def state_vector(self) -> np.ndarray:
        return np.concatenate([self._arm_state("left"), self._arm_state("right"),
                               self.base_velocity()]).astype(np.float32)

    def action_vector(self) -> np.ndarray:
        """Absolute held-arm targets and the actual commanded body velocity."""
        return np.concatenate([self.arm_targets, [self.applied_action[0], 0.0,
                                                  self.applied_action[1]]]).astype(np.float32)

    def _apply_servos(self, forward: float, yaw_velocity: float):
        yaw = self.base_pose()[2]
        for name, value in (("base_x", forward * np.cos(yaw)),
                            ("base_y", forward * np.sin(yaw)), ("base_yaw", yaw_velocity)):
            self.data.ctrl[self.model.actuator(name + "_servo").id] = value
        # Wheels spin consistently with the commanded planar-base motion.
        for side, sign in (("left", -1), ("right", 1)):
            self.data.ctrl[self.model.actuator(f"{side}_wheel_joint_servo").id] = (
                forward + sign * 0.255 * yaw_velocity) / 0.095
        for actuator, value in self.hold_targets.items():
            self.data.ctrl[actuator] = value

    def reset(self, *, color: str = "blue", seed: int = 0,
              start_jitter: float = 0.0, goal: str = "B", heading_jitter: float = 0.0) -> dict:
        if not np.isfinite([start_jitter, heading_jitter]).all() or min(start_jitter, heading_jitter) < 0:
            raise ValueError("Reset jitter ranges must be finite and nonnegative")
        if color not in {bottle["color"] for bottle in self.config["bottles"]}:
            raise ValueError(f"Unknown bottle color: {color}")
        if goal not in ("A", "B"):
            raise ValueError("goal must be A or B")
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        robot = self.config["robot"]
        source = np.array([robot["base_x"], robot["source_base_y"], robot["source_base_yaw"]])
        target = np.array([robot["target_base_x"], robot["target_base_y"], robot["target_base_yaw"]])
        start, self.goal_pose = (source, target) if goal == "B" else (target, source)
        start = start.copy()
        start[:2] += rng.uniform(-start_jitter, start_jitter, size=2)
        start[2] += rng.uniform(-heading_jitter, heading_jitter)
        for name, value in zip(("base_x", "base_y", "base_yaw"), start):
            self._set_joint(name, value)
        self._set_joint("lift_joint", 0.30)
        self._set_joint("head_pitch_joint", 0.25)
        for side in ("left", "right"):
            for index, value in enumerate(CARRY_JOINTS, start=1):
                self._set_joint(f"{side}_arm_joint{index}", value)
            self._set_joint(f"{side}_arm_gripper_left_joint", 0.034)
            self._set_joint(f"{side}_arm_gripper_right_joint", -0.034)
        self.hold_targets = {actuator: float(self.data.qpos[address])
                             for actuator, address in self.position_actuators.items()}
        for finger, value in (("left", 0.025), ("right", -0.025)):
            actuator = self.model.actuator(f"right_arm_gripper_{finger}_joint_servo").id
            self.hold_targets[actuator] = value
        mujoco.mj_forward(self.model, self.data)
        self.arm_targets = np.concatenate([self._arm_state("left"), self._arm_state("right")])
        self.arm_targets[19] = 0.05 / 0.0975
        self.bottle_body = self.model.body(f"bottle_{color}").id
        self.color = color
        self.goal_name = goal
        bottle_joint = self.model.joint(f"bottle_{color}_free")
        address = int(bottle_joint.qposadr[0])
        self.data.qpos[address:address + 3] = (
            self.data.xpos[self.gripper_body]
            + self.data.xmat[self.gripper_body].reshape(3, 3) @ GRASP_OFFSET)
        self.data.qpos[address + 3:address + 7] = self.data.xquat[self.gripper_body]
        self.applied_action[:] = 0
        self.done = False
        for _ in range(round(0.6 / PHYSICS_DT)):
            self._apply_servos(0.0, 0.0)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.grasp_reference = self._bottle_relative_position()
        self.stable_seconds = 0.0
        self.contact_lost_seconds = 0.0
        self.data.time = 0.0
        metrics = self.metrics()
        if metrics["grip_contacts"] != 2 or metrics["bottle_tilt_rad"] > 0.35:
            raise RuntimeError(f"Reset did not establish a stable physical grasp: {metrics}")
        return self.observe()

    def _bottle_relative_position(self) -> np.ndarray:
        return self.data.xmat[self.gripper_body].reshape(3, 3).T @ (
            self.data.xpos[self.bottle_body] - self.data.xpos[self.gripper_body])

    def metrics(self) -> dict:
        touching_fingers = set()
        forbidden_contacts = []
        for contact in self.data.contact[:self.data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            first_body = int(self.model.geom_bodyid[first])
            second_body = int(self.model.geom_bodyid[second])
            if first_body == self.bottle_body and second_body in self.finger_bodies:
                touching_fingers.add(second_body)
            if second_body == self.bottle_body and first_body in self.finger_bodies:
                touching_fingers.add(first_body)
            if self.bottle_body in (first_body, second_body):
                other_body = second_body if first_body == self.bottle_body else first_body
                if other_body not in self.robot_bodies and contact.dist < -0.001:
                    forbidden_contacts.append([self.model.geom(first).name,
                                               self.model.geom(second).name])
            for robot_geom, environment_geom, robot_body, environment_body in (
                (first, second, first_body, second_body), (second, first, second_body, first_body)
            ):
                if robot_body not in self.robot_bodies or environment_body in self.robot_bodies:
                    continue
                if environment_body == self.bottle_body:
                    continue
                if self.model.geom(environment_geom).name == "ground":
                    continue
                if contact.dist < -0.001:
                    forbidden_contacts.append([self.model.geom(robot_geom).name,
                                               self.model.geom(environment_geom).name])
        bottle_up = self.data.xmat[self.bottle_body].reshape(3, 3)[2, 2]
        position_error = np.linalg.norm(self.base_pose()[:2] - self.goal_pose[:2])
        return {
            "position_error_m": float(position_error),
            "heading_error_rad": abs(angle_difference(self.goal_pose[2], self.base_pose()[2])),
            "base_speed_m_s": float(np.linalg.norm(self.base_velocity()[:2])),
            "base_yaw_speed_rad_s": abs(float(self.base_velocity()[2])),
            "bottle_tilt_rad": float(np.arccos(np.clip(bottle_up, -1.0, 1.0))),
            "bottle_slip_m": float(np.linalg.norm(self._bottle_relative_position() - self.grasp_reference)),
            "bottle_height_m": float(self.data.xpos[self.bottle_body, 2]),
            "grip_contacts": len(touching_fingers),
            "forbidden_contacts": forbidden_contacts,
        }

    def step(self, action) -> tuple[dict, dict]:
        if self.done:
            raise RuntimeError("Episode has ended. Call reset before taking more actions.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("action must contain finite [forward_m_s, yaw_rad_s]")
        target = np.clip(action, [-0.18, -0.6], [0.18, 0.6])
        self.applied_action += np.clip(target - self.applied_action,
                                      -np.array([0.3, 1.2]) * CONTROL_DT,
                                      np.array([0.3, 1.2]) * CONTROL_DT)
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            self._apply_servos(*self.applied_action)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        metrics = self.metrics()
        self.contact_lost_seconds = (self.contact_lost_seconds + CONTROL_DT
                                     if metrics["grip_contacts"] < 2 else 0.0)
        stopped_at_goal = (metrics["position_error_m"] < 0.04
                           and metrics["heading_error_rad"] < 0.06
                           and metrics["base_speed_m_s"] < 0.015
                           and metrics["base_yaw_speed_rad_s"] < 0.03)
        self.stable_seconds = self.stable_seconds + CONTROL_DT if stopped_at_goal else 0.0
        reason = None
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            reason = "nonfinite_state"
        elif metrics["forbidden_contacts"]:
            reason = "collision"
        elif metrics["bottle_slip_m"] > 0.04 or self.contact_lost_seconds > 0.15:
            reason = "lost_grasp"
        elif metrics["bottle_tilt_rad"] > 0.45:
            reason = "bottle_tilt"
        elif self.data.time >= self.max_seconds:
            reason = "timeout"
        success = reason is None and self.stable_seconds >= 0.5
        self.done = success or reason is not None
        info = dict(metrics, success=success, done=self.done, failure_reason=reason,
                    simulated_seconds=float(self.data.time))
        return self.observe(), info

    def render(self, camera: str = "overview", *, width: int | None = None) -> np.ndarray:
        size = width or self.image_size
        if self.renderer is None or self.renderer.width != size:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observe(self) -> dict:
        observation = {"state": self.state_vector(), "base_pose": self.base_pose(),
                       "bottle_position": self.data.xpos[self.bottle_body].copy()}
        if self.render_enabled:
            observation["images"] = {camera: self.render(camera) for camera in CAMERAS}
        return observation

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class WaypointTeacher:
    """Waypoint teacher with smooth steering or the original stop-turn profile."""

    def __init__(self, env: CarryEnv, *, profile: str = "smooth"):
        if profile not in ("smooth", "stop-turn"):
            raise ValueError("Unknown teacher profile")
        self.profile = profile
        points = np.array(env.config["navigation"]["route_xy"], dtype=np.float64)
        self.points = points[1:] if env.goal_name == "B" else points[::-1][1:]
        self.index = 0
        self.goal_yaw = float(env.goal_pose[2])
        self.direction = None

    def action(self, observation: dict) -> np.ndarray:
        if self.profile == "smooth":
            return self._smooth_action(observation)
        x, y, yaw = observation["base_pose"]
        while self.index < len(self.points):
            delta = self.points[self.index] - [x, y]
            distance = float(np.linalg.norm(delta))
            if distance < 0.025:
                self.index += 1
                continue
            heading = float(np.arctan2(delta[1], delta[0]))
            error = angle_difference(heading, yaw)
            direction = 1.0
            if abs(error) > np.pi / 2:
                error = angle_difference(heading + np.pi, yaw)
                direction = -1.0
            forward = direction * min(0.16, distance * 1.2) if abs(error) < 0.18 else 0.0
            return np.array([forward, np.clip(2.0 * error, -0.5, 0.5)])
        error = angle_difference(self.goal_yaw, yaw)
        return np.array([0.0, np.clip(2.0 * error, -0.5, 0.5)])

    def _smooth_action(self, observation: dict) -> np.ndarray:
        x, y, yaw = observation["base_pose"]
        while self.index < len(self.points):
            delta = self.points[self.index] - [x, y]
            distance = float(np.linalg.norm(delta))
            if distance < 0.012:
                self.index += 1
                self.direction = None
                continue
            heading = float(np.arctan2(delta[1], delta[0]))
            error = angle_difference(heading, yaw)
            # Hold the choice for a whole segment so small heading changes
            # around 90 degrees cannot repeatedly switch forward/reverse.
            if self.direction is None:
                self.direction = -1.0 if abs(error) > np.pi / 2 else 1.0
            if self.direction < 0:
                error = angle_difference(heading + np.pi, yaw)
            alignment = max(0.0, float(np.cos(error))) ** 4
            forward = self.direction * min(0.18, 2.2 * distance) * alignment
            return np.array([forward, np.clip(3.0 * error, -0.55, 0.55)])
        error = angle_difference(self.goal_yaw, yaw)
        return np.array([0.0, np.clip(3.0 * error, -0.55, 0.55)])
