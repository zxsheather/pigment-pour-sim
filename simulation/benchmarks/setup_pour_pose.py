"""Generate a static MuJoCo pour start with a bottle fixed to the right gripper."""

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from simulation.world_model.collision import workspace_contacts
from simulation.world_model.build_scene import camera, element, fmt
from simulation.world_model.paths import ROOT, SCENE_CONFIG
from simulation.benchmarks.pose import set_display_pose
from simulation.world_model.actuators import ARM_JOINTS, FINGER_JOINTS, SERVO_SETTINGS, write_pour_actuators


GRIPPER = "right_arm_gripper_base_link"
IDLE_ARM_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))
BOTTLE_CENTER = np.array([0.142, 0., 0.])
MOUTH_OFFSET = np.array([0., -0.05, 0.06])
GRIPPER_ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def apply_start_pose(model, data, config):
    set_display_pose(model, data, config)
    robot = config["robot"]
    for name, value in {
        "base_x": robot["target_base_x"],
        "base_y": robot["target_base_y"],
        "base_yaw": robot["target_base_yaw"],
        "right_arm_gripper_left_joint": config["bottle"]["radius"],
        "right_arm_gripper_right_joint": -config["bottle"]["radius"],
    }.items():
        data.joint(name).qpos[0] = value
    for name in IDLE_ARM_JOINTS:
        data.joint(name).qpos[0] = 0.
    mujoco.mj_forward(model, data)
    target = data.site("cylinder_mouth").xpos.copy() + MOUTH_OFFSET
    gripper_id = model.body(GRIPPER).id
    mouth_id = model.site("held_bottle_mouth").id
    names = ["lift_joint", *ARM_JOINTS]
    joints = [model.joint(name) for name in names]
    qpos = [joint.qposadr[0] for joint in joints]
    dofs = [joint.dofadr[0] for joint in joints]
    lower, upper = np.asarray([joint.range for joint in joints]).T
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros_like(jac_pos)
    desired_quat = np.empty(4)
    mujoco.mju_mat2Quat(desired_quat, GRIPPER_ROTATION.ravel())
    current_quat, rotation_error = np.empty(4), np.empty(3)
    rng = np.random.default_rng(42)
    default = data.qpos[qpos].copy()
    for _ in range(32):
        for _ in range(170):
            mujoco.mj_forward(model, data)
            current = data.body(gripper_id).xmat.reshape(3, 3)
            mujoco.mju_mat2Quat(current_quat, current.ravel())
            mujoco.mju_subQuat(rotation_error, desired_quat, current_quat)
            position_error = target - data.site(mouth_id).xpos
            if np.linalg.norm(position_error) < 0.0005 and np.linalg.norm(rotation_error) < 0.002:
                if not workspace_contacts(model, data):
                    return
                break
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, mouth_id)
            jac = np.vstack((jac_pos[:, dofs], 0.2 * jac_rot[:, dofs]))
            error = np.r_[position_error, 0.2 * (current @ rotation_error)]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002**2 * np.eye(6), error)
            data.qpos[qpos] = np.clip(data.qpos[qpos] + np.clip(delta, -0.12, 0.12), lower, upper)
        data.qpos[qpos] = np.clip(default + rng.normal(0., 0.55, len(joints)), lower, upper)
    raise RuntimeError("No contact-free starting pose found at the configured dock")


def add_water_bottle(scene, config):
    gripper = scene.find(f".//body[@name='{GRIPPER}']")
    spec = config["bottle"]
    radius, height = spec["radius"], spec["height"]
    inner_radius = radius - 0.003
    volume = spec["initial_ml"] * 1e-6
    water_height = volume / (np.pi * inner_radius**2)
    bottle = element(gripper, "body", name="held_water_bottle", pos=fmt(BOTTLE_CENTER))
    glass = ".69 .85 .89 .25"
    bottom = -height / 2
    element(bottle, "geom", name="held_bottle_base", type="cylinder", pos=fmt([0, 0, bottom + .002]),
            size=fmt([radius, .002]), mass=".015", rgba=glass, contype=4, conaffinity=2)
    for i in range(32):
        angle = i * 2 * np.pi / 32
        center = radius - .001
        element(bottle, "geom", name=f"held_bottle_wall_{i}", type="box",
                pos=fmt([center * np.cos(angle), center * np.sin(angle), 0.]),
                size=fmt([.001, radius * np.tan(np.pi / 32), height / 2]),
                euler=fmt([0, 0, angle]), mass=fmt([.10 / 32]), rgba=glass,
                contype=4, conaffinity=2)
    element(bottle, "geom", name="held_water", type="cylinder", rgba=".21 .60 .92 .78",
            pos=fmt([0, 0, bottom + .004 + water_height / 2]),
            size=fmt([inner_radius, water_height / 2]), mass=fmt([1000 * volume]),
            contype=0, conaffinity=0)
    element(bottle, "site", name="held_bottle_mouth", pos=fmt([0, 0, height / 2]),
            size=".003", rgba="0 0 0 0")


def park_unused_joints(scene, model, data):
    """Anchor the base and idle arm at the solved start pose."""
    anchors = element(scene, "equality")
    for name in ("base_x", "base_y", "base_yaw", *IDLE_ARM_JOINTS):
        address = model.joint(name).qposadr[0]
        offset = data.qpos[address] - model.qpos0[address]
        element(anchors, "joint", name=f"park_{name}", joint1=name,
                polycoef=fmt([offset, 0, 0, 0, 0]), solref=".01 1")


def save_grasp_scene(scene, model, data, config, path):
    """Put the bottle in the existing finger grasp as a free body."""
    # Resolve several contact points along each pad and reduce numerical creep.
    option = scene.find("option")
    option.set("noslip_iterations", "20")
    element(option, "flag", multiccd="enable")
    world = scene.find("worldbody")
    gripper = world.find(f".//body[@name='{GRIPPER}']")
    bottle = gripper.find("body[@name='held_water_bottle']")
    pose = data.body("held_water_bottle")
    gripper.remove(bottle)
    world.append(bottle)
    bottle.set("pos", fmt(pose.xpos))
    bottle.set("quat", fmt(pose.xquat))
    element(bottle, "freejoint", name="held_bottle_free")
    for geom in bottle.findall("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    spec = config["bottle"]
    element(bottle, "geom", name="held_bottle_collision", type="cylinder",
            size=fmt([spec["radius"], spec["height"] / 2]), rgba="0 0 0 0",
            mass="0", contype=4, conaffinity=3, friction="0.9 0.01 0.001")
    for side, sign in (("left", 1), ("right", -1)):
        finger = gripper.find(f"body[@name='right_arm_gripper_{side}_link']")
        # Use a flat pad for bottle contact. Keep the imported mesh for
        # contacts with the workspace, where props still have mask 2.
        finger.find("geom[@contype='1']").set("contype", "16")
        element(finger, "geom", name=f"grip_pad_{side}", type="box",
                pos=fmt([BOTTLE_CENTER[0] - .080074, sign * .003, sign * .018]),
                size=".018 .003 .024", mass="0", rgba=".17 .20 .23 1",
                contype=0, conaffinity=4, condim=4, friction="0.9 0.01 0.001",
                solimp=".97 .995 .001")

    key = scene.find("keyframe/key")
    key.set("name", "grasp_start")
    key.set("qpos", fmt(np.r_[data.qpos, pose.xpos, pose.xquat]))
    targets = data.ctrl.copy()
    # The finger position commands preload the already closed grasp against the bottle.
    for name, direction in zip(FINGER_JOINTS, (1, -1)):
        targets[model.actuator(f"{name}_servo").id] = direction * (spec["radius"] - .005)
    key.set("ctrl", fmt(targets))
    ET.ElementTree(scene).write(path, encoding="unicode")


def build_pour_scenes(generated, config):
    """Generate both pour scenarios without opening a renderer."""
    scene = ET.parse(generated / "scene.xml").getroot()
    add_water_bottle(scene, config)
    world = scene.find("worldbody")
    # The base is supported by planar joints; its imported chassis overlaps the floor.
    floor = world.find("geom[@name='ground']")
    floor.set("contype", "8")
    floor.set("conaffinity", "6")
    cx, cy = config["cylinder"]["center_xy"]
    camera(world, "pour_start_view", [cx + 1.1, cy - .95, 1.55], [cx - .10, cy - .34, .80])
    camera(world, "pour_start_detail", [cx + .57, cy + .46, 1.40], [cx, cy - .06, 1.02], 35)
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    data = mujoco.MjData(model)
    apply_start_pose(model, data, config)
    park_unused_joints(scene, model, data)
    actuator_path = generated / "pour_actuators.xml"
    write_pour_actuators(model, actuator_path)
    element(scene, "include", file=actuator_path.name)
    targets = [data.joint(name).qpos[0] for name, _, _ in SERVO_SETTINGS]
    element(element(scene, "keyframe"), "key", name="pour_start",
            qpos=fmt(data.qpos), ctrl=fmt(targets))
    path = generated / "pour_start.xml"
    ET.ElementTree(scene).write(path, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("pour_start").id)
    mujoco.mj_forward(model, data)
    offset = data.site("held_bottle_mouth").xpos - data.site("cylinder_mouth").xpos
    assert np.linalg.norm(offset - MOUTH_OFFSET) < .0005 and not workspace_contacts(model, data)
    print(f"Saved {path}; bottle mouth offset XYZ (m): {offset.round(4)}")
    print(f"Bottle upright axis: {data.body('held_water_bottle').xmat.reshape(3, 3)[:, 2].round(4)}")
    grasp_path = generated / "pour_grasp.xml"
    save_grasp_scene(scene, model, data, config, grasp_path)
    mujoco.MjModel.from_xml_path(str(grasp_path))
    print(f"Saved {grasp_path} with a free bottle and finger contact")
    return path, grasp_path


def main():
    generated = ROOT / "generated"
    config = json.loads(SCENE_CONFIG.read_text())
    path, _ = build_pour_scenes(generated, config)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("pour_start").id)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, width=1080, height=720) as renderer:
        for view in ("pour_start_view", "pour_start_detail"):
            renderer.update_scene(data, camera=view)
            frame = Image.fromarray(renderer.render())
            if not any(low < high for low, high in frame.getextrema()):
                raise RuntimeError("Renderer returned a blank image; use software EGL")
            frame.save(generated / f"{view}.png")


if __name__ == "__main__":
    main()
