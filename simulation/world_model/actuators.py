"""Write the reusable MuJoCo actuator configuration for pouring."""

import xml.etree.ElementTree as ET

from simulation.world_model.build_scene import element, fmt


ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))
FINGER_JOINTS = tuple(f"right_arm_gripper_{side}_joint" for side in ("left", "right"))

SERVO_SETTINGS = (
    *zip(ARM_JOINTS, (100, 100, 100, 40, 40, 35), (10, 10, 10, 5, 5, 3)),
    ("lift_joint", 6000, 600),
    *((name, 1400, 14) for name in FINGER_JOINTS),
)


def write_pour_actuators(model, path):
    """Use the robot's joint and actuator force limits for the position servos."""
    fragment = ET.Element("mujocoinclude")
    servos = element(fragment, "actuator")
    for name, kp, kv in SERVO_SETTINGS:
        joint = model.joint(name)
        element(servos, "position", name=f"{name}_servo", joint=name, kp=kp, kv=kv,
                ctrllimited="true", ctrlrange=fmt(joint.range),
                forcelimited="true", forcerange=fmt(model.jnt_actfrcrange[joint.id]))
    ET.indent(fragment)
    ET.ElementTree(fragment).write(path, encoding="unicode")
