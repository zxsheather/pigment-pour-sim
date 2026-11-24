"""Add experiment-only position controllers to the reconstructed scene."""
from pathlib import Path
import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def prepare(assets, output, only_bottle=None):
    spec = importlib.util.spec_from_file_location("scene_builder", PROJECT/"simulation/build_scene.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    config_path = PROJECT/"simulation/scene_config.json"
    if only_bottle is not None:
        config = json.loads(config_path.read_text())
        config["bottles"] = [b for b in config["bottles"] if b["color"] == only_bottle]
        if not config["bottles"]:
            raise ValueError(f"Bottle not found: {only_bottle}")
        output.mkdir(parents=True, exist_ok=True)
        config_path = output/"scene_config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    source = builder.build(assets.resolve(), output.resolve(), config_path)
    tree = ET.parse(source)
    root = tree.getroot()
    robot = root.find("worldbody/body[@name='robot_planar_root']")
    for side in ["left", "right"]:
        gripper = robot.find(f".//body[@name='{side}_arm_gripper_base_link']")
        ET.SubElement(gripper, "site", name=f"{side}_tool", pos=".132 0 0", size=".006", rgba=".2 .75 .6 0")
    actuator = ET.SubElement(root, "actuator")
    for joint in robot.iter("joint"):
        name = joint.get("name")
        if not name:
            continue
        kp, kv = (20000, 1800) if name in ["base_x", "base_y"] else (2500, 160) if name in ["base_yaw", "lift_joint"] else (500, 35)
        if "gripper" in name:
            kp, kv = 400, 15
        attrs = dict(name="servo_"+name, joint=name, kp=str(kp), kv=str(kv))
        if joint.get("range"):
            attrs.update(ctrlrange=joint.get("range"), ctrllimited="true")
        ET.SubElement(actuator, "position", **attrs)
    tree.write(source, encoding="unicode")
    from decompose_gripper import decompose
    decompose(source, assets.resolve(), source)
    print(f"Prepared experiment controllers: {source}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, default=PROJECT/"simulation/third_party/sdk_robot")
    p.add_argument("--output", type=Path, default=ROOT/"generated")
    p.add_argument("--only-bottle", choices=["blue", "green", "yellow", "red"])
    args = p.parse_args()
    prepare(args.assets, args.output, args.only_bottle)
