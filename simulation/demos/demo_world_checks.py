"""Render scene reset and individual servo checks without an interactive viewer."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.world_model.actuators import ARM_JOINTS, FINGER_JOINTS
from simulation.world_model.collision import workspace_contacts
from simulation.world_model.paths import ROOT


def reset(model, data, key):
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    mujoco.mj_forward(model, data)


def capture(renderer, data, title, detail):
    renderer.update_scene(data, camera="pour_start_detail")
    frame = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 38), fill="#17242b")
    draw.text((8, 4), title, fill="white")
    draw.text((8, 21), detail, fill="white")
    return frame


def save(frames, path, duration):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"Saved {path}")


def show_resets(output):
    frames = []
    for file, key, label in (
        ("pour_start.xml", "pour_start", "Attached bottle"),
        ("pour_grasp.xml", "grasp_start", "Free bottle held by pads"),
    ):
        model = mujoco.MjModel.from_xml_path(str(output / file))
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, width=480, height=320) as renderer:
            reset(model, data, key)
            frames.append(capture(renderer, data, label, "Initial keyframe"))
            if key == "pour_start":
                servo = model.actuator("right_arm_joint2_servo").id
                mujoco.mj_step(model, data, nstep=round(1 / model.opt.timestep))
                data.ctrl[servo] += .04
                for _ in range(round(1 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    if workspace_contacts(model, data):
                        break
                mujoco.mj_forward(model, data)
                contacts = len(workspace_contacts(model, data))
                frames.append(capture(renderer, data, label, f"Arm step: {contacts} workspace contacts"))
            else:
                mujoco.mj_step(model, data, nstep=round(.5 / model.opt.timestep))
                mujoco.mj_forward(model, data)
                frames.append(capture(renderer, data, label, "After settling under gravity"))
            reset(model, data, key)
            frames.append(capture(renderer, data, label, "Reset: original keyframe restored"))
    save(frames, output / "world_scene_reset.gif", 1400)


def show_actuators(output):
    model = mujoco.MjModel.from_xml_path(str(output / "pour_start.xml"))
    data = mujoco.MjData(model)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    dt = model.opt.timestep
    frames = []
    with mujoco.Renderer(model, width=480, height=320) as renderer:
        for name in (*ARM_JOINTS, "lift_joint", *FINGER_JOINTS):
            reset(model, data, "pour_start")
            joint = model.joint(name)
            servo = model.actuator(f"{name}_servo")
            start = data.ctrl[servo.id]
            mujoco.mj_step(model, data, nstep=round(1 / dt))
            baseline = data.joint(name).qpos[0]
            linear = joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE
            delta, scale, unit = (.004, 1000, "mm") if linear else (.015, 180 / np.pi, "deg")
            targets = [start]
            for sign in (-1, 1):
                target = start + sign * delta
                if servo.ctrlrange[0] <= target <= servo.ctrlrange[1]:
                    targets.extend((target, start))
            for target in targets:
                data.ctrl[servo.id] = target
                for _ in range(3):
                    mujoco.mj_step(model, data, nstep=round(1 / (3 * dt)))
                    mujoco.mj_forward(model, data)
                    measured = data.joint(name).qpos[0] - baseline
                    detail = f"Target {(target-start)*scale:+.2f} {unit}   Measured {measured*scale:+.2f} {unit}"
                    frames.append(capture(renderer, data, f"{name} | contacts disabled", detail))
    save(frames, output / "world_actuators.gif", 220)


def main():
    output = ROOT / "generated"
    show_resets(output)
    show_actuators(output)


if __name__ == "__main__":
    main()
