"""Validate a free bottle grasp while moving the X1 torso lift."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.world_model.actuators import FINGER_JOINTS
from simulation.benchmarks.scenarios.build_pour_scenes import GRIPPER
from simulation.world_model.paths import ROOT
from simulation.world_model.collision import workspace_contacts


def main():
    output = ROOT / "generated"
    model = mujoco.MjModel.from_xml_path(str(output / "pour_grasp.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp_start").id)
    mujoco.mj_forward(model, data)

    lift = model.actuator("lift_joint_servo").id
    fingers = [model.actuator(f"{name}_servo").id for name in FINGER_JOINTS]
    pads = {model.geom(f"grip_pad_{side}").id for side in ("left", "right")}
    bottle_geom = model.geom("held_bottle_collision").id
    bottle = data.body("held_water_bottle")
    gripper = data.body(GRIPPER)
    initial_lift = data.ctrl[lift]
    initial_bottle_z = bottle.xpos[2]
    loaded_bottle_z = None
    timestep = model.opt.timestep
    frame_stride = max(1, round(.08 / timestep))
    frames = []
    max_slip = 0.
    grip_origin = gripper.xmat.reshape(3, 3).T @ (bottle.xpos - gripper.xpos)

    with mujoco.Renderer(model, width=720, height=480) as renderer:
        for label, lift_delta, finger_target in (
            ("Hold bottle", 0., None),
            ("Raise torso lift", .04, None),
            ("Lower torso lift", 0., None),
            ("Open gripper", 0., (.044, -.044)),
            ("Close empty gripper", 0., (.018, -.018)),
        ):
            lift_origin = data.ctrl[lift]
            target = initial_lift + lift_delta
            if not model.actuator_ctrlrange[lift, 0] <= target <= model.actuator_ctrlrange[lift, 1]:
                raise ValueError("Requested torso movement exceeds its joint limits")
            carrying = finger_target is None
            if not carrying:
                opening = finger_target[0] > data.ctrl[fingers[0]]
                released_from = bottle.xpos[2]
                data.ctrl[fingers] = finger_target
            for step in range(round(1.6 / timestep)):
                phase = min(1., (step + 1) * timestep)
                data.ctrl[lift] = lift_origin + (target - lift_origin) * phase**2 * (3 - 2 * phase)
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                finger_contacts = set()
                if carrying:
                    for a, b in workspace_contacts(model, data):
                        pair = {a, b}
                        if bottle_geom in pair and pair & pads:
                            finger_contacts.update(pair & pads)
                        else:
                            raise RuntimeError(f"Unexpected workspace contact at {data.time:.2f} s")
                    for contact in data.contact:
                        pair = set(contact.geom)
                        if bottle_geom in pair and not pair & pads:
                            raise RuntimeError(f"Bottle contacted another object at {data.time:.2f} s")
                    relative = gripper.xmat.reshape(3, 3).T @ (bottle.xpos - gripper.xpos)
                    max_slip = max(max_slip, np.linalg.norm(relative - grip_origin))
                if step % frame_stride == 0:
                    renderer.update_scene(data, camera="pour_start_detail")
                    frame = Image.fromarray(renderer.render())
                    overlay = ImageDraw.Draw(frame)
                    overlay.rectangle((8, 8, 285, 48), fill="#17242b")
                    lift_mm = (data.joint("lift_joint").qpos[0] - initial_lift) * 1000
                    overlay.text((16, 14), label, fill="white")
                    overlay.text((16, 30), f"Torso height change: {lift_mm:+.1f} mm", fill="white")
                    frames.append(frame)

            lift_error = abs(data.joint("lift_joint").qpos[0] - target)
            if not np.isfinite(lift_error) or lift_error > .002:
                raise RuntimeError("Torso lift missed its target by more than 2 mm")
            if carrying:
                if finger_contacts != pads or max_slip > .005:
                    raise RuntimeError("Bottle failed the two-pad hold or slipped more than 5 mm")
                if loaded_bottle_z is None:
                    loaded_bottle_z = bottle.xpos[2]
                    print(f"Initial loaded settling: {(loaded_bottle_z - initial_bottle_z) * 1000:+.1f} mm")
                rise = bottle.xpos[2] - loaded_bottle_z
                if abs(rise - lift_delta) > .005:
                    raise RuntimeError("The bottle failed to follow the lift")
                print(f"{label}: lift error {lift_error * 1000:.2f} mm; bottle moved {rise * 1000:+.1f} mm")
            else:
                positions = [data.joint(name).qpos[0] for name in FINGER_JOINTS]
                if not np.allclose(positions, data.ctrl[fingers], atol=.001):
                    raise RuntimeError("Fingers missed their targets by more than 1 mm")
                if opening:
                    drop = released_from - bottle.xpos[2]
                    if drop < .07:
                        raise RuntimeError("The bottle did not fall when the gripper opened")
                    print(f"Bottle fell {drop * 1000:.1f} mm after release")
                print(f"{label}: finger travel {np.round(np.asarray(positions) * 1000, 1)} mm")

    if not any(low < high for low, high in frames[0].crop((0, 50, 720, 480)).getextrema()):
        raise RuntimeError("Renderer returned a blank image; use software EGL")
    path = output / "gripper_lift.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(frame_stride * timestep * 1000), loop=0)
    print(f"Peak bottle translation relative to gripper: {max_slip * 1000:.2f} mm. Saved {path}")


if __name__ == "__main__":
    main()
