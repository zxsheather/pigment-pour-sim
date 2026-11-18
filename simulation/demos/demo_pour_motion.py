"""Validate the pose-to-actuator layer with an attached bottle."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.motion.pour_motion import BottlePose, PourMotion
from simulation.benchmarks.setup_pour_pose import IDLE_ARM_JOINTS
from simulation.world_model.paths import ROOT
from simulation.world_model.collision import workspace_contacts


def main():
    generated = ROOT / "generated"
    model = mujoco.MjModel.from_xml_path(str(generated / "pour_start.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("pour_start").id)
    motion = PourMotion(model)
    timestep = model.opt.timestep
    idle_addresses = [model.joint(name).qposadr[0] for name in IDLE_ARM_JOINTS]
    idle_start = data.qpos[idle_addresses].copy()
    max_idle_drift = 0.

    def check_idle_arm():
        drift = np.max(np.abs(data.qpos[idle_addresses] - idle_start))
        if not np.isfinite(drift) or drift > np.deg2rad(.5):
            raise RuntimeError("Idle left arm drifted by more than 0.5 degrees")
        return drift

    for _ in range(round(1. / timestep)):
        mujoco.mj_step(model, data)
        max_idle_drift = max(max_idle_drift, check_idle_arm())
    mujoco.mj_forward(model, data)

    start = motion.observe(data)
    cylinder = data.site("cylinder_mouth").xpos.copy()
    upright = start.bottle_quat_wxyz
    clear_height = cylinder[2] + .195
    high_start = BottlePose((*start.mouth_xyz_m[:2], clear_height), upright)
    over_rim = BottlePose((*cylinder[:2], clear_height), upright)
    turn = np.empty(4)
    tilted = np.empty(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([-1., 0., 0.]), np.deg2rad(35))
    mujoco.mju_mulQuat(tilted, turn, np.array(upright))
    tip = BottlePose(over_rim.mouth_xyz_m, tuple(tilted))

    # A direct horizontal move at the loaded starting height would hit the rim.
    low_target = BottlePose((*cylinder[:2], start.mouth_xyz_m[2]), upright)
    try:
        motion.plan(data, low_target, 1.5)
    except ValueError as error:
        if "contacts the workspace" not in str(error):
            raise
        print("Verified: rejected a direct move through the cylinder")
    else:
        raise RuntimeError("Motion planner accepted a path through the cylinder")

    frames = []
    frame_stride = max(1, round(.08 / timestep))
    with mujoco.Renderer(model, width=720, height=480) as renderer:
        for label, target in (
            ("Lift clear of the rim", high_start),
            ("Move above the cylinder", over_rim),
            ("Tilt 35 degrees", tip),
            ("Return upright", over_rim),
            ("Retract", high_start),
            ("Lower to starting pose", start),
        ):
            duration = 1.6
            references = motion.plan(data, target, duration)
            for step in range(round((duration + .8) / timestep)):
                motion.command(data, references, (step + 1) * timestep / duration)
                mujoco.mj_step(model, data)
                max_idle_drift = max(max_idle_drift, check_idle_arm())
                mujoco.mj_forward(model, data)
                if workspace_contacts(model, data):
                    raise RuntimeError(f"Unexpected workspace contact during: {label}")
                if step % frame_stride == 0:
                    renderer.update_scene(data, camera="pour_start_detail")
                    frame = Image.fromarray(renderer.render())
                    overlay = ImageDraw.Draw(frame)
                    overlay.rectangle((8, 8, 290, 48), fill="#17242b")
                    overlay.text((16, 14), label, fill="white")
                    height = (data.site(motion.mouth).xpos[2] - cylinder[2]) * 100
                    overlay.text((16, 30), f"Bottle mouth above rim: {height:.1f} cm", fill="white")
                    frames.append(frame)
            actual = motion.observe(data)
            position_error = np.linalg.norm(np.subtract(actual.mouth_xyz_m, target.mouth_xyz_m))
            rotation_error = np.empty(3)
            mujoco.mju_subQuat(rotation_error, np.array(target.bottle_quat_wxyz),
                               np.array(actual.bottle_quat_wxyz))
            angle_error = np.rad2deg(np.linalg.norm(rotation_error))
            print(f"{label}: error {position_error * 1000:.1f} mm / {angle_error:.2f} deg")
            if (not np.isfinite(position_error + angle_error)
                    or position_error > .005 or angle_error > 1.):
                raise RuntimeError(f"Actuator tracking missed the pose target during: {label}")

    if not any(low < high for low, high in frames[0].crop((0, 50, 720, 480)).getextrema()):
        raise RuntimeError("Renderer returned a blank image; use software EGL")
    path = generated / "pour_motion.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(frame_stride * timestep * 1000), loop=0)
    print(f"Maximum idle left-arm joint drift: {np.rad2deg(max_idle_drift):.3f} deg")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
