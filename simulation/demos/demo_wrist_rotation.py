"""Exercise right-arm position actuators from the MuJoCo pour starting pose."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.world_model.paths import ROOT
from simulation.world_model.collision import workspace_contacts


def main():
    generated = ROOT / "generated"
    model = mujoco.MjModel.from_xml_path(str(generated / "pour_start.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("pour_start").id)
    mujoco.mj_forward(model, data)
    wrist = model.joint("right_arm_joint6")
    servo = model.actuator("right_arm_joint6_servo").id
    address = wrist.qposadr[0]
    start_angle = data.ctrl[servo]
    timestep = model.opt.timestep
    frame_stride = max(1, round(0.08 / timestep))
    frames = []
    peak_torque = 0.

    with mujoco.Renderer(model, width=720, height=480) as renderer:
        for delta_deg in (0, 15, -15, 0):
            origin = data.ctrl[servo]
            target = start_angle + np.deg2rad(delta_deg)
            if not model.actuator_ctrlrange[servo, 0] <= target <= model.actuator_ctrlrange[servo, 1]:
                raise ValueError("Requested wrist rotation exceeds its joint limits")
            for step in range(round(2.0 / timestep)):
                phase = min(1., (step + 1) * timestep / 1.2)
                data.ctrl[servo] = origin + (target - origin) * phase**2 * (3 - 2 * phase)
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if workspace_contacts(model, data):
                    raise RuntimeError(f"Robot or bottle contacted the workspace at {data.time:.2f} s")
                peak_torque = max(peak_torque, abs(data.actuator_force[servo]))
                if step % frame_stride == 0:
                    renderer.update_scene(data, camera="pour_start_detail")
                    frame = Image.fromarray(renderer.render())
                    actual = np.rad2deg(data.qpos[address] - start_angle)
                    command = np.rad2deg(data.ctrl[servo] - start_angle)
                    overlay = ImageDraw.Draw(frame)
                    overlay.rectangle((8, 8, 302, 47), fill="#17242b")
                    overlay.text((16, 14), "Right wrist roll relative to starting pose", fill="white")
                    overlay.text((16, 30), f"Command {command:+.1f} deg   Measured {actual:+.1f} deg",
                                 fill="white")
                    frames.append(frame)
            actual = np.rad2deg(data.qpos[address] - start_angle)
            print(f"Target {delta_deg:+.0f} deg, reached {actual:+.2f} deg")
            if not np.isfinite(actual) or abs(actual - delta_deg) > 1.:
                raise RuntimeError("Wrist missed the target by more than 1 degree")

    if not any(low < high for low, high in frames[0].crop((0, 50, 720, 480)).getextrema()):
        raise RuntimeError("Renderer returned a blank image; use software EGL")
    path = generated / "wrist_rotation.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(frame_stride * timestep * 1000), loop=0)
    print(f"Peak wrist actuator torque: {peak_torque:.3f} N m. Saved {path}")


if __name__ == "__main__":
    main()
