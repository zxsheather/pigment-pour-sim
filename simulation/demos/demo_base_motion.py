"""Drive the simulated base around a small square beside the tables."""
import json

from simulation.world_model.paths import ROOT, SCENE_CONFIG
from simulation.benchmarks.pose import set_display_pose

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    output = ROOT / "generated"
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    # The planar base has no vertical suspension. Keep the floor collidable
    # with props (mask 2), but remove its contact with the robot (mask 1).
    ground = model.geom("ground").id
    model.geom_contype[ground] = 4
    model.geom_conaffinity[ground] = 2
    data = mujoco.MjData(model)
    cfg = json.loads(SCENE_CONFIG.read_text())
    set_display_pose(model, data, cfg)

    joints = [model.joint(name) for name in ("base_x", "base_y", "base_yaw")]
    position = [int(joint.qposadr[0]) for joint in joints]
    velocity = [int(joint.dofadr[0]) for joint in joints]
    start = data.qpos[position].copy()
    target = start.copy()
    timestep = model.opt.timestep
    frame_stride = max(1, round(0.08 / timestep))
    travel_steps = round(1.4 / timestep)
    settle_steps = round(0.6 / timestep)
    stiffness = np.array([4000., 4000., 1000.])
    damping = np.array([500., 500., 100.])
    frames = []

    with mujoco.Renderer(model, height=480, width=720) as renderer:
        def capture(label):
            renderer.update_scene(data, camera="overview")
            frame = Image.fromarray(renderer.render())
            x, y, _ = data.qpos[position]
            overlay = ImageDraw.Draw(frame)
            overlay.rectangle((8, 8, 280, 48), fill="#17242b")
            overlay.text((16, 14), label, fill="white")
            overlay.text((16, 29), f"Base: X {x:+.3f} m   Y {y:+.3f} m", fill="white")
            frames.append(frame)

        capture("Start")
        print(f"Start: {start[:2].round(4).tolist()} m")
        for label, dx, dy in (("-X", -.20, 0), ("+Y", 0, .20),
                              ("+X", .20, 0), ("-Y", 0, -.20)):
            origin = target.copy()
            target += [dx, dy, 0]
            for step in range(travel_steps + settle_steps):
                phase = min(1., (step + 1) / travel_steps)
                progress = phase * phase * (3 - 2 * phase)
                reference = origin + progress * (target - origin)
                # Apply feedback forces to the scene's simplified planar joints.
                # X/Y commands are forces, and the yaw command is a torque.
                data.qfrc_applied[velocity] = (
                    stiffness * (reference - data.qpos[position])
                    - damping * data.qvel[velocity]
                )
                mujoco.mj_step(model, data)
                if step % frame_stride == 0:
                    capture(f"Move {label}")
            actual = data.qpos[position][:2]
            error = np.linalg.norm(actual - target[:2])
            print(f"{label}: target {target[:2].round(4).tolist()} m, "
                  f"actual {actual.round(4).tolist()} m, error {error * 1000:.1f} mm")
            if not np.isfinite(error) or error > 0.02:
                raise RuntimeError(f"Base missed the {label} target by more than 2 cm")
        capture("Finished")

    if not any(high > low for low, high in frames[0].crop((0, 50, 720, 480)).getextrema()):
        raise RuntimeError("Renderer returned a blank scene. Try software OpenGL.")
    path = output / "base_motion.gif"
    durations = [round(frame_stride * timestep * 1000)] * len(frames)
    durations[0] = durations[-1] = 700
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
