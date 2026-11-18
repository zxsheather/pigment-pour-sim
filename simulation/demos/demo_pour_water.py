"""Drive the arm through a pour and visualize the measured water transfer."""

import argparse
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.motion.motion import BottlePose, PourMotion
from simulation.world_model.water import PourWater
from simulation.world_model.paths import ROOT, SCENE_CONFIG
from simulation.world_model.collision import workspace_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miss", action="store_true", help="Aim beside the cylinder and verify spill.")
    args = parser.parse_args()
    output = ROOT / "generated"
    config = json.loads(SCENE_CONFIG.read_text())
    model = mujoco.MjModel.from_xml_path(str(output / "pour_start.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("pour_start").id)
    dt = model.opt.timestep
    for _ in range(round(1. / dt)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    motion = PourMotion(model)
    water = PourWater(model, data, config)
    start = BottlePose(tuple(data.site("held_bottle_mouth").xpos),
                       tuple(data.body("held_water_bottle").xquat))
    height = water.cylinder[2] + .195
    raised = BottlePose((*start.mouth_xyz_m[:2], height), start.bottle_quat_wxyz)
    # Aim slightly upstream of the rim center to allow for the falling stream's
    # lateral velocity. The water model scores the actual crossing position.
    aim_offset_x = -.061 if args.miss else -.011
    aim = (water.cylinder[0] + aim_offset_x, water.cylinder[1], height)
    above = BottlePose(aim, start.bottle_quat_wxyz)
    turn = np.empty(4)
    tipped = np.empty(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0., 1., 0.]), np.deg2rad(105))
    mujoco.mju_mulQuat(tipped, turn, np.asarray(start.bottle_quat_wxyz))

    frames = []
    stride = max(1, round(.05 / dt))
    with mujoco.Renderer(model, width=720, height=480) as renderer:
        def step(label, record):
            result = motion.update(dt, data)
            if result.status == "failed":
                raise RuntimeError(f"{label}: {result.reason}")
            water.apply_load(data)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if workspace_contacts(model, data):
                raise RuntimeError(f"Unexpected workspace contact during {label}")
            water.step(data, dt)
            if record:
                renderer.update_scene(data, camera="pour_start_detail")
                water.draw(renderer.scene, data)
                frame = Image.fromarray(renderer.render())
                overlay = ImageDraw.Draw(frame)
                overlay.rectangle((8, 8, 373, 48), fill="#17242b")
                overlay.text((16, 14), label, fill="white")
                overlay.text((16, 29), f"Bottle {water.remaining_ml:.1f} mL   "
                             f"Cylinder {water.collected_ml:.1f} mL   "
                             f"Spill {water.spilled_ml:.1f} mL", fill="white")
                frames.append(frame)

        def move(label, goal, duration=1.6):
            result = motion.request_move(goal, duration, data)
            if result.status != "accepted":
                raise RuntimeError(f"{label}: {result.reason}")
            for i in range(round((duration + .65) / dt)):
                step(label, i % stride == 0)
            print(f"{label}: {water.collected_ml:.2f} mL collected, "
                  f"{water.spilled_ml:.2f} mL spilled")

        move("Lift above the rim", raised)
        move("Aim the bottle", above)
        move("Tilt the bottle", BottlePose(aim, tuple(tipped)), 2.4)
        # Leave room for the water that keeps flowing while the arm rights itself.
        for i in range(round(2. / dt)):
            if water.initial_ml - water.remaining_ml >= 13.2:
                break
            step("Water falling into the cylinder", i % stride == 0)
        move("Return upright", above, 2.4)
        move("Retract", raised)
        move("Lower", start)

    balance = water.remaining_ml + water.collected_ml + water.spilled_ml + water.in_flight_ml
    if not np.isclose(balance, water.initial_ml) or water.in_flight_ml > 1e-6:
        raise RuntimeError("Water volume balance failed")
    if args.miss:
        if water.collected_ml > 1e-6 or water.spilled_ml <= .5:
            raise RuntimeError("Deliberate miss did not register the expected spill")
    elif abs(water.collected_ml - 20.) > 1. or water.spilled_ml > .5:
        raise RuntimeError("Nominal pour missed the volume or spill target")
    if not any(low < high for low, high in frames[0].getextrema()):
        raise RuntimeError("Expected a visible water transfer")
    path = output / ("pour_water_miss.gif" if args.miss else "pour_water.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(stride * dt * 1000), loop=0)
    print(f"Water accounted for: {balance:.2f} mL. Saved {path}")


if __name__ == "__main__":
    main()
