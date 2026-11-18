import mujoco
import numpy as np

from simulation.benchmarks.scenarios.build_pour_scenes import GRIPPER
from simulation.world_model.actuators import FINGER_JOINTS


def test_free_bottle_hold_lift_and_release(reset_scene):
    model, data = reset_scene("grasp_start")
    bottle = data.body("held_water_bottle")
    gripper = data.body(GRIPPER)
    bottle_geom = model.geom("held_bottle_collision").id
    pads = {model.geom(f"grip_pad_{side}").id for side in ("left", "right")}
    lift = model.actuator("lift_joint_servo").id
    start = data.ctrl[lift]
    dt = model.opt.timestep

    def relative_position():
        return gripper.xmat.reshape(3, 3).T @ (bottle.xpos - gripper.xpos)

    relative_start = relative_position()
    for _ in range(round(1.6 / dt)):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(relative_position() - relative_start) <= .005
    initial_z = bottle.xpos[2]
    for step in range(round(1.6 / dt)):
        phase = min(1., (step + 1) * dt)
        data.ctrl[lift] = start + .04 * phase**2 * (3 - 2 * phase)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(relative_position() - relative_start) <= .005
        contacts = {int(geom) for pair in data.contact.geom if bottle_geom in pair
                    for geom in pair if geom != bottle_geom}
        assert contacts <= pads
    assert contacts == pads
    assert abs(bottle.xpos[2] - initial_z - .04) <= .005

    for name, target in zip(FINGER_JOINTS, (.044, -.044)):
        data.ctrl[model.actuator(f"{name}_servo").id] = target
    release_z = bottle.xpos[2]
    mujoco.mj_step(model, data, nstep=round(.6 / dt))
    mujoco.mj_forward(model, data)
    assert bottle.xpos[2] < release_z - .07
