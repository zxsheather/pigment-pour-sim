import mujoco
import numpy as np
import pytest

from simulation.world_model.collision import workspace_contacts


@pytest.mark.parametrize("key", ["pour_start", "grasp_start"])
def test_reset_geometry_and_repeatability(reset_scene, config, key):
    model, data = reset_scene(key)
    cylinder = data.site("cylinder_mouth").xpos
    np.testing.assert_allclose(cylinder[:2], config["cylinder"]["center_xy"])
    np.testing.assert_allclose(data.site("held_bottle_mouth").xpos - cylinder,
                               [0, -.05, .06], atol=.0005)
    np.testing.assert_allclose(data.body("held_water_bottle").xmat.reshape(3, 3)[:, 2],
                               [0, 0, 1], atol=.003)
    if key == "pour_start":
        assert not workspace_contacts(model, data)
    else:
        assert model.joint("held_bottle_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    initial_qpos, initial_ctrl = data.qpos.copy(), data.ctrl.copy()
    mujoco.mj_step(model, data, nstep=20)
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    np.testing.assert_array_equal(data.qpos, initial_qpos)
    np.testing.assert_array_equal(data.ctrl, initial_ctrl)
    assert not data.qvel.any()


def test_workspace_contact_detected_during_arm_motion(reset_scene):
    model, data = reset_scene("pour_start")
    dt = model.opt.timestep
    mujoco.mj_step(model, data, nstep=round(1. / dt))
    assert not workspace_contacts(model, data)
    servo = model.actuator("right_arm_joint2_servo").id
    data.ctrl[servo] += .04
    for _ in range(round(1. / dt)):
        mujoco.mj_step(model, data)
        if workspace_contacts(model, data):
            return
    pytest.fail("Expected the arm step to contact the workspace")
