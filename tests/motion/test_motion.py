import mujoco
import numpy as np
import pytest

from simulation.motion.motion import BottlePose, PourMotion
from simulation.world_model.collision import workspace_contacts


@pytest.fixture
def setup_motion(reset_scene):
    model, data = reset_scene("pour_start")
    dt = model.opt.timestep
    mujoco.mj_step(model, data, nstep=round(1 / dt))
    mujoco.mj_forward(model, data)
    mouth = tuple(data.site("held_bottle_mouth").xpos)
    quat = tuple(data.body("held_water_bottle").xquat)
    return model, data, PourMotion(model), BottlePose(mouth, quat)


def raised_pose(start):
    x, y, z = start.mouth_xyz_m
    return BottlePose((x, y, z + .04), start.bottle_quat_wxyz)


def tick(model, data, motion):
    result = motion.update(model.opt.timestep, data)
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return result


def finish(model, data, motion, target):
    for _ in range(round(3 / model.opt.timestep)):
        result = tick(model, data, motion)
        if result.status != "executing":
            break
    assert result.status == "reached", result
    np.testing.assert_allclose(data.site("held_bottle_mouth").xpos,
                               target.mouth_xyz_m, atol=.005)
    error = np.empty(3)
    mujoco.mju_subQuat(error, np.asarray(target.bottle_quat_wxyz),
                       data.body("held_water_bottle").xquat)
    assert np.linalg.norm(error) <= np.deg2rad(1.)


def test_reachable_move(setup_motion):
    model, data, motion, start = setup_motion
    dt = model.opt.timestep
    assert motion.update(dt, data).status == "idle"
    target = raised_pose(start)
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    assert motion.request_move(target, 1.2, data).status == "accepted"
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.ctrl, ctrl)
    finish(model, data, motion, target)
    assert not np.allclose(data.ctrl, ctrl)
    np.testing.assert_array_equal(data.ctrl[model.nu - 2:], ctrl[model.nu - 2:])
    assert motion.update(dt, data).status == "reached"


def test_retarget_during_execution(setup_motion):
    model, data, motion, start = setup_motion
    assert motion.request_move(raised_pose(start), 1.6, data).status == "accepted"
    for _ in range(round(.65 / model.opt.timestep)):
        assert tick(model, data, motion).status == "executing"
    assert data.site("held_bottle_mouth").xpos[2] > start.mouth_xyz_m[2] + .005
    assert motion.request_move(start, 1., data).status == "accepted"
    finish(model, data, motion, start)


def test_rejected_path_keeps_previous_move(setup_motion):
    model, data, motion, start = setup_motion
    target = raised_pose(start)
    assert motion.request_move(target, 1.2, data).status == "accepted"
    cylinder = data.site("cylinder_mouth").xpos
    blocked = BottlePose((*cylinder[:2], start.mouth_xyz_m[2]), start.bottle_quat_wxyz)
    ctrl = data.ctrl.copy()
    rejected = motion.request_move(blocked, 1.5, data)
    assert rejected.status == "rejected"
    assert "contacts the workspace" in rejected.reason
    np.testing.assert_array_equal(data.ctrl, ctrl)
    finish(model, data, motion, target)


@pytest.mark.parametrize("duration", [0., -1., float("nan")])
def test_invalid_duration(setup_motion, duration):
    model, data, motion, start = setup_motion
    assert motion.request_move(start, duration, data).status == "rejected"
    assert motion.update(model.opt.timestep, data).status == "idle"


def test_tracking_timeout_clears_move_and_holds(setup_motion):
    model, data, motion, start = setup_motion
    assert motion.request_move(raised_pose(start), 1., data).status == "accepted"
    # Freeze physics to simulate the plant failing to track the commands.
    for _ in range(round(2 / model.opt.timestep)):
        result = motion.update(model.opt.timestep, data)
        if result.status == "failed":
            break
    assert result.status == "failed"
    assert "did not settle" in result.reason
    np.testing.assert_allclose(data.ctrl[motion.servos], data.qpos[motion.qpos])
    assert motion.update(model.opt.timestep, data) == result


def test_contact_during_execution_fails_move(setup_motion):
    model, data, motion, start = setup_motion
    assert motion.request_move(raised_pose(start), 1.2, data).status == "accepted"
    dt = model.opt.timestep
    assert motion.update(dt, data).status == "executing"
    # Override the servo externally to exercise the execution guard independently
    # of the planner's own collision rejection.
    servo = model.actuator("right_arm_joint2_servo").id
    data.ctrl[servo] = model.key("pour_start").ctrl[servo] + .04

    for _ in range(round(1 / dt)):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if workspace_contacts(model, data):
            break
    else:
        pytest.fail("Disturbance did not produce workspace contact")
    result = motion.update(dt, data)
    assert result.status == "failed"
    assert result.reason == "Workspace contact during execution"
    np.testing.assert_allclose(data.ctrl[motion.servos], data.qpos[motion.qpos])
