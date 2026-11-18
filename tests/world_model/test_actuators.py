import mujoco
import numpy as np
import pytest

from simulation.world_model.actuators import ARM_JOINTS, FINGER_JOINTS


@pytest.mark.parametrize("name", [*ARM_JOINTS, "lift_joint", *FINGER_JOINTS])
def test_servo_moves_its_joint_and_returns(reset_scene, name):
    model, data = reset_scene("pour_start")
    servo = model.actuator(f"{name}_servo")
    joint = model.joint(name)
    assert model.actuator_trnid[servo.id, 0] == joint.id
    np.testing.assert_allclose(servo.ctrlrange, joint.range)
    # Isolate tracking from workspace contacts, which have a separate test.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    dt = model.opt.timestep

    def execute(target, seconds):
        data.ctrl[servo.id] = target
        mujoco.mj_step(model, data, nstep=round(seconds / dt))
        return data.joint(name).qpos[0]

    start = data.ctrl[servo.id]
    baseline = execute(start, 1.)
    linear = joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE
    delta, tolerance = (.004, .002) if linear else (.015, .0075)
    directions_tested = 0
    for direction in (-1, 1):
        target = start + direction * delta
        if not servo.ctrlrange[0] <= target <= servo.ctrlrange[1]:
            continue
        directions_tested += 1
        measured = execute(target, 1.)
        assert direction * (measured - baseline) >= delta / 2
        # Compare against the settled baseline to allow steady load error.
        assert measured == pytest.approx(baseline + direction * delta, abs=tolerance)
        assert execute(start, 1.) == pytest.approx(baseline, abs=tolerance)
    assert directions_tested
