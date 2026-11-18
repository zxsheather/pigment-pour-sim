import mujoco
import numpy as np
import pytest

from simulation.world_model.water import PourWater


@pytest.mark.parametrize("case", ["upright", "capture", "miss", "overflow"])
def test_prescribed_bottle_pose(config, case):
    # Prescribe pose independently of arm IK, servo tracking, and grasp.
    model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <site name="cylinder_mouth" pos=".88 .3 1"/>
          <body name="held_water_bottle" mocap="true">
            <geom name="held_water" type="sphere" size=".02"/>
            <site name="held_bottle_mouth" size=".001"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    data = mujoco.MjData(model)
    data.mocap_pos[0] = [.88 + (.15 if case == "miss" else 0), .3, 1.2]
    if case != "upright":
        mujoco.mju_axisAngle2Quat(data.mocap_quat[0], np.array([0., 1., 0.]), 1.83)
    if case == "overflow":
        config = {**config, "cylinder": {**config["cylinder"], "capacity_ml": 1.}}
    mujoco.mj_forward(model, data)
    water = PourWater(model, data, config)

    def advance(steps):
        for _ in range(steps):
            water.step(data, .004)
            assert water.remaining_ml >= 0
            assert water.remaining_ml + water.in_flight_ml + water.collected_ml + water.spilled_ml == pytest.approx(water.initial_ml)

    advance(125)
    data.mocap_quat[0] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    advance(250)
    assert water.in_flight_ml == 0
    expected = {
        "upright": (0, 0),
        "capture": (6, 0),
        "miss": (0, 6),
        "overflow": (1, 5),
    }
    assert (water.collected_ml, water.spilled_ml) == pytest.approx(expected[case], abs=.05)


def test_discharged_water_reduces_static_load(reset_scene, config):
    model, data = reset_scene("pour_start")
    water = PourWater(model, data, config)
    water.apply_load(data)
    np.testing.assert_array_equal(data.xfrc_applied[water.body], np.zeros(6))
    water.remaining_ml -= 20
    water.apply_load(data)
    np.testing.assert_allclose(data.xfrc_applied[water.body, :3], [0, 0, .1962])
