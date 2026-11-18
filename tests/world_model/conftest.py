import json

import mujoco
import pytest

from simulation.benchmarks.setup_pour_pose import build_pour_scenes
from simulation.world_model.build_scene import build
from simulation.world_model.paths import ROOT, SCENE_CONFIG


@pytest.fixture(scope="session")
def config():
    return json.loads(SCENE_CONFIG.read_text())


@pytest.fixture(scope="session")
def scenes(tmp_path_factory, config):
    assets = ROOT / "third_party" / "sdk_robot"
    if not (assets / "urdf/quanta_x1/ex001_robot.urdf").is_file():
        pytest.fail("Missing X1 assets. Run python scripts/bootstrap_sources.py.")
    output = tmp_path_factory.mktemp("world")
    build(assets, output, SCENE_CONFIG)
    return dict(zip(("pour_start", "grasp_start"), build_pour_scenes(output, config)))


@pytest.fixture
def reset_scene(scenes):
    def reset(key):
        model = mujoco.MjModel.from_xml_path(str(scenes[key]))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        mujoco.mj_forward(model, data)
        return model, data

    return reset
