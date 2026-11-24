"""Integration regression: execute the recorded chunk that reset MuJoCo."""
from pathlib import Path
import argparse
import json
import tempfile
import unittest

from grasp_walloss import GraspEnvironment


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=ROOT/"generated/scene.xml")
parser.add_argument("--config", type=Path, default=ROOT.parents[1]/"simulation/scene_config.json")
parser.add_argument("--trace", type=Path, default=ROOT/"fixtures/recorded_chunk.json")
args, extra = parser.parse_known_args()


class RecordedGripperCommand(unittest.TestCase):
    def test_entire_recorded_chunk_has_continuous_time(self):
        actions = json.loads(args.trace.read_text())["follow2_pos"]
        with tempfile.TemporaryDirectory() as folder:
            env = GraspEnvironment(args.scene, args.config, Path(folder), "blue")
            try:
                start = float(env.d.time)
                for pose in actions:
                    before = float(env.d.time)
                    env.execute(pose)
                    self.assertGreater(env.d.time, before)
                expected = len(actions)*round(env.step_time/env.m.opt.timestep)*env.m.opt.timestep
                self.assertAlmostEqual(env.d.time-start, expected, places=6)
            finally:
                env.renderer.close()


if __name__ == "__main__":
    unittest.main(argv=[__file__, *extra])
