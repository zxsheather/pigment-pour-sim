# MuJoCo pour benchmark

The fixed-start scene has a parked robot, a rigidly attached open bottle with 100 mL of water, and an empty cylinder. After one simulated second of servo settling, the benchmark starts the task and uses the settled pose as the retract target.

## Run

From the repository root, with `simulation/generated/scene.xml` already built, generate the pour scenarios:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.benchmarks.scenarios.build_pour_scenes
```

Run this in the project Python environment:

```python
from simulation.benchmarks.mujoco_pour_benchmark import run_episode
from simulation.controllers.pour_scripted import ScriptedPourController

run = run_episode(ScriptedPourController(), requested_ml=20)
print(run.status, run.collected_ml, run.spilled_ml, run.score.signed_error_ml)
```

## Interface and scoring

- `PourSystem` runs the controller and motion layer. About every 50 ms, the controller receives bottle and cylinder poses, collected and requested volumes, time, active motion status, and the last request result. It returns `Move(target, duration_s)`, `"hold"` to continue the current move, `"finish"` after retraction, or `"abort"`.
- The benchmark starts one task and applies a 60-second simulated deadline. At termination, controller and trajectory updates stop, and airborne water settles before scoring.
- Completion requires water discharge, a reached pose within 5 mm and 1° of the retract pose, and no detected workspace contact. Passing also requires at most 1 mL absolute volume error and 0.5 mL total modeled spill.

## Demos

Run with the same environment variables as scene generation. GIFs are saved under `simulation/generated/`.

| Module | Shows |
| --- | --- |
| `simulation.demos.demo_wrist_rotation` | Wrist servo tracking |
| `simulation.demos.demo_gripper_lift` | Free-bottle hold, lift, and release with simplified pads |
| `simulation.demos.demo_pour_motion` | Collision rejection and pose tracking |
| `simulation.demos.demo_pour_water` | Water collection, or spill with `--miss` |
| `simulation.demos.demo_scripted_pour` | Scripted benchmark run with phase and volume overlays |

The scripted controller budgets 9 mL of additional collection after requesting upright. The [Stage 4 TODO](../controller/TODO.json) tracks the small-model controller. See the [world-modeling record](../world-modeling/README.md) for grasp and water-model validation.
