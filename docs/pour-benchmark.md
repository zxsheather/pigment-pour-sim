# Pour phase benchmark

## Fixed-start MuJoCo benchmark

After generating the pour scene with `simulation.benchmarks.scenarios.build_pour_scenes`, call
`simulation.benchmarks.mujoco_pour_benchmark.run_episode` with a controller. The benchmark
loads the attached-bottle scene, settles the servos for one simulated second,
and starts the episode clock at zero with a fresh water model. The settled
starting bottle pose also serves as the retract target. The scene starts with
a parked robot, a held open bottle containing 100 mL, and an empty cylinder.

```python
from simulation.benchmarks.mujoco_pour_benchmark import run_episode

run = run_episode(lambda observation: "hold", requested_ml=20.)
print(run.status, run.collected_ml, run.spilled_ml, run.score.signed_error_ml)
# timeout 0.0 0.0 -20.0
```

The benchmark starts one task with the requested volume. The pouring system
in `simulation.pour_system` runs the controller and motion layer together.
The controller receives bottle and cylinder poses, simulated collected volume,
requested volume, time, active motion status, and the last request result at
approximately 50 ms intervals. It can return `Move(target, duration_s)`,
`"hold"` to keep the current motion request, `"finish"` after retraction, or
`"abort"`. Use a fresh controller instance for each stateful run.

The deadline defaults to 60 seconds of simulated time. At termination the
controller and active trajectory stop. Already airborne water lands before
scoring, with additional settling time recorded separately. A completed trial
requires discharge, a reached motion target within 5 mm and 1° of the retract
pose, and no detected workspace contacts. The rigid attachment supplies grasp
retention. Passing also requires at most 1 mL absolute volume error and
0.5 mL total spill.

## Checkpoints

| Checkpoint | Completion check |
| --- | --- |
| Stage 4 controller integration | Build scripted and learned controllers that issue timed bottle-pose targets through the pouring system. Verify both on the same fixed-start MuJoCo benchmark before varying initial conditions. |
| Observation and flow robustness | Measure the bottle and cylinder poses and obtain a usable estimate of collected volume. Add measured pose error, observation delay, and flow variation. Agree on bounds before running a separate test set; target at least 95% success for each approach under those agreed conditions. |
| Bench validation | Calibrate with water and measure actual collected volume with a scale. Verify the nominal case before varying bottle placement or requested volume; compare measured spill and error against the simulation results. |

An imitation policy matching its teacher is evidence that the training and deployment pipeline works. To test whether learning improves the task, compare both controllers on the same perturbations once those perturbations are based on measurements.

## MuJoCo starting pose

After generating `simulation/generated/scene.xml`, run:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.benchmarks.scenarios.build_pour_scenes
```

The script saves `simulation/generated/pour_start.xml` and two PNG previews. Reset to the `pour_start` keyframe before running a controller. The right gripper carries a rigidly attached open bottle with a 100 mL water proxy. Its mouth starts 50 mm toward the robot and 60 mm above the cylinder mouth. The script solves for the arm and lift pose and checks for contacts with the workspace at reset. The water proxy adds mass and shows the fill level; it has no fluid dynamics. The preview is a static snapshot. Future motion must clear the cylinder rim as the arm brings the bottle into pouring position.

Robot and workspace geometry live in the generated `scene.xml`. The startup script adds the held bottle and the task-specific parking constraints and reset keyframes. Both `pour_start.xml` and `pour_grasp.xml` include the same generated `pour_actuators.xml` alongside them. Edit servo settings in `simulation/world_model/actuators.py` and rerun the startup script to regenerate the models.

The generated scenes have six right-arm position actuators, one torso lift actuator, and two mirrored finger position actuators. The wrist demo keeps the lift target at its starting height. The base remains parked and the left arm's joint angles are held at their starting values. The torso lift still carries both arms vertically. To send wrist commands through the simulated actuators and record the motion, run:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.demos.demo_wrist_rotation
```

The demo commands wrist offsets of +15°, −15°, and 0°, checks workspace contacts along the run, and saves `simulation/generated/wrist_rotation.gif`. Servo gains are simulation settings, pending hardware calibration.

The setup script also generates `simulation/generated/pour_grasp.xml`. Its `grasp_start` keyframe assumes finger closure has already occurred. The bottle is a free body held by two simplified friction pads. The finger servos use mirrored position commands to preload the grip. The pads, friction, and servo gains are provisional simulation settings. The water is still a solid mass proxy.

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.demos.demo_gripper_lift
```

This demo checks upright retention, raises the loaded torso by 40 mm, returns it to the starting height, opens the fingers to release the bottle, and closes them again. It checks both pad contacts at each loaded target, bottle slip while carrying, workspace contact, lift tracking, and release. The animation is saved to `simulation/generated/gripper_lift.gif`.

## Arm motion with an attached bottle

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.demos.demo_pour_motion
```

The demo rejects an upright move through the cylinder. It then moves through six absolute bottle poses: lift above the rim, move over the cylinder, tilt 35°, return upright, retract, and lower. The arm and lift move through actuator commands. The demo measures each endpoint and checks modeled workspace contacts throughout. It saves `simulation/generated/pour_motion.gif`.

- [ ] Validate a free bottle grasp with the imported X1 fingers, without the added pads or support from the cylinder. Verify upright hold, lift, and a slow tilt. The previous trial with imported fingers shifted about 12 mm and touched the cylinder. Use the attached bottle for motion-layer work until this check passes.
- [ ] Build the Stage 4 controllers against the pouring system interface. Calibrate the water model against measured pours before comparing the controllers.

## Water transfer in the arm scene

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.demos.demo_pour_water
```

The arm moves the attached bottle. The water model uses the bottle's measured pose to discharge parcels from the lower lip. Each parcel inherits the mouth's velocity, receives an assumed exit velocity, and falls under gravity. Its crossing position at the cylinder opening determines collection. Water missing the opening counts as spill when it reaches the tabletop or floor. The animation shows the stream, the changing fill levels, and the volume totals. The simulation compensates for the weight of discharged water, and checks the volume balance at the end.

This is a reduced model. Flow onset, discharge rate, and exit velocity are provisional. The model approximates the remaining water's static weight and does not resolve internal sloshing, inertia changes, splashes, or surface interactions. The demo uses a hand-tuned stop threshold for the 20 mL example. It saves `simulation/generated/pour_water.gif`.

Pass `--miss` to the same command to aim 50 mm beside the nominal target. The demo checks that the cylinder stays empty, spill registers, and all water is accounted for after settling. It saves `simulation/generated/pour_water_miss.gif`.
