# Pour phase benchmark

This task-space benchmark checks controller logic. The [project plan](../context/Plan.json) describes the proposed MuJoCo benchmark and controller interface.

The episode starts with the base parked, the source bottle already held upright and uncapped, and an empty graduated cylinder in front of the robot. The bottle mouth starts within ±5 cm of the cylinder mouth along each horizontal axis and 5–10 cm above it. The source starts with 100 mL. Each episode requests 15, 20, or 25 mL.

The controller reads the mouth-to-mouth offset in meters, the bottle tilt in radians, collected and remaining volume, and the target volume. It commands three mouth velocities and a tilt rate every 50 ms. The benchmark ends successfully when the bottle is upright within 0.1 rad, its mouth is still aligned with the cylinder, the volume error is at most 1 mL, and spill is at most 0.5 mL. Episodes have an eight-second limit.

This first benchmark integrates commands directly in task space. It assumes exact observations and uses a simplified flow curve based on tilt. Liquid is counted as collected only when the mouth is within 12 mm horizontally and 35–90 mm above the cylinder mouth. The benchmark does not exercise arm joint limits, collisions, camera perception, or real liquid behavior.

## Run both controllers

From the repository root, with the Python environment installed:

```bash
.venv/bin/python -m simulation.benchmarks.pour_benchmark
.venv/bin/python -m simulation.controllers.pour_learned
.venv/bin/python -m simulation.controllers.pour_learned --load
```

The scripted controller aligns the mouths, tilts to pour, and begins to right the bottle early to account for flow during the return. The learned controller fits a linear model for translation and a small decision tree for tilt using the scripted controller's demonstrations and corrections near those demonstrations. Training uses seed 41 and saves the parameters to `simulation/generated/pour_learned.npz`. The default evaluation uses a separate seed (2026) and runs the same 60 starts for both controllers.

## Checkpoints

| Checkpoint | Completion check |
| --- | --- |
| Idealized reference | Both controllers finish all 60 shared cases. Record success rate, final volume error, spill, and time. Check that idle and deliberately misaligned pours fail. |
| MuJoCo arm integration | Start with the bottle attached to the gripper. Convert mouth velocity and tilt commands to arm joint targets using the known grasp transform. Verify both controllers can complete a nominal pour with joint limits and collision checks before testing randomized starts. Reuse the simplified flow rule here to isolate arm-control failures. |
| Observation and flow robustness | Measure the bottle and cylinder poses and obtain a usable estimate of collected volume. Add measured pose error, observation delay, and flow variation. Agree on bounds before running a separate test set; target at least 95% success for each approach under those agreed conditions. |
| Bench validation | Calibrate with water and measure actual collected volume with a scale. Verify the nominal case before varying bottle placement or requested volume; compare measured spill and error against the simulation results. |

An imitation policy matching its teacher is evidence that the training and deployment pipeline works. To test whether learning improves the task, compare both controllers on the same perturbations once those perturbations are based on measurements.

## MuJoCo starting pose

After generating `simulation/generated/scene.xml`, run:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.benchmarks.setup_pour_pose
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
- [ ] Connect the scripted and learned dosing commands to the motion layer. Calibrate the water model against measured pours before comparing the controllers.

## Water transfer in the arm scene

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.demos.demo_pour_water
```

The arm moves the attached bottle. The water model uses the bottle's measured pose to discharge parcels from the lower lip. Each parcel inherits the mouth's velocity, receives an assumed exit velocity, and falls under gravity. Its crossing position at the cylinder opening determines collection. Water missing the opening counts as spill when it reaches the tabletop or floor. The animation shows the stream, the changing fill levels, and the volume totals. The simulation compensates for the weight of discharged water, and checks the volume balance at the end.

This is a reduced model. Flow onset, discharge rate, and exit velocity are provisional. The model approximates the remaining water's static weight and does not resolve internal sloshing, inertia changes, splashes, or surface interactions. The demo uses a hand-tuned stop threshold for the 20 mL example. It saves `simulation/generated/pour_water.gif`.

Pass `--miss` to the same command to aim 50 mm beside the nominal target. The demo checks that the cylinder stays empty, spill registers, and all water is accounted for after settling. It saves `simulation/generated/pour_water_miss.gif`.
