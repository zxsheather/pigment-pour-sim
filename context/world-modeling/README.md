# World modeling

This records the implementation available for Step 1 of the [project plan](../Plan.json). The modeled pour starts with a parked base, an upright open bottle already in the gripper, and an empty cylinder. The model covers the pour and retraction. Pickup and navigation precede that handoff.

## Robot and workspace

- The [scene builder](../../simulation/world_model/build_scene.py) imports the official Quanta X1 geometry and joints into MuJoCo, adds a simplified planar base, and constructs the tables, trays, bottles, and cylinder from the photo-based [scene configuration](../../simulation/world_model/scene_config.json).
- The [actuator module](../../simulation/world_model/actuators.py) generates a shared MuJoCo XML include with position servos for six right-arm joints, the torso lift, and two fingers. Servo gains are provisional. Control position and force limits come from the model.
- The [startup script](../../simulation/benchmarks/setup_pour_pose.py) solves the upright bottle pose, sets the base and idle arm parking constraints, and writes reset keyframes and initial servo targets. The attached-bottle variant is used for motion and water demos. The free-body variant uses friction pads and finger preload. Its demo checks upright retention, lift, and release. Repeating reset and settling reproduced the robot state, bottle pose, and initial water counters in both variants.
- MuJoCo provides exact simulator state to the demos. The [offline HTML viewer](../../场景重建.html) uses the shared layout configuration and exported robot asset for geometry review.

## Water

- The [parcel model](../../simulation/world_model/water.py) uses the simulated bottle pose and mouth velocity each step. A provisional tilt-to-flow curve releases water from the lower lip. Parcels inherit mouth motion, receive an assumed exit velocity, and fall under gravity.
- A parcel crossing inside the cylinder opening counts toward collection up to cylinder capacity. Overflow and water reaching the table or floor outside the cylinder count as spill. The balance includes water remaining in the bottle, in flight, collected, and spilled.
- A rigid mass proxy represents the starting water load. An applied force and torque offset the static weight of discharged water. Internal sloshing and changing inertia are outside the current model.
- The [attached-bottle water demo](../../simulation/demos/demo_pour_water.py) uses fixed waypoints and a hand-tuned righting threshold. The nominal run collected 20.07 mL with no spill. With `--miss`, the arm aimed 50 mm to the side of the nominal target, collected no water, and spilled 20.09 mL. Both runs accounted for all 100.00 mL after the stream settled.

## Collision and contact

- The scene uses separate robot visual and collision geometry. Contact masks enable interactions between the robot and workspace props. Tables and trays use primitive shapes, and the cylinder wall is approximated by box segments. Current masks do not provide robot self-collision.
- The pouring scenario adjusts robot-floor contact for the planar base. The [workspace contact checks](../../simulation/world_model/collision.py) inspect MuJoCo contacts between the robot or attached bottle and the environment, excluding the ground. The free-grasp demo separately checks finger-pad contacts and bottle motion.
- The [motion layer](../../simulation/motion/pour_motion.py) checks sampled poses for workspace contact. The demos check contacts during simulated execution, and the motion demo includes a path through the cylinder that is expected to be rejected. Liquid collection uses geometric opening tests, with no physical interaction between parcels and cylinder walls.

The [remaining work](real-world-modeling-gaps.json) tracks validation, calibration, sensing, and integration.
