"""Test deeper seating using fixed recorded approaches and identical grasp hold.

Depth is a Cartesian target offset along the requested gripper's positive X
axis, ramped in before closure. FK, IK, TCP definition, collision geometry,
friction and actuator limits are unchanged. This is a privileged replay
diagnostic, not independent model execution or a hardware result.
"""
from pathlib import Path
import argparse
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from diagnostic_environment import DiagnosticEnvironment, PhysicsTrace
from reactive_expert import apply_setup


APPROACHES = {
    "original": "takeover_training_eval_v6/augmented_closedloop_internal",
    "direct8": "feedback_diagnostic_v7/direct8",
    "repeat8": "feedback_diagnostic_v7/repeat8",
}
CASE = "train_episode_reactive_train_v4_e0000"


def read(path):
    return json.loads(path.read_text())


def hand_geometry(env):
    rotation = env.d.xmat[env.ee].reshape(3, 3)
    origin = env.d.xpos[env.ee]
    bounds = {}
    for body in [env.ee, *sorted(env.fingers)]:
        vertices = []
        for geom in range(env.m.ngeom):
            if env.m.geom_bodyid[geom] != body or not (env.m.geom_contype[geom] or env.m.geom_conaffinity[geom]):
                continue
            if env.m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = env.m.geom_dataid[geom]
            start, count = env.m.mesh_vertadr[mesh], env.m.mesh_vertnum[mesh]
            world = env.d.geom_xpos[geom] + env.m.mesh_vert[start:start + count] @ env.d.geom_xmat[geom].reshape(3, 3).T
            vertices.append((world - origin) @ rotation)
        if vertices:
            values = np.concatenate(vertices)
            bounds[env.m.body(body).name] = {"min_m": values.min(axis=0).tolist(), "max_m": values.max(axis=0).tolist()}
    fingers = [bounds[env.m.body(body).name] for body in sorted(env.fingers)]
    descendants = set()
    for body in range(env.m.nbody):
        ancestor = body
        while ancestor and ancestor != env.ee:
            ancestor = int(env.m.body_parentid[ancestor])
        if ancestor == env.ee:
            descendants.add(body)
    return {"bounds": bounds, "finger_root_x_m": min(b["min_m"][0] for b in fingers),
            "finger_tip_x_m": max(b["max_m"][0] for b in fingers)}, descendants


class SeatingTrace(PhysicsTrace):
    def __init__(self, depth_m, geometry, hand_bodies, lift_axis=None):
        super().__init__()
        self.depth_m, self.geometry, self.hand_bodies = depth_m, geometry, hand_bodies
        self.handoff = None
        self.retained_s = 0.
        self.lift_axis = lift_axis

    def handoff_allowed(self, env):
        return True

    def command(self, raw, control_step, chunk):
        command = super().command(raw, control_step, chunk)
        if self.handoff is None:
            fraction = min(1., control_step * .04 / .4)
            offset = self.requested_rotation.apply([self.depth_m * fraction, 0., 0.])
            command[:3] += offset
            assert np.array_equal(command[3:], raw[3:])
            assert abs(np.linalg.norm(command[:3] - raw[:3]) - self.depth_m * fraction) < 1e-9
        else:
            command[:6] = self.handoff["tool_pose"][:6]
            if self.lift_axis is not None:
                elapsed = control_step * .04 - self.handoff["time_s"]
                fraction = float(np.clip((elapsed - .2) / 2., 0., 1.))
                smooth = 10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5
                command[:3] += np.asarray(self.lift_axis) * (.12 * smooth)
            command[6] = 2.2
        self.applied = command.copy()
        self.requested_rotation = Rotation.from_euler("xyz", command[3:6])
        return command

    def observe(self, env):
        super().observe(env)
        rotation = env.d.xmat[env.ee].reshape(3, 3)
        origin = env.d.xpos[env.ee]
        local_center = (env.d.xpos[env.bottle] - origin) @ rotation
        contact_x, touched = [], set()
        for con in env.d.contact:
            bodies = [int(env.m.geom_bodyid[g]) for g in [con.geom1, con.geom2]]
            if env.bottle not in bodies:
                continue
            other = bodies[1] if bodies[0] == env.bottle else bodies[0]
            if other in self.hand_bodies:
                touched.add(other)
            if other in env.fingers:
                contact_x.append(float(((con.pos - origin) @ rotation)[0]))
        row = self.rows[-1]
        self.retained_s = self.retained_s + float(env.m.opt.timestep) if row["lift_m"] >= .05 and touched else 0.
        row.update(bottle_x_in_hand_m=float(local_center[0]),
            bottle_y_in_hand_m=float(local_center[1]), bottle_z_in_hand_m=float(local_center[2]),
            bottle_center_behind_tip_m=self.geometry["finger_tip_x_m"] - float(local_center[0]),
            contact_x_mean_m=float(np.mean(contact_x)) if contact_x else None,
            contact_behind_tip_m=self.geometry["finger_tip_x_m"] - max(contact_x) if contact_x else None,
            any_hand_contact=bool(touched), palm_contact_bodies=len(touched - env.fingers),
            retained_s=self.retained_s)
        if (self.handoff is None and self.handoff_allowed(env) and self.contact_seconds >= .08 - 1e-9
                and self.applied[6] < min(env.gripper_feedback()["opening_motor_units"]) - .1):
            self.handoff = {"time_s": row["time_s"], "tool_pose": env.arm_pose("right").tolist(),
                "bottle_x_in_hand_m": float(local_center[0]),
                "bottle_center_behind_tip_m": row["bottle_center_behind_tip_m"],
                "contact_behind_tip_m": row["contact_behind_tip_m"],
                "uses_simulator_contact_identity": True, "hold_gripper": 2.2,
                "takes_effect_next_control_step": True}


def closeup(env, depth_mm):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = env.d.xpos[env.bottle]
    camera.distance, camera.azimuth, camera.elevation = .30, 90, -12
    env.renderer.update_scene(env.d, camera=camera, scene_option=env.opt)
    result = Image.fromarray(env.renderer.render())
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 28), fill=(20, 28, 26))
    draw.text((8, 8), f"REPLAY + {depth_mm}mm INSERTION | EVALUATION CAMERA", fill="white")
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/grasp_depth_v8"))
    parser.add_argument("--config", type=Path, default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    parser.add_argument("--depths-mm", type=int, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--approaches", choices=list(APPROACHES), nargs="+", default=list(APPROACHES))
    parser.add_argument("--controlled-lift", action="store_true",
                        help="After the same grasp trigger, settle 0.2s then lift 12cm in world Z over 2s and hold")
    args = parser.parse_args()
    assert len(set(args.depths_mm)) == len(args.depths_mm) and all(0 <= d <= 30 for d in args.depths_mm)
    output = args.output / ("lift_replays" if args.controlled_lift else "replays")
    output.mkdir(parents=True, exist_ok=False)
    source = args.runs / "reactive_train_v4/episode_0000"
    protocol = {"same_fixed_scene": True, "different_recorded_approaches_not_independent_positions": True,
        "approaches": args.approaches, "depths_mm": args.depths_mm,
        "target_offset_axis": "positive local gripper X, transformed into arm-base coordinates",
        "offset_ramp_seconds": .4, "tcp_definition_unchanged_m": [.132, 0., 0.],
        "same_postgrasp_controller": "after 80ms bilateral closing resistance, hold current actual tool pose and command gripper 2.2",
        "trigger_uses_simulator_contact_identity": True, "model_inference_executed": False,
        "physics_friction_mass_actuator_limits_unchanged": True,
        "max_control_steps": 384, "max_simulated_seconds": 15.36,
        "original_criterion": "bottom lift >=5cm and bilateral contact continuously >=1s at any time",
        "extended_criterion": "bottom lift >=5cm and bilateral contact continuously >=3s at the end",
        "secondary_retention": "bottom lift >=5cm and some physical finger/palm contact, continuously at end",
        "not_model_successes": True}
    if args.controlled_lift:
        protocol["same_postgrasp_controller"] = "after 80ms bilateral closing resistance, settle current pose for 0.2s; lift 12cm in world Z over 2s with minimum-jerk profile, hold pose and gripper 2.2"
    prefix = "lift_" if args.controlled_lift else ""
    (args.output / (prefix + "protocol.json")).write_text(json.dumps(protocol, indent=2))
    report = {"completed": False, "protocol": protocol, "cases": []}
    for approach in args.approaches:
        recorded = args.runs / APPROACHES[approach] / CASE
        commands = np.asarray(read(recorded / "actions.json"))[:, :8].reshape(-1, 7)
        expected = read(recorded / "telemetry.json")
        assert len(commands) == len(expected) == 384
        for depth_mm in args.depths_mm:
            name = f"{approach}_depth_{depth_mm:02d}mm"
            folder = output / name
            suffix = "PRIVILEGED SLOW LIFT" if args.controlled_lift else "PRIVILEGED HOLD"
            env = DiagnosticEnvironment(source / "input_scene.xml", args.config, folder, "blue", tcp="grasp",
                controller_label=f"RECORDED POLICY + {depth_mm}mm INSERTION + {suffix}")
            env.step_time = .04
            apply_setup(env, read(source / "episode.json")["setup"])
            geometry, bodies = hand_geometry(env)
            lift_axis = env.d.xmat[env.arm_base].reshape(3, 3).T @ np.array([0., 0., 1.]) if args.controlled_lift else None
            trace = SeatingTrace(depth_mm / 1000, geometry, bodies, lift_axis=lift_axis)
            env.physics_trace = trace
            result = {"approach": approach, "depth_mm": depth_mm, "directory": name,
                "trial_valid": False, "model_inference_executed": False,
                "controlled_lift": args.controlled_lift}
            close_frames, prefix_error = [], 0.
            captured_handoff = False
            try:
                for step, raw in enumerate(commands):
                    before_handoff = trace.handoff is None
                    command = trace.command(raw, step, step // 8)
                    row = env.execute(command)
                    if depth_mm == 0 and before_handoff:
                        for field in ["right_ee", "bottle_xyz", "held_seconds"]:
                            prefix_error = max(prefix_error, float(np.max(np.abs(np.asarray(row[field]) - expected[step][field]))))
                    env.capture()
                    if approach == args.approaches[0]:
                        close_frames.append(closeup(env, depth_mm))
                    if trace.handoff and not captured_handoff:
                        Image.fromarray(closeup(env, depth_mm)).save(folder / "grasp_closeup.png")
                        captured_handoff = True
                assert depth_mm != 0 or prefix_error < 1e-6
                diagnostics = trace.finish(folder, float(env.m.opt.timestep))
                final = trace.rows[-1]
                result.update(trial_valid=True, original_criterion_reached=bool(env.success),
                    max_bilateral_lifted_s=diagnostics["max_bilateral_lifted_s"],
                    end_bilateral_lifted_s=final["held_s"], end_retained_s=final["retained_s"],
                    end_bilateral_at_least_3s=final["held_s"] >= 3.,
                    end_retention_at_least_3s=final["retained_s"] >= 3.,
                    final_lift_m=final["lift_m"], final_finger_contacts=final["contacts"],
                    final_palm_contact_bodies=final["palm_contact_bodies"],
                    final_bottle_x_in_hand_m=final["bottle_x_in_hand_m"],
                    handoff=trace.handoff, geometry=geometry,
                    baseline_prefix_max_abs_error=prefix_error if depth_mm == 0 else None,
                    diagnostics=diagnostics)
            finally:
                env.capture("final")
                env.close()
                if close_frames:
                    imageio.mimsave(folder / "closeup.mp4", close_frames, fps=25, codec="libx264", macro_block_size=2)
            (folder / "result.json").write_text(json.dumps(result, indent=2))
            report["cases"].append(result)
            (args.output / (prefix + "report.json")).write_text(json.dumps(report, indent=2))
            print(json.dumps({key: value for key, value in result.items() if key not in ["geometry", "diagnostics", "handoff"]}), flush=True)
    report["completed"] = True
    (args.output / (prefix + "report.json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
