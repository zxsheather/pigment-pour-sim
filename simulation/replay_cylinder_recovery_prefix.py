"""Generate teacher recovery start states by replaying bounded joint commands.

The prefix is physically executed in new training-only reset cases using the
original gripper geometry and servos. No model inference or task scoring occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_gripper import audit
from carry_env import CONTROL_DT
from cylinder_joint_actions import APERTURE_SCALE_M, JOINT_ACTION_WIDTH, JOINT_STATE_WIDTH, require, robot_joint_observation
from inspect_cylinder import CylinderEnv


def save_case(root, case, controller, source, commands, snapshot_frames, reference=None):
    env = CylinderEnv(case['volume'], **controller['environment'],
                      friction_scale=case.get('friction_scale', 1.),
                      mass_scale=case.get('mass_scale', 1.))
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0.))
        initial_base_y = float(env.base_pose()[1])
        geometry = audit(env)
        require(geometry['passed'] and geometry['custom_pads'] == 0, 'Official geometry audit failed')
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        size = mujoco.mj_stateSize(env.model, spec)
        actuator_ids = sorted(env.hold_targets)
        states, targets, base, observations, notes = [], [], [], [], []
        collision = None
        for frame in range(snapshot_frames[-1] + 1):
            observation = robot_joint_observation(env)
            if reference is not None:
                np.testing.assert_array_equal(observation, reference[frame])
            if frame in snapshot_frames:
                state = np.empty(size)
                mujoco.mj_getState(env.model, env.data, state, spec)
                states.append(state)
                targets.append([env.hold_targets[i] for i in actuator_ids])
                base.append(env.applied_action.copy())
                observations.append(observation)
                shaft_y = env.data.xmat[env.gripper_body].reshape(3, 3)[1, 0]
                notes.append(dict(index=len(notes), frame=frame, time_s=float(env.data.time),
                                  aperture_m=float(observation[6] * APERTURE_SCALE_M),
                                  base_retreat_m=initial_base_y - float(env.base_pose()[1]),
                                  wrist_shaft_error_deg=float(np.degrees(np.arccos(np.clip(shaft_y, -1., 1.)))),
                                  max_right_joint_speed_rad_s=float(abs(env.data.qvel[env.arm_dofs]).max()),
                                  grasp_distance_m=float(np.linalg.norm(env.grip_point_relative()
                                                                       - env.grasp_offset)),
                                  loaded_contacts=env.metrics()['grip_contacts']))
            if frame < snapshot_frames[-1]:
                command = commands[frame]
                _, metrics = env.step_joints(command[:6], opening=float(command[6]),
                                             forward=float(command[7]), yaw=float(command[8]))
                if metrics['forbidden_contacts']:
                    collision = dict(frame=frame, contacts=metrics['forbidden_contacts'])
                    break
        if reference is not None:
            require(collision is None and len(notes) == len(snapshot_frames),
                    'Unexpected collision in the source prefix')
            return dict(verified_frames=snapshot_frames[-1] + 1, exact_robot_observations=True)
        folder = root / f"volume_{case['volume']:g}_seed_{case['seed']}"
        episode = folder / 'episode_000'
        episode.mkdir(parents=True)
        if notes:
            np.savez_compressed(episode / 'recovery_physics.npz', state_spec=spec,
                                full_state=states, hold_actuator_ids=actuator_ids,
                                hold_targets=targets, applied_base_action=base)
            np.savez_compressed(episode / 'model_observations.npz',
                                timestamp=[row['time_s'] for row in notes], state=observations)
        summary = dict(case=case, source_motion=source, geometry_audit=geometry,
                       snapshots=notes, collision=collision, task_success_measured=False)
        (episode / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (folder / 'summary.json').write_text(json.dumps(dict(controller=controller,
            case=case, task_success_measured=False), indent=2) + '\n')
        return dict(episode=str(episode), snapshots=notes, collision=collision)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--snapshot-frames', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frames = sorted(set(args.snapshot_frames))
    require(len(frames) == len(args.snapshot_frames) and frames[0] > 0,
            'Use unique positive snapshot frames')
    summary = json.loads((args.rollout / 'summary.json').read_text())
    require(summary['evaluation_complete'] and summary['action_space'] == 'joint'
            and summary['total'] == 1, 'Expected a completed single joint-policy rollout')
    result = summary['episodes'][0]
    require(result['geometry_audit']['passed'], 'Source geometry audit failed')
    source_case = result['case']
    cases = json.loads(args.cases.read_text())
    seeds = {case['seed'] for case in cases}
    require(len(seeds) == len(cases) > 0 and source_case['seed'] not in seeds,
            'Use distinct new training seeds')
    trajectory = args.rollout / 'episode_000/trajectory.npz'
    with np.load(trajectory) as raw:
        end = frames[-1] + 1
        require(end <= len(raw['timestamp']), 'Snapshot lies beyond the completed source rollout')
        commands = raw['joint_command'][:end]
        require(commands.shape == (end, JOINT_ACTION_WIDTH) and np.isfinite(commands).all()
                and np.allclose(raw['timestamp'][:end], np.arange(end) * CONTROL_DT, rtol=0, atol=1e-8),
                'Invalid source command timeline')
        reference = raw['robot_joint_state'][:end]
        require(reference.shape == (end, JOINT_STATE_WIDTH) and np.isfinite(reference).all(),
                'Invalid recorded joint observations')
    source = dict(kind='bounded_joint_command_prefix_replay_training_only',
                  trajectory=str(trajectory.resolve()),
                  trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                  neural_inference_on_new_cases=False)
    args.output.mkdir(parents=True)
    verification = save_case(args.output.resolve(), source_case, summary['controller'], source,
                             commands, frames, reference=reference)
    outputs = [save_case(args.output.resolve(), case, summary['controller'], source, commands, frames)
               for case in cases]
    report = dict(source_motion=source, source_replay=verification, cases=outputs,
                  task_success_measured=False)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
