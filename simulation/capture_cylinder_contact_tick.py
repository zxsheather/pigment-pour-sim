"""Recover an executed single-finger state between two saved neural observations."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from audit_gripper import audit
from cylinder_joint_actions import (
    bounded_joint_command, require, robot_joint_observation, with_joint_servo_state,
)
from cylinder_progress_prompts import CylinderProgressPrompts, PROGRESS_PROMPTS, PROGRESS_PROTOCOL
from cylinder_tray_vision import WRIST_TARGET_CUE_PROTOCOL, observed_tray_cue_state
from inspect_cylinder import CylinderEnv
from select_cylinder_approach_recovery import read, sha256


def replay_tick(env, episode, trace, observed, physics, frame):
    commands = trace['joint_command']
    times = trace['timestamp']
    require(0 < frame < len(times) and trace['contacts'][frame] == 1
            and not (trace['contacts'][:frame] == 2).any(), 'Expected an executed pregrasp single-finger frame')
    observations = np.searchsorted(times, observed['timestamp'])
    np.testing.assert_array_equal(times[observations], observed['timestamp'])
    following = int(np.searchsorted(observations, frame))
    require(0 < following < len(observations) and observations[following] != frame,
            'Choose a tick strictly between two saved model observations')
    previous = following - 1
    start, end = int(observations[previous]), int(observations[following])
    require(np.all(observed['instruction'][previous:following+1] == PROGRESS_PROMPTS[0]),
            'Expected an approach observation on both sides')
    initial_command = bounded_joint_command(env)
    np.testing.assert_array_equal(initial_command, episode['progress_initial_joint_command'])
    prefix = with_joint_servo_state(trace['robot_joint_state'][:frame+1],
                                   np.vstack((initial_command, commands[:frame])))
    router = CylinderProgressPrompts(env.base_pose())
    require(all(router.observe(state) == 0 for state in prefix), 'The capture is outside approach')

    ids = physics['hold_actuator_ids'].tolist()
    spec = int(physics['state_spec'])
    mujoco.mj_setState(env.model, env.data, physics['full_state'][previous], spec)
    env.hold_targets = dict(zip(ids, physics['hold_targets'][previous].tolist(), strict=True))
    env.applied_action[:] = physics['applied_base_action'][previous]
    mujoco.mj_forward(env.model, env.data)
    values = np.empty(mujoco.mj_stateSize(env.model, spec))
    for i in range(start, end+1):
        np.testing.assert_array_equal(robot_joint_observation(env), trace['robot_joint_state'][i])
        metrics = env.metrics()
        require(env.data.time == times[i] and metrics['grip_contacts'] == trace['contacts'][i],
                'Executed clock or grip contact differed from the replay')
        if i == frame:
            require(not metrics['forbidden_contacts'], 'Unexpected collision at the capture tick')
            mujoco.mj_getState(env.model, env.data, values, spec)
            archived = dict(state_spec=spec, full_state=[values.copy()], hold_actuator_ids=ids,
                            hold_targets=[[env.hold_targets[j] for j in ids]],
                            applied_base_action=[env.applied_action.copy()])
            teacher_state = observed_tray_cue_state(
                env, robot_joint_observation(env, servo_state=True),
                episode['initial_tray_cue'], WRIST_TARGET_CUE_PROTOCOL)
        if i == end:
            mujoco.mj_getState(env.model, env.data, values, spec)
            np.testing.assert_array_equal(values, physics['full_state'][following])
            np.testing.assert_array_equal([env.hold_targets[j] for j in ids], physics['hold_targets'][following])
            np.testing.assert_array_equal(env.applied_action, physics['applied_base_action'][following])
            break
        command = commands[i]
        env.step_joints(command[:6], opening=float(command[6]), forward=float(command[7]), yaw=float(command[8]))
        np.testing.assert_array_equal(bounded_joint_command(env), command)
    inputs = dict(timestamp=[float(times[frame])], state=[teacher_state], instruction=[PROGRESS_PROMPTS[0]])
    return archived, inputs, (start, end)


def capture(selection_path, seed, frame, output):
    require(not output.exists(), 'Use a new capture directory')
    selection = read(selection_path)
    require(selection['kind'] == 'training_only_model_approach_recovery_selection'
            and selection['official_geometry_verified'] and selection['independent_cases_read'] is False,
            'Expected a verified training-only selection')
    row, = (row for row in selection['results'] if row['seed'] == seed)
    directory = Path(row['episode'])
    collection = Path(selection['source'])
    aggregate_path = collection/'summary.json'
    require(directory.parent == collection and sha256(aggregate_path) == selection['summary_sha256'],
            'The model collection changed')
    aggregate = read(aggregate_path)
    episode = read(directory/'summary.json')
    case = episode['case']
    require(case['seed'] == seed and episode in aggregate['episodes']
            and episode['progress_protocol'] == PROGRESS_PROTOCOL
            and episode['tray_cue_protocol'] == WRIST_TARGET_CUE_PROTOCOL,
            'Capture does not match the recorded training episode')
    paths = dict(applied_trajectory=directory/'trajectory.npz',
                 robot_observations=directory/'model_observations.npz',
                 physics=directory/'recovery_physics.npz')
    require(all(sha256(p) == row['snapshot_hashes'][k] for k,p in paths.items()), 'The source archives changed')
    env = CylinderEnv(case['volume'], **aggregate['controller']['environment'],
                      mass_scale=case.get('mass_scale', 1.), friction_scale=case.get('friction_scale', 1.))
    try:
        env.reset_inspection(seed=seed, jitter=case.get('jitter', .004), yaw_jitter=case.get('yaw_jitter', 0.))
        geometry = audit(env)
        require(geometry['passed'] and geometry['custom_pads'] == 0, 'Official finger audit failed')
        with (np.load(paths['applied_trajectory'], allow_pickle=False) as trace,
              np.load(paths['robot_observations'], allow_pickle=False) as observed,
              np.load(paths['physics'], allow_pickle=False) as physics):
            archived, inputs, (start, end) = replay_tick(env, episode, trace, observed, physics, frame)
    finally:
        env.close()

    proof = dict(kind='bounded_joint_command_prefix_replay_training_only',
                 original_training_episode=str(directory), actual_executed_training_tick=frame,
                 previous_saved_model_observation_tick=start, next_saved_model_observation_tick=end,
                 complete_next_integration_state_exact=True, source_model_requested_observation_at_tick=False,
                 trajectory_sha256=row['snapshot_hashes']['applied_trajectory'])
    episode.update(source_motion=proof, exact_input_images_recorded=False)
    aggregate.update(episodes=[episode], total=1, successes=int(episode['success']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temp:
        staging = Path(temp)/'capture'
        dst = staging/'episode_000'
        dst.mkdir(parents=True)
        np.savez_compressed(dst/'recovery_physics.npz', **archived)
        np.savez_compressed(dst/'model_observations.npz', **inputs)
        for filename in ['trajectory.npz', *(f'initial_tray_{name}.png' for name in episode['initial_tray_images'])]:
            os.link(directory/filename, dst/filename)
        (dst/'summary.json').write_text(json.dumps(episode, indent=2)+'\n')
        (staging/'summary.json').write_text(json.dumps(aggregate, indent=2)+'\n')
        (staging/'bracketed-frame-proof.json').write_text(json.dumps(proof, indent=2)+'\n')
        staging.rename(output)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--frame', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.selection, args.seed, args.frame, args.output.resolve())))


if __name__ == '__main__':
    main()
