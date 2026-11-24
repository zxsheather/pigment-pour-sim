"""Replay recorded commands to check the joint decoder and Cartesian regression.

No neural model is loaded. Demonstration replay successes are decoder checks,
never learned-policy scores or independent-suite results.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import pyarrow.parquet as pq

from cylinder_joint_actions import (
    JOINT_ACTION_WIDTH, JOINT_STATE_WIDTH, JOINT_SERVO_STATE_WIDTH, absolute_joint_chunk,
    relative_joint_chunks, robot_joint_observation, require,
)
from run_wall_cylinder import CylinderTrial


def cartesian_regression(episode, output):
    source = json.loads((episode.parent/'summary.json').read_text())
    case = source['episodes'][0]['case']
    trial = CylinderTrial(case, source['controller'], output, with_odometry=True)
    maximum = 0.
    try:
        with np.load(episode/'trajectory.npz') as data:
            for state, action in zip(data['state'], data['action']):
                maximum = max(maximum, float(np.abs(trial.env.state_vector()-state).max()))
                trial.step(action)
                require(not trial.done, 'Cartesian replay ended before the recorded IK-failure boundary')
            frames = len(data['state'])
        report=dict(kind='cartesian_recorded_prefix_regression', source=str(episode),
                    frames=frames, robot_state23_max_difference=maximum, passed=maximum == 0.,
                    original_next_unrecorded_failure_not_reexecuted=True)
        (output/'replay-check.json').write_text(json.dumps(report, indent=2)+'\n')
        require(maximum == 0., f'Default Cartesian physics/state replay changed: {maximum}')
        return report
    finally:
        trial.env.close()


def joint_replay(dataset, episode, output):
    provenance = json.loads((dataset/'demonstration_provenance.json').read_text())
    demo = provenance[episode]
    pieces = [pq.read_table(p) for p in sorted((dataset/'data').rglob('*.parquet'))]
    states, actions = [], []
    for table in pieces:
        ids = np.asarray(table['episode_index']);selected = np.flatnonzero(ids == episode)
        if len(selected):
            order = selected[np.argsort(np.asarray(table['frame_index'])[selected])]
            states.extend(np.asarray(table['observation.state'].to_pylist())[order])
            actions.extend(np.asarray(table['action'].to_pylist())[order])
    states = np.asarray(states, np.float32);actions = np.asarray(actions, np.float32)
    require(len(states) == demo['metadata']['frames'], 'Frame count mismatch')
    return replay_joint_labels(Path(demo['source']), demo['metadata']['result'], states, actions,
                               output, episode=episode)


def joint_replay_raw(source, reconstruction, validation_file, output):
    from export_carry_dataset import file_digest, verify_joint_reconstruction

    source = source.resolve()
    validation = json.loads(validation_file.read_text())
    require(validation['data_integrity_passed'] and validation['normalization_covers_all_labels']
            and validation['action_space'] == 'joint' and Path(validation['episode']) == source
            and validation['trajectory_sha256'] == file_digest(source / 'trajectory.npz'),
            'Expected a verified raw joint recovery')
    metadata = json.loads((source / 'metadata.json').read_text())
    require(metadata['frames'] == validation['frames'], 'Validated raw frame count differs')
    result = metadata['result']
    require(result == validation['teacher_result'], 'Teacher result changed after validation')
    require(result['success'] and result['grip_profile'] == 'official' and result['geometry_audit_passed'],
            'Expected a successful official teacher demonstration')
    with np.load(source / 'trajectory.npz', allow_pickle=False) as raw:
        states, actions = verify_joint_reconstruction(source, reconstruction, raw, metadata['frames'])
    return replay_joint_labels(source, result, states, actions, output)


def replay_joint_labels(raw_source, result, states, actions, output, *, episode=None):
    require(result['grip_force_target_n'] is None, 'Replay requires position servos')
    require(states.shape in ((len(actions), JOINT_STATE_WIDTH), (len(actions), JOINT_SERVO_STATE_WIDTH))
            and actions.shape == (len(states), JOINT_ACTION_WIDTH)
            and len(states) > 0, 'Invalid joint replay arrays')
    controller = dict(grasp_x=result['controller_parameters']['grasp_x_m'],
        grasp_height=result['controller_parameters']['grasp_height_m'],
        solver_impratio=result['contact_parameters']['impratio'],
        physics_dt=result['contact_parameters']['physics_timestep'])
    case = dict(volume=result['expected_volume_ml'], seed=result['seed'],
        jitter=result['position_jitter_m'], yaw_jitter=result['yaw_jitter_rad'],
        friction_scale=result['friction_scale'], mass_scale=result['mass_scale'])
    trial = CylinderTrial(case, {'environment': controller}, output, action_space='joint')
    env = trial.env;maximum_position=maximum_velocity=0.;decoded_difference=0.
    try:
        require(env.model.opt.timestep == controller['physics_dt']
                and env.model.opt.impratio == controller['solver_impratio']
                and env.grasp_offset[0] == controller['grasp_x'], 'Replay environment configuration mismatch')
        recovery = bool(result.get('initial_state_provenance'))
        if recovery:
            with np.load(raw_source/'starting_integration_state.npz') as saved:
                mujoco.mj_setState(env.model, env.data, saved['state'], int(saved['state_spec']))
                env.hold_targets = dict(zip(saved['hold_actuator_ids'].tolist(), saved['hold_targets'].tolist()))
                env.applied_action[:] = saved['applied_base_action']
            mujoco.mj_forward(env.model, env.data);env.data.time=0.
        chunks = relative_joint_chunks(states, actions, 32).astype(np.float32)
        for anchor in range(0, len(actions), 8):
            decoded = absolute_joint_chunk(chunks[anchor], states[anchor])
            length = min(8, len(actions)-anchor)
            decoded_difference = max(decoded_difference, float(np.abs(decoded[:length]-actions[anchor:anchor+length]).max()))
            for offset, command in enumerate(decoded[:length]):
                index = anchor+offset;observed = robot_joint_observation(env)
                maximum_position = max(maximum_position, float(abs(observed[:6]-states[index, :6]).max()))
                maximum_velocity = max(maximum_velocity, float(abs(observed[9:15]-states[index, 9:15]).max()))
                trial.step(command)
                if trial.done:
                    break
            if trial.done:
                break
        extra = 0
        while not trial.done and extra < 20:
            trial.step(actions[-1]);extra += 1
        report = dict(kind='recorded_teacher_joint_decoder_replay', episode=episode,
            source=str(raw_source), demonstration_frames=len(actions), executed_frames=len(trial.rows),
            recovery_integration_state_restored=recovery, terminal_hold_seconds=extra*.05,
            relative_decode_max_difference=decoded_difference,
            relative_label_dtype=str(chunks.dtype),
            robot_joint_position_max_difference_rad=maximum_position,
            robot_joint_velocity_max_difference_rad_s=maximum_velocity,
            demonstration_replay_success=trial.success, failure_reason=trial.reason,
            max_slip_m=trial.max_slip, max_lifted_m=trial.max_lift,
            final_task_gates=trial.task_gates, reading_history=trial.readings,
            joint_limit_projected_frames=trial.joint_limit_projected_frames,
            geometry_audit_passed=trial.geometry_report['passed'],
            controller_environment_parameters=controller,
            actual_physics_timestep_s=env.model.opt.timestep,
            actual_solver_impratio=env.model.opt.impratio,
            neural_weights_loaded=False, learned_task_success_measured=False)
        (output/'decoder-replay.json').write_text(json.dumps(report, indent=2)+'\n')
        require(decoded_difference < 2e-7, 'Joint label decoding error')
        require(trial.joint_limit_projected_frames == 0, 'Recorded labels need joint-limit projection')
        require(trial.success, f'Teacher joint-label replay failed: {trial.reason}')
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--raw-episode', type=Path)
    parser.add_argument('--cartesian-episode', type=Path)
    parser.add_argument('--joint-episode', type=int, action='append', default=[])
    parser.add_argument('--joint-reconstruction', type=Path)
    parser.add_argument('--raw-validation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();require(not args.output.exists(), 'Output exists')
    if args.raw_episode is not None:
        if not args.joint_reconstruction or not args.raw_validation or args.cartesian_episode or args.joint_episode:
            parser.error('A raw replay requires its joint reconstruction and raw validation')
        report = joint_replay_raw(args.raw_episode, args.joint_reconstruction, args.raw_validation, args.output)
        print(json.dumps(report), flush=True)
        return
    if not args.cartesian_episode or args.joint_reconstruction or args.raw_validation:
        parser.error('The dataset path requires the Cartesian regression episode')
    args.output.mkdir(parents=True);started=time.time()
    regression = cartesian_regression(args.cartesian_episode, args.output/'cartesian')
    (args.output/'cartesian-regression.json').write_text(json.dumps(regression, indent=2)+'\n')
    reports=[]
    for episode in args.joint_episode or [0, 51]:
        report = joint_replay(args.dataset, episode, args.output/f'joint_{episode:03d}')
        reports.append(report)
        (args.output/f'joint-{episode}.json').write_text(json.dumps(report, indent=2)+'\n')
    result=dict(kind='joint_execution_and_cartesian_regression', passed=True,
        cartesian_regression=regression, joint_demonstration_replays=reports,
        elapsed_seconds=time.time()-started, neural_weights_loaded=False,
        learned_policy_or_independent_suite_success=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
