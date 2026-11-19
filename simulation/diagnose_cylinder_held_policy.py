"""Run a training-only held-cylinder diagnostic from certified teacher starts."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import yaml

from carry_env import CONTROL_DT
from cylinder_joint_actions import (
    APERTURE_SCALE_M, JOINT_SERVO_INCREMENT, bounded_joint_command, require,
    with_joint_servo_state,
)
from cylinder_progress_prompts import CylinderProgressPrompts, PROGRESS_PROTOCOL
from cylinder_tray_vision import WRIST_TARGET_CUE_PROTOCOL
from prepare_cylinder_correction import restore_correction
from run_wall_cylinder import CylinderTrial, CylinderWallPolicy


def read(path):
    return json.loads(path.read_text())


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def progress_before_correction(trial, prior, provenance):
    source = read(prior / 'summary.json')
    np.testing.assert_array_equal(trial.progress_initial_joint_command, source['progress_initial_joint_command'])
    snapshot = provenance['snapshot_index']
    with (np.load(prior / 'trajectory.npz', allow_pickle=False) as trace,
          np.load(prior / 'model_observations.npz', allow_pickle=False) as observations):
        time_s = observations['timestamp'][snapshot]
        split = int(np.searchsorted(trace['timestamp'], time_s))
        require(trace['timestamp'][split] == time_s == provenance['source_simulated_seconds'],
                'Source observation and correction start differ')
        targets = np.vstack((trial.progress_initial_joint_command, trace['joint_command'][:split]))
        history = with_joint_servo_state(trace['robot_joint_state'][:split + 1], targets)
        trial.progress_router = CylinderProgressPrompts(trial.initial_base)
        for row in history[:-1]:
            trial.progress_router.observe(row)
        require(trial.progress_router.index == 0
                and trial.progress_router.instruction == observations['instruction'][snapshot],
                'Expected source correction during approach')


def state_digest(trial):
    env = trial.env
    values = np.empty(mujoco.mj_stateSize(env.model, trial.recovery_spec))
    mujoco.mj_getState(env.model, env.data, values, trial.recovery_spec)
    commands = [env.hold_targets[i] for i in trial.hold_actuator_ids]
    combined = np.r_[values, commands, env.applied_action, env.grasp_point_reference]
    return hashlib.sha256(combined.tobytes()).hexdigest()


def start_trial(row, output, lift_m, max_seconds):
    proof = row['certification']
    episode = Path(proof['source'])
    prior = Path(proof['source_model_episode'])
    trace = episode / 'trajectory.npz'
    require(digest(trace) == proof['trajectory_sha256'], 'Teacher trajectory changed')
    require(read(Path(row['proof'])) == proof, 'Stored certification differs')
    case = read(prior / 'summary.json')['case']
    controller = read(prior.parent / 'summary.json')['controller']
    metadata = read(episode / 'metadata.json')
    provenance = metadata['result']['initial_state_provenance']
    require(case['seed'] == proof['seed'] and case['volume'] == proof['volume_ml']
            and row['seed'] == proof['seed'] and proof['teacher_full_task_succeeded']
            and proof['geometry_audit_passed'] and not proof['independent_cases_read']
            and provenance['kind'] == 'expert_recovery_training_only'
            and not provenance['starts_holding'], 'Expected a certified correction in a training-only scene')
    with np.load(trace, allow_pickle=False) as raw, np.load(row['training_arrays'], allow_pickle=False) as data:
        lift = raw['cylinder_position'][:, 2] - raw['cylinder_position'][0, 2]
        candidates = np.flatnonzero((raw['phase'] == 'lift') & (lift >= lift_m) & (raw['contacts'] == 2)
                                   & (data['task_index'] == 1))
        require(len(candidates) > 0, 'No certified two-finger lift frame')
        frame = int(candidates[0])
        expected_state = data['state'][frame]
        actions = data['action'].copy()
        tasks = data['task_index'].copy()
        original_commands = raw['joint_command'][:frame]
    trial = CylinderTrial(case, controller, output, action_space='joint', joint_servo_state=True,
                          servo_increments=True, joint_wrist_cue=True, progress_protocol=PROGRESS_PROTOCOL,
                          tray_cue_protocol=WRIST_TARGET_CUE_PROTOCOL)
    try:
        require(trial.geometry_report['passed'] and trial.geometry_report['custom_pads'] == 0
                and trial.initial_tray_cue == proof['earlier_rgb_estimate'],
                'Geometry or reset visual observation changed')
        progress_before_correction(trial, prior, provenance)
        restore_correction(trial, episode, prior, proof['snapshot'])
        trial.env.data.time = 0.
        trial.progress_changes.clear()
        trial.update_progress()
        max_command_error = 0.
        for i in range(frame):
            require(not trial.done and trial.progress_router.index == tasks[i], 'Teacher prefix diverged')
            trial.step(actions[i])
            executed = bounded_joint_command(trial.env)
            max_command_error = max(max_command_error, float(np.abs(executed - original_commands[i]).max()))
        require(not trial.done and trial.held and trial.env.metrics()['grip_contacts'] == 2
                and trial.progress_router.index == tasks[frame], 'No established two-finger hold at takeover')
        observed = trial.observation()
        max_input_error = float(np.abs(observed['state'] - expected_state).max())
        require(max_command_error <= 1e-6 and max_input_error <= 3e-5,
                'Scene near takeover does not reproduce the certified trajectory')
        require(not trial.good_readings and not trial.task_gates['retreated'],
                'The teacher prefix must end before retreat and reading')
        takeover = dict(case=case, teacher_source=str(episode), teacher_reference_frame=frame,
                        time_s=float(trial.env.data.time), lift_m=trial.max_lift, slip_m=trial.max_slip,
                        task_index=trial.progress_router.index, state_sha256=state_digest(trial),
                        prefix_max_command_error=max_command_error, max_certified_model_state_error=max_input_error,
                        actual_contacts=trial.env.metrics()['grip_contacts'], scoring_hold_established=trial.held)
        trial.max_seconds = takeover['time_s'] + max_seconds
        trial.rows.clear()
        trial.observations.clear()
        trial.recovery_states.clear()
        return trial, takeover, actions[frame:]
    except BaseException:
        trial.env.close()
        raise


def finish(trial, takeover, contact_counts, controlled_by):
    result = trial.finish()
    folder = trial.output
    result.update(probe_kind='held_state_training_only', formal_acceptance_eligible=False)
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return dict(takeover=takeover, result=str(folder / 'summary.json'),
                controlled_by=controlled_by,
                full_reset_neural_task=False, success_after_takeover=result['success'],
                reason=result['failure_reason'], duration_s=result['simulated_seconds'] - takeover['time_s'],
                max_lift_m=result['max_lifted_m'], max_slip_m=result['max_slip_m'],
                contact_counts=contact_counts, final_gates=result['final_task_gates'],
                valid_readings=[x for x in result['reading_history'] if x['valid']])


def run_teacher(row, output, args):
    trial, start, suffix = start_trial(row, output, args.lift_m, args.max_seconds)
    counts = [0, 0, 0]
    try:
        for label in suffix:
            trial.step(label)
            counts[trial.env.metrics()['grip_contacts']] += 1
            if trial.done:
                break
        require(trial.success, 'Certified teacher continuation failed')
        return finish(trial, start, counts, controlled_by='saved_teacher_commands')
    except BaseException:
        trial.env.close()
        raise


def run_policy(row, output, args, policy, teacher_start):
    trial, start, _ = start_trial(row, output, args.lift_m, args.max_seconds)
    counts = [0, 0, 0]
    try:
        require(start == teacher_start, 'Model and teacher warm starts differ')
        initial_opening = 2 * bounded_joint_command(trial.env)[6] / APERTURE_SCALE_M
        torch.manual_seed(47)
        while not trial.done:
            observed = trial.observation()
            clock = time.monotonic()
            chunk, = policy.predict_many([observed])
            trial.latencies.append(time.monotonic() - clock)
            trial.chunks.append(chunk)
            for raw in chunk[:8, :9]:
                action = raw
                if args.hold_opening:
                    action = raw.copy()
                    action[6] = initial_opening
                trial.step(action, raw_model_action=raw)
                counts[trial.env.metrics()['grip_contacts']] += 1
                if trial.done:
                    break
        controlled_by = 'wall_oss_with_fixed_gripper_command' if args.hold_opening else 'wall_oss'
        result = finish(trial, start, counts, controlled_by)
        result['first_raw_model_aperture_m'] = (trial.chunks[0][:8, 6] * APERTURE_SCALE_M).tolist()
        if args.hold_opening:
            result['fixed_gripper_command_m'] = initial_opening * APERTURE_SCALE_M
        return result
    except BaseException:
        trial.env.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True,
                        help='Training-only certified wrist correction report')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--weight-audit', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hold-opening', action='store_true',
                        help='For causal diagnosis, hold the takeover gripper target while Wall runs arm and base.')
    parser.add_argument('--lift-m', type=float, default=.025)
    parser.add_argument('--max-seconds', type=float, default=45.)
    args = parser.parse_args()
    require(.005 <= args.lift_m <= .09 and 30 <= args.max_seconds <= 60, 'Invalid diagnostic horizon')
    require(bool(args.checkpoint) == bool(args.config) == bool(args.weight_audit),
            'Provide checkpoint, weight audit and config together')
    require(not args.hold_opening or args.checkpoint, 'Fixed opening is a model action diagnostic')
    certified = read(args.qualification)
    require(not certified['independent_cases_read'] and not certified['neural_policy_success_measured'],
            'Expected training-only teacher certification')
    rows = [row for row in certified['results'] if row['qualified']]
    require(len(rows) == 2 and len({row['seed'] for row in rows}) == 2, 'Expected two separate certified reserves')
    args.output.mkdir(parents=True)
    teachers = [run_teacher(row, args.output / f"teacher-{row['seed']}", args) for row in rows]
    report = dict(kind='wall_oss_held_state_training_diagnostic', source=str(args.qualification),
                  from_training_only=True, independent_cases_read=False, formal_acceptance_eligible=False,
                  fixed_gripper=args.hold_opening, teachers=teachers, models=[], checkpoint=None)
    if args.checkpoint:
        proof = read(args.weight_audit)
        require(proof['passed'] and proof['changed_action_tensors'] and not proof['changed_frozen_tensors']
                and digest(args.checkpoint / 'model.safetensors') == proof['checkpoint_sha256'],
                'Checkpoint and previously audited weights differ')
        config = yaml.safe_load(args.config.read_text())
        datasets = Path(config['data']['lerobot_config']['root'])
        seen = {x['metadata']['result']['seed'] for x in read(datasets / 'demonstration_provenance.json')}
        require(not seen.intersection(row['seed'] for row in rows),
                'Diagnostic scenes overlap with this model training corpus')
        torch.manual_seed(47)
        policy = CylinderWallPolicy(args.checkpoint, args.config, 'cylinder_joint_v1')
        require(policy.joint_action_representation == JOINT_SERVO_INCREMENT and policy._action_horizon == 8
                and policy.progress_protocol == PROGRESS_PROTOCOL and policy.tray_cue_protocol == WRIST_TARGET_CUE_PROTOCOL,
                'Unexpected model control contract')
        report['checkpoint'] = dict(path=str(args.checkpoint), sha256=proof['checkpoint_sha256'],
                                    load=policy.load_report, audited_action_tensors=len(proof['changed_action_tensors']))
        for row, ref in zip(rows, teachers, strict=True):
            result = run_policy(row, args.output / f"wall-{row['seed']}", args, policy, ref['takeover'])
            report['models'].append(result)
            print(json.dumps(result), flush=True)
    (args.output / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(teacher_successes=len(teachers), model_successes=sum(x['success_after_takeover']
          for x in report['models']), report=str(args.output / 'diagnostic.json'))), flush=True)


if __name__ == '__main__':
    main()
