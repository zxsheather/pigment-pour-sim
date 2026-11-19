"""Select recoverable approach states from a scoped, completed model collection."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_cylinder_grasp_approach import wrist_pose_breakdown
from audit_gripper import audit
from cylinder_joint_actions import APERTURE_SCALE_M, JOINT_SERVO_INCREMENT, require, robot_joint_observation
from cylinder_progress_prompts import PROGRESS_PROMPTS, PROGRESS_PROTOCOL
from cylinder_tray_vision import WRIST_TARGET_CUE_PROTOCOL, observed_tray_cue_state
from inspect_cylinder import CylinderEnv


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_snapshots(directory, episode, environment):
    case = episode['case']
    env = CylinderEnv(case['volume'], **environment, mass_scale=case.get('mass_scale', 1.),
                      friction_scale=case.get('friction_scale', 1.))
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0.))
        installed = audit(env)
        require(installed['passed'] and installed['custom_pads'] == 0, 'Official finger audit failed')
        initial_base = env.base_pose().copy()
        initial_cylinder = env.data.xpos[env.cylinder_body].copy()
        forward = np.array([np.cos(initial_base[2]), np.sin(initial_base[2])])
        dof = env.model.joint('cylinder_free').dofadr[0]
        snapshots = directory/'recovery_physics.npz'
        observations = directory/'model_observations.npz'
        rows = []
        with (np.load(snapshots, allow_pickle=False) as physics,
              np.load(observations, allow_pickle=False) as observed):
            times = observed['timestamp']
            require(len(times) and np.all(np.diff(times) > 0)
                    and len(times) == len(physics['full_state']) == len(observed['state']),
                    'Snapshot history is incomplete')
            spec = int(physics['state_spec'])
            actuator_ids = physics['hold_actuator_ids'].tolist()
            require(set(actuator_ids) == set(env.hold_targets), 'Saved actuator identities differ')
            for i in np.flatnonzero(observed['instruction'] == PROGRESS_PROMPTS[0]):
                mujoco.mj_setState(env.model, env.data, physics['full_state'][i], spec)
                env.hold_targets = dict(zip(actuator_ids, physics['hold_targets'][i].tolist(), strict=True))
                env.applied_action[:] = physics['applied_base_action'][i]
                mujoco.mj_forward(env.model, env.data)
                state = robot_joint_observation(env, servo_state=True)
                cue = observed_tray_cue_state(env, state, episode['initial_tray_cue'], WRIST_TARGET_CUE_PROTOCOL)
                np.testing.assert_array_equal(cue, observed['state'][i])
                require(env.data.time == times[i], 'Snapshot clock differs from model observation')
                metrics = env.metrics()
                error = env.grip_point_relative() - env.grasp_offset
                pose = wrist_pose_breakdown(env, error, initial_base)
                location = env.data.xpos[env.cylinder_body]
                rows.append(dict(index=int(i), timestamp_s=float(times[i]),
                                 true_grasp_distance_m=float(np.linalg.norm(error)),
                                 current_aperture_m=float(state[6] * APERTURE_SCALE_M),
                                 loaded_fingers=metrics['grip_contacts'], collision=bool(metrics['forbidden_contacts']),
                                 cylinder_shift_m=float(np.linalg.norm(location - initial_cylinder)),
                                 cylinder_z_delta_m=float(location[2] - initial_cylinder[2]),
                                 cylinder_upright=bool(env.data.xmat[env.cylinder_body, 8] > np.cos(.08)),
                                 cylinder_speed_m_s=float(np.linalg.norm(env.data.qvel[dof:dof+3])),
                                 retreat_m=float((initial_base[:2] - env.base_pose()[:2]) @ forward),
                                 **pose))
        return rows, dict(physics=sha256(snapshots), robot_observations=sha256(observations))
    finally:
        env.close()


def choose(rows):
    stable = [r for r in rows if r['timestamp_s'] >= 5. and r['loaded_fingers'] == 0
              and not r['collision'] and r['cylinder_upright'] and r['cylinder_speed_m_s'] < .02
              and r['cylinder_shift_m'] < .009]
    wrist = [r for r in stable if .035 <= r['true_grasp_distance_m'] <= .24
             and r['error_world_m'][1] >= .025 and abs(r['retreat_m']) < .18
             and abs(r['base_yaw_delta_deg']) < 20.]
    base = [r for r in stable if .055 <= r['retreat_m'] <= .32 and abs(r['base_yaw_delta_deg']) < 30.]
    wrist.sort(key=lambda r: (r['true_grasp_distance_m'] + .25 * max(0., .055-r['error_world_m'][1]),
                              r['timestamp_s']))
    base.sort(key=lambda r: (abs(r['retreat_m'] - .10), r['timestamp_s']))
    return dict(clearance_wrist_alignment=wrist, restore_base=base)


def select(collection, cases_path, scope_path):
    scope = read(scope_path)
    cases = read(cases_path)
    summary_path = collection/'summary.json'
    summary = read(summary_path)
    audit_path = Path(scope['checkpoint_audit'])
    weight_audit = read(audit_path)
    seeds = [case['seed'] for case in cases]
    require(scope['training_only'] and scope['independent_cases_read'] is False
            and scope['candidate_output'] is False and scope['closed_loop_model_actuator_outputs'] == 9
            and len(seeds) == len(set(seeds)) and seeds == scope['training_seeds']
            and scope['used_seeds_disjoint_from_prior_training_teacher_reserve_and_known_test_suites']
            and weight_audit['passed'] and scope['sha256'] == weight_audit['checkpoint_sha256']
            and Path(summary['model']).resolve() == Path(scope['checkpoint']).resolve(),
            'Collection does not match its training-only scope and audited model')
    require(summary['evaluation_complete'] and not summary['probe_only']
            and summary['joint_action_representation'] == JOINT_SERVO_INCREMENT
            and summary['progress_protocol'] == PROGRESS_PROTOCOL
            and summary['tray_cue_protocol'] == WRIST_TARGET_CUE_PROTOCOL
            and [e['case'] for e in summary['episodes']] == cases,
            'Expected complete original nine-action rollouts for precisely the training seeds')
    results = []
    for i, case in enumerate(cases):
        directory = collection/f'episode_{i:03d}'
        episode = read(directory/'summary.json')
        require(episode == summary['episodes'][i] and episode['geometry_audit']['passed']
                and episode['geometry_audit']['custom_pads'] == 0 and episode['exact_input_images_recorded'],
                'Episode lacks matched observations or the expected official geometry')
        rows, hashes = observed_snapshots(directory, episode, summary['controller']['environment'])
        eligible = choose(rows)
        results.append(dict(seed=case['seed'], episode=str(directory), snapshot_hashes=hashes,
                            recorded_model_success=episode['success'], approach_snapshots_verified=len(rows),
                            closest=min(rows, key=lambda r: r['true_grasp_distance_m'], default=None),
                            candidates=eligible))
    return dict(kind='training_only_model_approach_recovery_selection',
                source=str(collection), scope_sha256=sha256(scope_path), summary_sha256=sha256(summary_path),
                official_geometry_verified=True, independent_cases_read=False, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('collection', 'cases', 'scope', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    args = parser.parse_args()
    report = select(args.collection.resolve(), args.cases, args.scope)
    with args.output.open('x') as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in report['results']:
        print(row['seed'], {mode: len(eligible) for mode, eligible in row['candidates'].items()}, flush=True)


if __name__ == '__main__':
    main()
