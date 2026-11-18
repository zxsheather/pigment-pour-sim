"""Measure approach error on saved joint-policy rollouts and teacher traces."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from cylinder_joint_actions import APERTURE_SCALE_M, require, robot_joint_observation
from inspect_cylinder import CylinderEnv


def measure_rollout(folder):
    summary = json.loads((folder / 'summary.json').read_text())
    require(summary['evaluation_complete'] and summary['total'] == 1, 'Expected one completed rollout')
    episode = summary['episodes'][0]
    require(episode['action_space'] == 'joint' and episode['geometry_audit']['passed'],
            'Expected an audited joint rollout')
    case = episode['case']
    env = CylinderEnv(case['volume'], friction_scale=case.get('friction_scale', 1),
                      mass_scale=case.get('mass_scale', 1), **summary['controller']['environment'])
    directory = folder / 'episode_000'
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0))
        start_y = float(env.base_pose()[1])
        snapshots = []
        with (np.load(directory / 'recovery_physics.npz') as physics,
              np.load(directory / 'model_observations.npz') as observations,
              np.load(directory / 'trajectory.npz') as trajectory):
            require(len(physics['full_state']) == len(observations['state']) == len(observations['timestamp']),
                    'Snapshot and observation counts differ')
            spec = int(physics['state_spec'])
            for state, timestamp, observed in zip(physics['full_state'], observations['timestamp'],
                                                  observations['state']):
                mujoco.mj_setState(env.model, env.data, state, spec)
                mujoco.mj_forward(env.model, env.data)
                np.testing.assert_array_equal(robot_joint_observation(env), observed)
                require(abs(env.data.time - timestamp) < 1e-9, 'Snapshot clock mismatch')
                error = env.grip_point_relative() - env.grasp_offset
                snapshots.append(dict(time_s=float(timestamp), grasp_distance_m=float(np.linalg.norm(error)),
                                      grasp_point_error_components_m=error.tolist(),
                                      aperture_m=float(observed[6] * APERTURE_SCALE_M),
                                      base_retreat_m=start_y - float(env.base_pose()[1])))
            maximum_contacts = int(trajectory['contacts'].max(initial=0))
        require(bool(snapshots), 'No rollout snapshots')
        return dict(rollout=str(folder.resolve()), snapshots=len(snapshots),
                    robot_observations_reproduced_exactly=True, maximum_recorded_loaded_contacts=maximum_contacts,
                    nearest=min(snapshots, key=lambda row: row['grasp_distance_m']), first=snapshots[0],
                    last=snapshots[-1], failure_reason=episode['failure_reason'])
    finally:
        env.close()


def close_references(dataset, volume):
    provenance = json.loads((dataset / 'demonstration_provenance.json').read_text())
    references = []
    for demo in provenance:
        result = demo['metadata']['result']
        if result['expected_volume_ml'] != volume or result.get('initial_state_provenance'):
            continue
        with np.load(Path(demo['source']) / 'trajectory.npz') as raw:
            offset = [result['controller_parameters']['grasp_x_m'], 0, 0]
            positions = raw['grip_point_relative'][raw['phase'] == 'close']
            require(len(positions) > 0, 'Full teacher demonstration has no close phase')
            distances = np.linalg.norm(positions - offset, axis=1)
        references.append(dict(seed=result['seed'], episode=demo['source'],
                               close_distance_m_min=float(distances.min()),
                               close_distance_m_max=float(distances.max()),
                               initial_close_distance_m=float(distances[0])))
    require(bool(references), 'No full training demonstration for the reference volume')
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', nargs='+', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference-volume', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(kind='recorded_joint_policy_geometric_approach_diagnostic',
                  metric='Distance from cylinder grasp point to the intended offset in the gripper frame',
                  reference='Recorded close phase of full training demonstrations, for descriptive comparison',
                  teacher_references=close_references(args.dataset, args.reference_volume),
                  policy_results=[measure_rollout(folder) for folder in args.rollouts],
                  new_policy_inference_run=False, independent_cases_read=False)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
