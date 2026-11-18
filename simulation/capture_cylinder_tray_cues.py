"""Measure initial tray positions on verified robot resets for a training view."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.dataset as arrow

from audit_gripper import audit
from cylinder_joint_actions import JOINT_SERVO_STATE_NAMES, require, robot_joint_observation
from cylinder_tray_vision import capture_initial_tray_cue
from inspect_cylinder import CylinderEnv


def capture(recording, result, expected_robot):
    p = result['controller_parameters']
    physics = result['contact_parameters']
    env = CylinderEnv(result['expected_volume_ml'], image_size=768,
                      grasp_x=p['grasp_x_m'], grasp_height=p['grasp_height_m'],
                      physics_dt=physics['physics_timestep'], solver_impratio=physics['impratio'],
                      friction_scale=result['friction_scale'], mass_scale=result['mass_scale'])
    provenance = result.get('initial_state_provenance')
    try:
        env.reset_inspection(seed=result['seed'], jitter=result['position_jitter_m'],
                             yaw_jitter=result['yaw_jitter_rad'])
        if provenance:
            with np.load(recording/'starting_integration_state.npz', allow_pickle=False) as saved:
                mujoco.mj_setState(env.model, env.data, saved['state'], int(saved['state_spec']))
                ids = saved['hold_actuator_ids'].tolist()
                require(set(ids) == set(env.hold_targets), 'The recorded servo cache differs')
                env.hold_targets = dict(zip(ids, saved['hold_targets'].tolist(), strict=True))
                env.applied_action[:] = saved['applied_base_action']
            mujoco.mj_forward(env.model, env.data)
        geometry = audit(env)
        require(geometry['passed'] and geometry['custom_pads'] == 0, 'Official geometry audit failed')
        np.testing.assert_allclose(robot_joint_observation(env, servo_state=True), expected_robot,
                                   rtol=0, atol=2e-12)
        is_reset = (not provenance or provenance['source_simulated_seconds'] == 0)
        if not is_reset:
            return dict(reset_frame=False, estimate=None, source_state_verified=True)

        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(env.model, spec))
        mujoco.mj_getState(env.model, env.data, before, spec)
        images, estimate = capture_initial_tray_cue(env)
        after = np.empty_like(before)
        mujoco.mj_getState(env.model, env.data, after, spec)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(robot_joint_observation(env, servo_state=True), expected_robot,
                                   rtol=0, atol=2e-12)
        # Truth is consulted after the estimate and is excluded from its inputs.
        actual_xy = env.data.xpos[env.cylinder_body, :2]
        error = None if estimate is None else float(np.linalg.norm(estimate['world_xy_m'] - actual_xy))
        return dict(reset_frame=True, estimate=estimate, error_m=error,
                    initial_base=expected_robot[15:18].tolist(), source_state_verified=True,
                    tray_center_xy_m=env.config['target_tray']['center_xy'],
                    rgb_sha256={name: hashlib.sha256(image.tobytes()).hexdigest()
                                for name, image in images.items()})
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    require(not args.output.exists(), 'Output already exists')
    manifest = dataset/'demonstration_provenance.json'
    provenance = json.loads(manifest.read_text())
    info = json.loads((dataset/'meta/info.json').read_text())
    require(info['features']['observation.state']['names'] == JOINT_SERVO_STATE_NAMES,
            'Expected ordered joint-servo observations')
    frames = arrow.dataset(dataset/'data', format='parquet').to_table(
        columns=['episode_index', 'observation.state'],
        filter=arrow.field('frame_index') == 0).sort_by('episode_index')
    np.testing.assert_array_equal(np.asarray(frames['episode_index']), np.arange(len(provenance)))
    reports = []
    for episode, source in enumerate(provenance):
        recorded = Path(source['source'])
        result = source['metadata']['result']
        observed = np.asarray(frames['observation.state'][episode].as_py(), dtype=np.float32)
        report = capture(recorded, result, observed)
        reports.append(dict(episode=episode, source=str(recorded), seed=result['seed'], **report))
    errors = [row['error_m'] for row in reports if row['estimate'] is not None]
    report = dict(kind='calibrated_reset_rgb_tray_cue_ledger', dataset=str(dataset),
                  provenance_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  episodes=len(reports), reset_episodes=sum(row['reset_frame'] for row in reports),
                  valid_estimates=len(errors), maximum_error_m=max(errors, default=None),
                  future_inputs_used=False, truth_used_for_estimation=False,
                  independent_cases_read=False, episodes_with_estimates=reports)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'episodes_with_estimates'}), flush=True)


if __name__ == '__main__':
    main()
