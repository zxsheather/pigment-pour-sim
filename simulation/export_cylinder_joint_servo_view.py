"""Append robot-command history to a verified joint dataset and share its videos."""
import argparse
import json
import os
from pathlib import Path
import shutil

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow
import pyarrow.parquet as pq

from cylinder_joint_actions import (
    JOINT_SERVO_STATE_NAMES, JOINT_SERVO_STATE_WIDTH, JOINT_STATE_WIDTH,
    bounded_joint_command, joint_command_action, require, robot_joint_observation,
    with_joint_servo_state,
)
from export_cylinder_joint_dataset import digest, replace_vector, statistics
from inspect_cylinder import CylinderEnv


def initial_command(source, result, expected_state):
    controller = result['controller_parameters']
    physics = result['contact_parameters']
    env = CylinderEnv(result['expected_volume_ml'], grasp_x=controller['grasp_x_m'],
                      grasp_height=controller['grasp_height_m'], solver_impratio=physics['impratio'],
                      physics_dt=physics['physics_timestep'], mass_scale=result['mass_scale'],
                      friction_scale=result['friction_scale'])
    try:
        env.reset_inspection(seed=result['seed'], jitter=result['position_jitter_m'],
                             yaw_jitter=result['yaw_jitter_rad'])
        if result.get('initial_state_provenance'):
            with np.load(source/'starting_integration_state.npz', allow_pickle=False) as saved:
                mujoco.mj_setState(env.model, env.data, saved['state'], int(saved['state_spec']))
                ids = saved['hold_actuator_ids'].tolist()
                require(set(ids) == set(env.hold_targets), 'Expected the archived servo cache')
                env.hold_targets = dict(zip(ids, saved['hold_targets'].tolist(), strict=True))
                env.applied_action[:] = saved['applied_base_action']
            mujoco.mj_forward(env.model, env.data)
        observed = robot_joint_observation(env)
        # Settling a historical reset can change near-zero values by rounding.
        np.testing.assert_allclose(observed, expected_state, rtol=0., atol=2e-12)
        return bounded_joint_command(env), float(np.abs(observed - expected_state).max())
    finally:
        env.close()


def norm_stats(output, base, states):
    norm = json.loads((base/'joint_norm_stats.json').read_text())
    tail = states[:, JOINT_STATE_WIDTH:]
    extra = statistics(tail)
    radius = np.maximum(np.abs(tail).max(0) * 1.2, [.01]*6 + [.05, .12])
    extra.update(q01=(-radius).tolist(), q99=radius.tolist())
    require(np.all(tail >= -radius) and np.all(tail <= radius), 'Servo history clipping in training')
    for key, values in norm['norm_stats']['observation.state'].items():
        if key != 'count':
            require(len(values) == JOINT_STATE_WIDTH, 'Expected legacy state normalization')
            values.extend(extra[key])
    norm['provenance']['servo_state_bounds'] = dict(
        radius=radius.tolist(), frames=len(states), earlier_affine_bounds_unchanged=True,
        features='Previous bounded actuator targets minus current measured arm position and aperture; previous applied base forward command')
    (output/'joint_norm_stats.json').write_text(json.dumps(norm, indent=2)+'\n')
    return radius.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base_dataset.resolve(), args.output.resolve()
    require(not output.exists(), 'Output exists')
    info = json.loads((base/'meta/info.json').read_text())
    provenance_file = base/'demonstration_provenance.json'
    provenance = json.loads(provenance_file.read_text())
    feature = 'observation.state'
    require(info['features'][feature]['shape'] == [JOINT_STATE_WIDTH]
            and len(provenance) == info['total_episodes'], 'Expected the verified legacy joint dataset')
    dataset = arrow.dataset(base/'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'frame_index', feature, 'action']).sort_by('index')
    indices = np.asarray(dataset['index'])
    episode_ids = np.asarray(dataset['episode_index'])
    frame_ids = np.asarray(dataset['frame_index'])
    observed = np.asarray(dataset[feature].to_pylist(), dtype=np.float32)
    actions = np.asarray(dataset['action'].to_pylist(), dtype=np.float32)
    require(np.array_equal(indices, np.arange(info['total_frames'])), 'Expected contiguous frame indices')
    states = np.empty((len(indices), JOINT_SERVO_STATE_WIDTH), dtype=np.float32)
    episode_stats, source_rows = [], []
    for index, demo in enumerate(provenance):
        source = Path(demo['source'])
        path = source/'trajectory.npz'
        require(digest(path) == demo['trajectory_sha256'], 'Source trajectory changed')
        metadata = json.loads((source/'metadata.json').read_text())
        result = metadata['result']
        require(metadata == demo['metadata'] and result['success'] and result['geometry_audit_passed']
                and result['grip_profile'] == 'official' and result['grip_force_target_n'] is None,
                'Expected a successful recorded position-servo demonstration')
        select = episode_ids == index
        require(np.array_equal(frame_ids[select], np.arange(metadata['frames'])), 'Frame alignment changed')
        with np.load(path, allow_pickle=False) as raw:
            commands = raw['joint_command']
            np.testing.assert_array_equal(joint_command_action(commands), actions[select])
        state = observed[select]
        initial, initial_error = initial_command(source, result, state[0])
        historical = np.vstack((initial, commands[:-1]))
        expanded = with_joint_servo_state(state, historical)
        states[select] = expanded
        episode_stats.append(statistics(expanded))
        source_rows.append(dict(episode=index, frames=len(state), sha256=demo['trajectory_sha256'],
                                initial_servo_command_reconstructed=True, initial_state_max_difference=initial_error))
    np.testing.assert_array_equal(states[:, :JOINT_STATE_WIDTH], observed)

    output.mkdir()
    shutil.copytree(base/'meta', output/'meta')
    shutil.copytree(base/'videos', output/'videos', copy_function=os.link)
    videos = list((base/'videos').rglob('*.mp4'))
    require(videos and all(os.path.samefile(path, output/path.relative_to(base)) for path in videos),
            'Training videos are not shared')
    for path in sorted((base/'data').rglob('*.parquet')):
        table = pq.read_table(path)
        view = replace_vector(table, feature, states[np.asarray(table['index'])])
        require(all(view[name].equals(table[name]) for name in table.column_names if name != feature),
                'A non-state data column changed')
        dest = output/path.relative_to(base)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(view, dest)
    for path in sorted((base/'meta/episodes').rglob('*.parquet')):
        table = pq.read_table(path)
        ids = np.asarray(table['episode_index'])
        for key in episode_stats[0]:
            name = f'stats/{feature}/{key}'
            field = table.schema.field(name)
            values = pa.array([episode_stats[i][key] for i in ids], type=field.type)
            table = table.set_column(table.schema.get_field_index(name), field, values)
        pq.write_table(table, output/path.relative_to(base))
    info['features'][feature].update(shape=[JOINT_SERVO_STATE_WIDTH], names=JOINT_SERVO_STATE_NAMES)
    (output/'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
    stats = json.loads((base/'meta/stats.json').read_text())
    stats[feature] = statistics(states)
    (output/'meta/stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    shutil.copyfile(provenance_file, output/'demonstration_provenance.json')
    radius = norm_stats(output, base, states)
    report = dict(kind='joint_servo_observation_view', source=str(base), output=str(output),
                  episodes=len(provenance), frames=len(indices), legacy_state_and_actions_exact=True,
                  shared_video_files=len(videos), shared_video_bytes=sum(p.stat().st_size for p in videos),
                  added_state_dimensions=JOINT_SERVO_STATE_NAMES[JOINT_STATE_WIDTH:],
                  bounds_radius=radius, episodes_audited=source_rows,
                  initial_state_comparison_atol=2e-12,
                  uses_only_earlier_commands=True, independent_cases_read=False)
    (output/'servo_view_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes_audited'}), flush=True)


if __name__ == '__main__':
    main()
