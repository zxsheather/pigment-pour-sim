"""Derive joint labels from verified replays, sharing immutable RGB videos.

Only state/action vectors, their metadata/statistics and robot type change.
Episode order, task text, sampling weights, timestamps and video references
stay identical. This prepares data; it does not train or score a policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from audit_gripper import audit
from cylinder_joint_actions import (JOINT_ACTION_NAMES, JOINT_STATE_NAMES,
                                    joint_command_action, joint_observation,
                                    relative_joint_chunks, require)
from inspect_cylinder import CylinderEnv


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def statistics(array):
    array = np.asarray(array, dtype=np.float64)
    result = dict(min=array.min(0).tolist(), max=array.max(0).tolist(),
                  mean=array.mean(0).tolist(), std=array.std(0).tolist(),
                  count=[len(array)])
    for quantile in [.01, .10, .50, .90, .99]:
        result[f'q{round(quantile*100):02d}'] = np.quantile(array, quantile, axis=0).tolist()
    return result


def replace_vector(table, name, array):
    width = array.shape[1]
    values = pa.FixedSizeListArray.from_arrays(pa.array(array.reshape(-1), type=pa.float32()), width)
    table = table.set_column(table.schema.get_field_index(name),
                             pa.field(name, values.type), values)
    metadata = dict(table.schema.metadata or {})
    if b'huggingface' in metadata:
        hf = json.loads(metadata[b'huggingface'])
        feature = hf['info']['features'][name]
        require(feature['_type'] in ('List', 'Sequence') and 'length' in feature,
                'Unsupported Hugging Face vector metadata')
        feature['length'] = width
        metadata[b'huggingface'] = json.dumps(hf).encode()
        table = table.replace_schema_metadata(metadata)
    return table


def write_norm_stats(output, states, actions, limits, horizon):
    relative = np.concatenate([relative_joint_chunks(s, a, horizon).reshape(-1, 9)
                               for s, a in zip(states, actions)])
    state = np.concatenate(states).astype(np.float64)
    action_stats, state_stats = statistics(relative), statistics(state)
    radius = np.maximum(np.abs(relative[:, :6]).max(0) * 1.05, .05)
    action_low = np.r_[-radius, 0., -.12, -.3]
    action_high = np.r_[radius, 1., .12, .3]
    state_low, state_high = state.min(0), state.max(0)
    state_low[:6] = np.minimum(limits[:, 0], state_low[:6]) - .02
    state_high[:6] = np.maximum(limits[:, 1], state_high[:6]) + .02
    state_low[6:9], state_high[6:9] = [0., -.12, -.3], [1., .12, .3]
    speed_radius = np.maximum(np.abs(state[:, 9:15]).max(0) * 1.1, .65)
    state_low[9:15], state_high[9:15] = -speed_radius, speed_radius
    center = (state_low[15:18] + state_high[15:18]) / 2
    width = np.maximum(state_high[15:18] - state_low[15:18] + [.10, .10, .20], .5)
    state_low[15:18], state_high[15:18] = center - width/2, center + width/2
    require(np.all(relative >= action_low - 1e-7) and np.all(relative <= action_high + 1e-7),
            'Training action outside declared bounds')
    require(np.all(state >= state_low - 1e-7) and np.all(state <= state_high + 1e-7),
            'Training state outside declared bounds')
    action_stats.update(q01=action_low.tolist(), q99=action_high.tolist())
    state_stats.update(q01=state_low.tolist(), q99=state_high.tolist())
    report = dict(norm_stats={'action': action_stats, 'observation.state': state_stats},
                  provenance=dict(horizon=horizon, episodes=len(states), frames=len(state),
                      cross_episode_pairs=0, end_padding='Repeat last action inside each episode',
                      relative_conversion='Subtract anchor robot joint positions from first six targets only',
                      action_bounds='q01/q99 slots are declared bounds: symmetric1.05*maximum absolute training H32 joint delta, radius at least0.05rad; gripper[0,1], base vx+/-0.12m/s, yaw+/-0.3rad/s',
                      state_bounds='q01/q99 slots are declared bounds: official joint ranges and observed extrema plus0.02rad; gripper[0,1], bounded base velocities; measured joint speeds with10% margin and radius at least0.65rad/s; robot odometry margins',
                      empirical_quantile_fields_overridden_and_disclosed=True,
                      no_training_label_clipped=True, evaluation_cases_read=False))
    (output/'joint_norm_stats.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--joint-reconstruction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=32, choices=range(1, 33))
    args = parser.parse_args()
    base, reconstruction, output = (p.resolve() for p in
                                    (args.base_dataset, args.joint_reconstruction, args.output))
    require(not output.exists(), 'Output exists')
    provenance = json.loads((base/'demonstration_provenance.json').read_text())
    replay = json.loads((reconstruction/'summary.json').read_text())
    info = json.loads((base/'meta/info.json').read_text())
    require(replay['complete'] and replay['passed'] and replay['dataset'] == str(base),
            'Expected completed reconstruction of this exact dataset')
    require(len(provenance) == info['total_episodes'] == len(replay['episodes']),
            'Episode count mismatch')
    original_hashes = {str(p.relative_to(base)): digest(p)
                       for p in base.rglob('*') if p.is_file()}
    states, actions, old_states, old_actions, episode_stats, sources = [], [], [], [], [], []
    for index, (demo, row) in enumerate(zip(provenance, replay['episodes'])):
        source = Path(demo['source'])
        require(row['episode_index'] == index and row['passed'] and row['source'] == str(source),
                'Reconstruction episode order or validity mismatch')
        report_path = Path(row['summary'])
        report = json.loads(report_path.read_text())
        trajectory_path = source/'trajectory.npz'
        require(digest(trajectory_path) == report['trajectory_sha256'] == demo['trajectory_sha256'],
                f'Changed source trajectory: {source}')
        require(json.loads((source/'metadata.json').read_text()) == demo['metadata'],
                f'Changed source metadata: {source}')
        joint_path = report_path.parent/'robot_joint_observations.npz'
        with np.load(joint_path) as joints, np.load(trajectory_path) as raw:
            state = joint_observation(joints['right_qpos'], joints['right_qvel'],
                                      raw['state'], raw['base_pose'])
            action = joint_command_action(raw['joint_command'])
            old_state = np.concatenate([raw['state'], raw['base_pose']], axis=1).astype(np.float32)
            old_action = raw['action'].astype(np.float32)
            require(np.allclose(raw['timestamp'], np.arange(len(state))/20), 'Invalid source times')
        require(len(state) == demo['metadata']['frames'] == row['frames_reconstructed'],
                'Reconstructed frame count mismatch')
        states.append(state); actions.append(action)
        old_states.append(old_state); old_actions.append(old_action)
        episode_stats.append({'observation.state': statistics(state), 'action': statistics(action)})
        sources.append(dict(episode_index=index, source=str(source), frames=len(state),
                            trajectory_sha256=demo['trajectory_sha256'],
                            robot_joint_observations_sha256=digest(joint_path)))
    require(sum(map(len, states)) == info['total_frames'], 'Frame total mismatch')
    output.mkdir(parents=True)
    shutil.copytree(base/'meta', output/'meta')
    shared_videos = []
    for source in sorted((base/'videos').rglob('*')):
        if not source.is_file():
            continue
        target = output/source.relative_to(base); target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        require(os.path.samefile(source, target), 'Video sharing failed')
        shared_videos.append(dict(path=str(source.relative_to(base)), bytes=source.stat().st_size))
    offsets = np.r_[0, np.cumsum([len(s) for s in states])]
    seen = np.zeros(info['total_frames'], dtype=bool)
    for source in sorted((base/'data').rglob('*.parquet')):
        table = pq.read_table(source)
        episode_ids = np.asarray(table['episode_index']); frames = np.asarray(table['frame_index'])
        existing_state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        existing_action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        replacement_state = np.empty((len(table), 18), np.float32)
        replacement_action = np.empty((len(table), 9), np.float32)
        for episode in np.unique(episode_ids):
            episode = int(episode); selected = np.flatnonzero(episode_ids == episode); frame = frames[selected]
            require(0 <= episode < len(states) and np.all(frame >= 0)
                    and np.all(frame < len(states[episode])), 'Invalid frame mapping')
            require(np.array_equal(existing_state[selected],
                                   old_states[episode][frame])
                    and np.array_equal(existing_action[selected],
                                       old_actions[episode][frame]), 'Original export/source mismatch')
            global_indices = offsets[episode] + frame
            require(len(np.unique(global_indices)) == len(frame) and not seen[global_indices].any(),
                    'Duplicate frame mapping')
            seen[global_indices] = True
            replacement_state[selected], replacement_action[selected] = states[episode][frame], actions[episode][frame]
        result = replace_vector(replace_vector(table, 'observation.state', replacement_state),
                                 'action', replacement_action)
        for name in table.column_names:
            if name not in ('observation.state', 'action'):
                require(result[name].equals(table[name]), f'Changed non-vector column: {name}')
        target = output/source.relative_to(base); target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(result, target)
    require(seen.all(), 'Missing source frames')
    for source in sorted((base/'meta/episodes').rglob('*.parquet')):
        table = pq.read_table(source); ids = np.asarray(table['episode_index'])
        for feature in ('observation.state', 'action'):
            for name in episode_stats[0][feature]:
                column = f'stats/{feature}/{name}'
                require(column in table.column_names, f'Missing episode statistic: {column}')
                values = pa.array([episode_stats[int(i)][feature][name] for i in ids],
                                   type=table.schema.field(column).type)
                table = table.set_column(table.schema.get_field_index(column), table.schema.field(column), values)
        pq.write_table(table, output/source.relative_to(base))
    for feature, names in [('observation.state', JOINT_STATE_NAMES), ('action', JOINT_ACTION_NAMES)]:
        info['features'][feature].update(shape=[len(names)], names=names)
    info['robot_type'] = 'quanta_x1_official_cylinder_joint_control'
    (output/'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
    stats = json.loads((base/'meta/stats.json').read_text())
    stats.update({'observation.state': statistics(np.concatenate(states)),
                  'action': statistics(np.concatenate(actions))})
    (output/'meta/stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    shutil.copyfile(base/'demonstration_provenance.json', output/'demonstration_provenance.json')
    for name in ('frame-sampling-weights.npy', 'frame-sampling-weights.json'):
        if (base/name).exists():
            shutil.copyfile(base/name, output/name)
    env = CylinderEnv(render=False)
    try:
        geometry = audit(env)
        require(geometry['passed'], 'Official geometry audit failed')
        limits = np.array([env.model.joint(f'right_arm_joint{i}').range for i in range(1, 7)])
    finally:
        env.close()
    targets = np.concatenate(actions)[:, :6]
    require(np.all(targets >= limits[:, 0] - 1e-6)
            and np.all(targets <= limits[:, 1] + 1e-6), 'Recorded joint target outside official limits')
    write_norm_stats(output, states, actions, limits, args.horizon)
    require(original_hashes == {str(p.relative_to(base)): digest(p)
                                for p in base.rglob('*') if p.is_file()}, 'Original dataset changed')
    report = dict(kind='verified_joint_dataset_export', passed=True, dataset=str(output),
        base_dataset=str(base), episodes=len(states), frames=info['total_frames'],
        robot_state_dimensions=18, action_dimensions=9, sources=sources,
        shared_videos=shared_videos, duplicate_video_bytes=0,
        source_dataset_sha256_unchanged=True, original_dataset_sha256=original_hashes,
        task_text_unchanged=True, non_vector_data_columns_unchanged=True,
        source_episode_order_and_frame_indices_preserved=True,
        official_joint_ranges_rad=limits.tolist(),
        joint_target_min_rad=targets.min(0).tolist(), joint_target_max_rad=targets.max(0).tolist(),
        official_geometry_audit=geometry, metadata_stats='Exact NumPy statistics; image statistics retained',
        new_model_training_started=False, task_success_measured=False,
        source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                       ('export_cylinder_joint_dataset.py', 'cylinder_joint_actions.py')})
    (output/'joint_export_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ('sources', 'original_dataset_sha256', 'official_geometry_audit')}), flush=True)


if __name__ == '__main__':
    main()
