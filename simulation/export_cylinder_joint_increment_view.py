"""Re-encode recorded joint setpoints as increments of the official servo command."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow
import pyarrow.parquet as pq

from carry_env import CONTROL_DT
from cylinder_joint_actions import (
    ARM_SERVO_RATE_RAD_S, JOINT_ACTION_NAMES, JOINT_SERVO_INCREMENT, JOINT_STATE_WIDTH,
    joint_command_action, require, servo_increment_targets,
)
from export_cylinder_joint_dataset import digest, replace_vector, statistics
from export_cylinder_joint_servo_view import initial_command


def export(base, output):
    require(not output.exists(), 'Output exists')
    manifest = base/'demonstration_provenance.json'
    demos = json.loads(manifest.read_text())
    info = json.loads((base/'meta/info.json').read_text())
    require(info['features']['action']['names'] == JOINT_ACTION_NAMES
            and info['total_episodes'] == len(demos), 'Expected the recorded absolute joint actions')
    table = arrow.dataset(base/'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'frame_index', 'observation.state', 'action']).sort_by('index')
    indices, episodes, frames = (np.asarray(table[key]) for key in ('index', 'episode_index', 'frame_index'))
    np.testing.assert_array_equal(indices, np.arange(info['total_frames']))
    np.testing.assert_array_equal(np.unique(episodes), np.arange(len(demos)))
    robot_states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    original = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    increments = original.copy()
    cap = ARM_SERVO_RATE_RAD_S * CONTROL_DT
    episode_stats = []
    roundtrip_errors = []
    for episode, demo in enumerate(demos):
        select = episodes == episode
        source = Path(demo['source'])
        trace = source/'trajectory.npz'
        require(digest(trace) == demo['trajectory_sha256'], 'Recorded trajectory changed')
        metadata = demo['metadata']
        result = metadata['result']
        require(result['success'] and result['geometry_audit_passed'] and result['grip_profile'] == 'official'
                and result['grip_force_target_n'] is None, 'Expected successful official servo recording')
        np.testing.assert_array_equal(frames[select], np.arange(metadata['frames']))
        with np.load(trace, allow_pickle=False) as sample:
            commands = sample['joint_command']
        np.testing.assert_array_equal(joint_command_action(commands), original[select])
        first, _ = initial_command(source, result, robot_states[select][0, :JOINT_STATE_WIDTH])
        previous = np.vstack((first, commands[:-1]))
        delta = commands[:, :6] - previous[:, :6]
        require(np.abs(delta).max() <= cap + 1e-12, 'A recorded label exceeds the physical joint servo rate')
        increments[select, :6] = delta
        decoded = servo_increment_targets(increments[select], previous, CONTROL_DT)
        np.testing.assert_allclose(decoded[:, :6], commands[:, :6], rtol=0, atol=2e-9)
        roundtrip_errors.append(float(np.abs(decoded[:, :6] - commands[:, :6]).max()))
        episode_stats.append(statistics(increments[select]))
    np.testing.assert_array_equal(increments[:, 6:], original[:, 6:])

    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging:
        view = Path(staging)/'dataset'
        view.mkdir()
        shutil.copytree(base/'meta', view/'meta')
        shutil.copytree(base/'videos', view/'videos', copy_function=os.link)
        videos = list((base/'videos').rglob('*.mp4'))
        require(videos and all(os.path.samefile(path, view/path.relative_to(base)) for path in videos),
                'Training videos are not shared')
        for path in sorted((base/'data').rglob('*.parquet')):
            table = pq.read_table(path)
            replaced = replace_vector(table, 'action', increments[np.asarray(table['index'])])
            require(all(replaced[name].equals(table[name]) for name in table.column_names if name != 'action'),
                    'Observations or episode indices changed')
            dest = view/path.relative_to(base)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(replaced, dest)
        for path in sorted((base/'meta/episodes').rglob('*.parquet')):
            table = pq.read_table(path)
            ids = np.asarray(table['episode_index'])
            for key in episode_stats[0]:
                name = f'stats/action/{key}'
                field = table.schema.field(name)
                values = pa.array([episode_stats[i][key] for i in ids], type=field.type)
                table = table.set_column(table.schema.get_field_index(name), field, values)
            pq.write_table(table, view/path.relative_to(base))
        info['features']['action']['names'] = [
            *(f'right_joint_command_increment_{i}_rad' for i in range(1, 7)), *JOINT_ACTION_NAMES[6:],
        ]
        (view/'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
        stats = json.loads((base/'meta/stats.json').read_text())
        stats['action'] = statistics(increments)
        (view/'meta/stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        shutil.copyfile(manifest, view/manifest.name)

        norm = json.loads((base/'joint_norm_stats.json').read_text())
        axis_low, axis_high = (norm['norm_stats']['action'][key][6:] for key in ('q01', 'q99'))
        norm['norm_stats']['action'] = dict(stats['action'], q01=[*([-cap]*6), *axis_low],
                                            q99=[*([cap]*6), *axis_high])
        norm['provenance'].update(
            relative_conversion='No loader subtraction; first six labels already increment the prior servo command',
            action_bounds='Arm q01/q99 are the physical per-tick increment bounds; gripper and base bounds are inherited',
            end_padding='Zero arm increments after the final frame, retaining final gripper and base requests')
        report = dict(kind='joint_actuator_increment_view', action_representation=JOINT_SERVO_INCREMENT,
                      source=str(base), output=str(output), source_provenance_sha256=digest(manifest),
                      episodes=len(demos), frames=len(increments), joint_rate_limit_rad_per_tick=cap,
                      maximum_arm_increment_rad=float(np.abs(increments[:, :6]).max()),
                      maximum_reconstructed_joint_error_rad=max(roundtrip_errors),
                      terminal_nonzero_increment_episodes=int(sum(
                          np.any(increments[episodes==episode][-1, :6] != 0) for episode in range(len(demos)))),
                      arm_padding='Zero increments in extra horizon rows after episode end',
                      gripper_base_observations_frames_and_prompts_exact=True, shared_video_files=len(videos),
                      shared_video_bytes=sum(path.stat().st_size for path in videos), independent_cases_read=False)
        norm['provenance']['servo_increment'] = report
        (view/'joint_norm_stats.json').write_text(json.dumps(norm, indent=2)+'\n')
        (view/'servo_increment_view_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        view.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base_dataset.resolve(), args.output.resolve())))


if __name__ == '__main__':
    main()
