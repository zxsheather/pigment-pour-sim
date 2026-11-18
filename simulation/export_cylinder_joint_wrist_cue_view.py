"""Join initial wrist-camera detections to an existing joint-servo dataset."""
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

from cylinder_joint_actions import (
    JOINT_SERVO_STATE_NAMES, JOINT_SERVO_STATE_WIDTH, WRIST_CUE_STATE_INDICES,
    WRIST_CUE_STATE_NAMES, require, with_joint_wrist_cue_state,
)
from cylinder_wrist_cue import MAX_MARKER_GAP_M, WRIST_CAMERA, WRIST_RESOLUTION
from export_cylinder_joint_dataset import digest, replace_vector, statistics


FEATURE = 'observation.state'


def write_view(base, output, info, states, episode_stats, cue_provenance, *,
               norm_provenance_key='initial_wrist_cue', state_names=WRIST_CUE_STATE_NAMES,
               normalization_radius_m=None):
    shutil.copytree(base/'meta', output/'meta')
    shutil.copytree(base/'videos', output/'videos', copy_function=os.link)
    videos = list((base/'videos').rglob('*.mp4'))
    require(videos and all(os.path.samefile(path, output/path.relative_to(base)) for path in videos),
            'Training videos are not shared')
    for path in sorted((base/'data').rglob('*.parquet')):
        table = pq.read_table(path)
        view = replace_vector(table, FEATURE, states[np.asarray(table['index'])])
        require(all(view[name].equals(table[name]) for name in table.column_names if name != FEATURE),
                'A non-state data column changed')
        dest = output/path.relative_to(base)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(view, dest)
    for path in sorted((base/'meta/episodes').rglob('*.parquet')):
        table = pq.read_table(path)
        ids = np.asarray(table['episode_index'])
        for key in episode_stats[0]:
            name = f'stats/{FEATURE}/{key}'
            field = table.schema.field(name)
            values = pa.array([episode_stats[i][key] for i in ids], type=field.type)
            table = table.set_column(table.schema.get_field_index(name), field, values)
        pq.write_table(table, output/path.relative_to(base))
    for index, name in zip(WRIST_CUE_STATE_INDICES, state_names, strict=True):
        info['features'][FEATURE]['names'][index] = name
    (output/'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
    stats = json.loads((base/'meta/stats.json').read_text())
    stats[FEATURE] = statistics(states)
    (output/'meta/stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    shutil.copyfile(base/'demonstration_provenance.json', output/'demonstration_provenance.json')

    cue = states[:, WRIST_CUE_STATE_INDICES]
    extra = statistics(cue)
    magnitudes = np.abs(cue[:, :2]).max(0)
    if normalization_radius_m is None:
        radius = np.maximum(magnitudes * 1.2, .65)
    else:
        require(np.isfinite(normalization_radius_m) and normalization_radius_m > 0
                and np.all(magnitudes <= normalization_radius_m), 'Cue exceeds its normalization interval')
        radius = np.full(2, normalization_radius_m)
    extra.update(q01=[*(-radius).tolist(), 0.], q99=[*radius.tolist(), 1.])
    norm = json.loads((base/'joint_norm_stats.json').read_text())
    for key, values in norm['norm_stats'][FEATURE].items():
        if key == 'count':
            continue
        require(len(values) == JOINT_SERVO_STATE_WIDTH, 'Expected joint-servo normalization')
        for index, value in zip(WRIST_CUE_STATE_INDICES, extra[key], strict=True):
            values[index] = value
    norm['provenance'][norm_provenance_key] = cue_provenance
    (output/'joint_norm_stats.json').write_text(json.dumps(norm, indent=2)+'\n')
    return len(videos), sum(path.stat().st_size for path in videos)


def export(base, ledger_path, output):
    require(not output.exists(), 'Output exists')
    info = json.loads((base/'meta/info.json').read_text())
    provenance_path = base/'demonstration_provenance.json'
    demos = json.loads(provenance_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    require(info['features'][FEATURE]['names'] == JOINT_SERVO_STATE_NAMES
            and len(demos) == len(ledger['rows']) == info['total_episodes'], 'Expected the joint-servo dataset')
    require(Path(ledger['data']).resolve() == base and ledger['proven_sha256'] == digest(provenance_path)
            and ledger['future_inputs_used'] is False and ledger['truth_used_for_cue'] is False,
            'Expected the causal RGB ledger for this dataset')
    dataset = arrow.dataset(base/'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'frame_index', FEATURE]).sort_by('index')
    indices, episode_ids, frame_ids = (np.asarray(dataset[key]) for key in ('index', 'episode_index', 'frame_index'))
    original = np.asarray(dataset[FEATURE].to_pylist(), dtype=np.float32)
    require(np.array_equal(indices, np.arange(info['total_frames']))
            and np.array_equal(np.unique(episode_ids), np.arange(len(demos))), 'Unexpected dataset index')
    states = original.copy()
    episode_stats, valid_episodes = [], 0
    for episode, (demo, row) in enumerate(zip(demos, ledger['rows'], strict=True)):
        require(row['episode'] == episode and row['source'] == demo['source']
                and row['recorded_seed'] == demo['metadata']['result']['seed']
                and row['camera'] == WRIST_CAMERA and row['resolution'] == WRIST_RESOLUTION,
                'Cue does not match the recorded initial frame')
        select = episode_ids == episode
        require(np.array_equal(frame_ids[select], np.arange(demo['metadata']['frames'])), 'Frame alignment changed')
        gap, xy = row['marker_pair_consistency_m'], row['world_xy_m']
        require(row['detected'] == (gap is not None and xy is not None), 'Inconsistent RGB detection')
        initial_xy = xy if row['detected'] and 0 <= gap <= MAX_MARKER_GAP_M else None
        valid_episodes += initial_xy is not None
        states[select] = with_joint_wrist_cue_state(original[select], initial_xy)
        episode_stats.append(statistics(states[select]))
    unchanged = sorted(set(range(JOINT_SERVO_STATE_WIDTH)) - set(WRIST_CUE_STATE_INDICES))
    np.testing.assert_array_equal(states[:, unchanged], original[:, unchanged])
    cue = states[:, WRIST_CUE_STATE_INDICES]
    require(np.all(cue[cue[:, 2] == 0] == 0), 'Invalid cue rows must be zero')
    provenance = dict(source_dataset=str(base), ledger=str(ledger_path), ledger_sha256=digest(ledger_path),
                      camera=WRIST_CAMERA, resolution=WRIST_RESOLUTION, max_marker_gap_m=MAX_MARKER_GAP_M,
                      state_indices=WRIST_CUE_STATE_INDICES, state_names=WRIST_CUE_STATE_NAMES,
                      episodes=len(demos), valid_episodes=valid_episodes, frames=len(states),
                      valid_frames=int(cue[:, 2].sum()), source_only_initial_RGB_and_current_odometry=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging:
        staged = Path(staging)/'dataset'
        staged.mkdir()
        videos, video_bytes = write_view(base, staged, info, states, episode_stats, provenance)
        report = dict(kind='joint_initial_wrist_cue_observation_view', **provenance, output=str(output),
                      other_state_fields_and_actions_exact=True, shared_video_files=videos,
                      shared_video_bytes=video_bytes, independent_cases_read=False)
        (staged/'wrist_cue_view_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--cue-ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base_dataset.resolve(), args.cue_ledger.resolve(), args.output.resolve())))


if __name__ == '__main__':
    main()
