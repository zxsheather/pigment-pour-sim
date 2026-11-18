"""Add measured reset positions to the approach phase of joint demonstrations."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.dataset as arrow

from cylinder_joint_actions import (
    JOINT_SERVO_STATE_NAMES, JOINT_SERVO_STATE_WIDTH, WRIST_CUE_STATE_INDICES,
    WRIST_CUE_STATE_NAMES, require, with_joint_wrist_cue_state,
)
from cylinder_progress_prompts import PROGRESS_PROMPTS, PROGRESS_PROTOCOL
from export_cylinder_joint_dataset import digest, statistics
from export_cylinder_joint_wrist_cue_view import FEATURE, write_view


def export(base, ledger_path, output):
    require(not output.exists(), 'Output exists')
    info = json.loads((base/'meta/info.json').read_text())
    provenance_path = base/'demonstration_provenance.json'
    demos = json.loads(provenance_path.read_text())
    prompt_view = json.loads((base/'progress_prompt_view.json').read_text())
    ledger = json.loads(ledger_path.read_text())
    rows = ledger['episodes_with_estimates']
    require(info['features'][FEATURE]['names'] == JOINT_SERVO_STATE_NAMES
            and len(demos) == len(rows) == info['total_episodes'], 'Expected the joint-servo dataset')
    require(prompt_view['protocol'] == PROGRESS_PROTOCOL
            and tuple(prompt_view['prompts']) == PROGRESS_PROMPTS, 'Expected the physical progress prompts')
    require(ledger['kind'] == 'calibrated_reset_rgb_tray_cue_ledger'
            and Path(ledger['dataset']).resolve() == base
            and ledger['provenance_sha256'] == digest(provenance_path)
            and ledger['future_inputs_used'] is False and ledger['truth_used_for_estimation'] is False
            and ledger['independent_cases_read'] is False,
            'Expected the causal tray cue ledger for this dataset')
    table = arrow.dataset(base/'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'frame_index', 'task_index', FEATURE]).sort_by('index')
    indices, episodes, frames, tasks = (
        np.asarray(table[key]) for key in ('index', 'episode_index', 'frame_index', 'task_index')
    )
    np.testing.assert_array_equal(indices, np.arange(info['total_frames']))
    np.testing.assert_array_equal(np.unique(episodes), np.arange(len(demos)))
    np.testing.assert_array_equal(np.unique(tasks), np.arange(len(PROGRESS_PROMPTS)))
    original = np.asarray(table[FEATURE].to_pylist(), dtype=np.float32)
    states = with_joint_wrist_cue_state(original, None)
    valid = np.zeros(len(states), dtype=bool)
    episode_stats = []
    for episode, (demo, row) in enumerate(zip(demos, rows, strict=True)):
        require(row['episode'] == episode and row['source'] == demo['source']
                and row['seed'] == demo['metadata']['result']['seed']
                and row['source_state_verified'] is True, 'Cue does not match the initial observation')
        select = episodes == episode
        np.testing.assert_array_equal(frames[select], np.arange(demo['metadata']['frames']))
        if row['estimate'] is not None:
            require(row['reset_frame'] and row['rgb_sha256'], 'Cue requires a recorded reset observation')
            approach = select & (tasks == 0)
            states[approach] = with_joint_wrist_cue_state(original[approach], row['estimate']['world_xy_m'])
            valid[approach] = True
        episode_stats.append(statistics(states[select]))

    unchanged = sorted(set(range(JOINT_SERVO_STATE_WIDTH)) - set(WRIST_CUE_STATE_INDICES))
    np.testing.assert_array_equal(states[:, unchanged], original[:, unchanged])
    cue = states[:, WRIST_CUE_STATE_INDICES]
    np.testing.assert_array_equal(cue[:, 2], valid)
    require(np.all(cue[~valid] == 0) and np.all(tasks[valid] == 0), 'Cue leaked outside valid approach')
    provenance = dict(source_dataset=str(base), ledger=str(ledger_path), ledger_sha256=digest(ledger_path),
                      state_indices=WRIST_CUE_STATE_INDICES, state_names=WRIST_CUE_STATE_NAMES,
                      prompts_protocol=PROGRESS_PROTOCOL, enabled_task_index=0,
                      episodes=len(demos), valid_episodes=int(np.unique(episodes[valid]).size),
                      frames=len(states), valid_frames=int(valid.sum()),
                      source_only_initial_RGB_and_current_odometry=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging:
        staged = Path(staging)/'dataset'
        staged.mkdir()
        videos, video_bytes = write_view(base, staged, info, states, episode_stats, provenance,
                                         norm_provenance_key='initial_tray_vision')
        report = dict(kind='joint_initial_tray_cue_observation_view', **provenance, output=str(output),
                      other_state_fields_actions_and_prompts_exact=True, shared_video_files=videos,
                      shared_video_bytes=video_bytes, independent_cases_read=False)
        (staged/'tray_cue_view_audit.json').write_text(json.dumps(report, indent=2)+'\n')
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
