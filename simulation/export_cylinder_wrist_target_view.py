"""Join the RGB target estimate to measured wrist kinematics in recorded demos."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.dataset as arrow

from cylinder_joint_actions import JOINT_SERVO_INCREMENT, JOINT_SERVO_STATE_WIDTH, require
from cylinder_progress_prompts import PROGRESS_PROTOCOL
from cylinder_tray_vision import (
    TRAY_CENTER_CUE_PROTOCOL, WRIST_TARGET_CUE_INDICES, WRIST_TARGET_CUE_NAMES,
    WRIST_TARGET_CUE_PROTOCOL, WRIST_TARGET_NEAR_SCALE_M, tray_grasp_height_m, with_wrist_target_state,
)
from export_cylinder_joint_dataset import digest, statistics
from export_cylinder_joint_wrist_cue_view import FEATURE, write_view


LATERAL_NORMALIZATION_RANGE_M = .45


def export(base, ledger_path, output):
    require(not output.exists(), 'Output exists')
    manifest = base/'demonstration_provenance.json'
    demos = json.loads(manifest.read_text())
    info = json.loads((base/'meta/info.json').read_text())
    ledger = json.loads(ledger_path.read_text())
    previous = json.loads((base/'servo_increment_view_audit.json').read_text())
    calibration_path = Path(__file__).with_name('scene_config.json')
    scene = json.loads(calibration_path.read_text())
    norm = json.loads((base/'joint_norm_stats.json').read_text())['provenance']
    require(previous['action_representation'] == JOINT_SERVO_INCREMENT
            and previous['source_provenance_sha256'] == digest(manifest)
            and norm['initial_tray_vision']['tray_cue_protocol'] == TRAY_CENTER_CUE_PROTOCOL
            and norm['initial_tray_vision']['ledger_sha256'] == digest(ledger_path),
            'Expected the increment dataset with its original tray cue ledger')
    rows = ledger['episodes_with_estimates']
    require(len(demos) == len(rows) == info['total_episodes']
            and ledger['kind'] == 'calibrated_reset_rgb_tray_cue_ledger'
            and ledger['provenance_sha256'] == digest(manifest)
            and ledger['future_inputs_used'] is False and ledger['truth_used_for_estimation'] is False
            and ledger['independent_cases_read'] is False, 'Expected verified causal RGB observations')

    table = arrow.dataset(base/'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'frame_index', 'task_index', FEATURE]).sort_by('index')
    ids, episodes, frames, tasks = (
        np.asarray(table[k]) for k in ('index','episode_index','frame_index','task_index'))
    np.testing.assert_array_equal(ids, np.arange(info['total_frames']))
    np.testing.assert_array_equal(np.unique(episodes), np.arange(len(demos)))
    original = np.asarray(table[FEATURE].to_pylist(), dtype=np.float32)
    updated = original.copy()
    updated[:, WRIST_TARGET_CUE_INDICES] = 0.
    episode_stats = []
    for episode, (demo, row) in enumerate(zip(demos, rows, strict=True)):
        record = demo['metadata']
        p = record['result']['controller_parameters']
        require(row['episode'] == episode and row['source'] == demo['source']
                and row['seed'] == record['result']['seed'] and row['source_state_verified'] is True,
                'Reset cue does not match the recorded episode')
        select = episodes == episode
        np.testing.assert_array_equal(frames[select], np.arange(record['frames']))
        if row['estimate'] is not None:
            require(row['reset_frame'] and row['rgb_sha256'], 'Expected captured reset RGB')
            trace = Path(demo['source'])/'trajectory.npz'
            require(digest(trace) == demo['trajectory_sha256'], 'Recorded wrist kinematics changed')
            approach = tasks[select] == 0
            with np.load(trace, allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['base_pose'].astype(np.float32), original[select, 15:18])
                updated[select & (tasks == 0)] = with_wrist_target_state(
                    original[select & (tasks == 0)], raw['state'][approach], row['estimate'],
                    grasp_x_m=p['grasp_x_m'], grasp_target_height_m=tray_grasp_height_m(scene, p['grasp_height_m']))
        episode_stats.append(statistics(updated[select]))

    preserved = sorted(set(range(JOINT_SERVO_STATE_WIDTH)) - set(WRIST_TARGET_CUE_INDICES))
    np.testing.assert_array_equal(original[:, preserved], updated[:, preserved])
    cues = updated[:, WRIST_TARGET_CUE_INDICES]
    valid = cues[:, -1] == 1
    np.testing.assert_array_equal(valid, original[:, WRIST_TARGET_CUE_INDICES[-1]] == 1)
    require(np.all(cues[~valid] == 0) and np.all(tasks[valid] == 0), 'Unexpected cue outside approach')
    minimum_radius = [.65, np.arcsinh(LATERAL_NORMALIZATION_RANGE_M / WRIST_TARGET_NEAR_SCALE_M), .65]
    radii = np.maximum(np.abs(cues[:, :3]).max(0) * 1.2, minimum_radius)
    proof = dict(kind='measured_wrist_to_visual_target_observation_view',
                 source=str(base), output=str(output), ledger=str(ledger_path), ledger_sha256=digest(ledger_path),
                 scene_sha256=digest(calibration_path), observations_protocol=WRIST_TARGET_CUE_PROTOCOL,
                 prompts_protocol=PROGRESS_PROTOCOL, observation_indices=WRIST_TARGET_CUE_INDICES,
                 observation_names=WRIST_TARGET_CUE_NAMES, observed_residual_scale_m=WRIST_TARGET_NEAR_SCALE_M,
                 normalization_radii_asinh=radii.tolist(), lateral_normalization_minimum_range_m=LATERAL_NORMALIZATION_RANGE_M,
                 episodes=len(demos), frames=len(updated), cue_frames=int(valid.sum()),
                 cue_episodes=int(np.unique(episodes[valid]).size),
                 cue_inputs='Reset RGB estimate, current robot wrist and base kinematics with static grasp calibration',
                 actions_other_observations_frames_and_prompts_exact=True, independent_cases_read=False)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as tmp:
        staged = Path(tmp)/'dataset'
        staged.mkdir()
        videos, size = write_view(base, staged, info, updated, episode_stats, proof,
                                  norm_provenance_key='wrist_to_visual_target', state_names=WRIST_TARGET_CUE_NAMES,
                                  state_indices=WRIST_TARGET_CUE_INDICES, normalization_radius_m=radii)
        proof.update(shared_video_files=videos, shared_video_bytes=size)
        (staged/'wrist_target_view_audit.json').write_text(json.dumps(proof, indent=2)+'\n')
        staged.rename(output)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--cue-ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base_dataset.resolve(), args.cue_ledger.resolve(), args.output.resolve())))


if __name__ == '__main__':
    main()
