"""Reweight grasp-correction and closing windows without modifying a dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow
import torch

from cylinder_joint_actions import (APERTURE_SCALE_M, JOINT_ACTION_WIDTH, JOINT_STATE_WIDTH,
                                    joint_command_action, require)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grasp_groups(dataset):
    provenance = json.loads((dataset / 'demonstration_provenance.json').read_text())
    table = arrow.dataset(dataset / 'data', format='parquet').to_table(columns=[
        'index', 'episode_index', 'frame_index', 'timestamp', 'observation.state', 'action'])
    table = table.sort_by('index')
    require(np.array_equal(np.asarray(table['index']), np.arange(len(table))),
            'Expected one row for each frame index')
    episode_ids = np.asarray(table['episode_index'])
    frame_ids = np.asarray(table['frame_index'])
    timestamps = np.asarray(table['timestamp'])
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    require(states.shape == (len(table), JOINT_STATE_WIDTH)
            and actions.shape == (len(table), JOINT_ACTION_WIDTH)
            and np.isfinite(states).all() and np.isfinite(actions).all(),
            'Expected finite joint states and actions')
    require(np.array_equal(np.unique(episode_ids), np.arange(len(provenance))),
            'Dataset episodes do not match demonstration provenance')
    reopen = np.zeros(len(table), dtype=bool)
    closing = np.zeros_like(reopen)
    phases = np.empty(len(table), dtype='<U64')
    sources = []
    for episode, record in enumerate(provenance):
        trajectory = Path(record['source']) / 'trajectory.npz'
        require(sha256(trajectory) == record['trajectory_sha256'], 'Changed source trajectory')
        result = record['metadata']['result']
        require(result['success'] and result['geometry_audit_passed'] and result['grip_profile'] == 'official',
                'Expected a successful official-geometry demonstration')
        selected = episode_ids == episode
        with np.load(trajectory, allow_pickle=False) as raw:
            phase = raw['phase']
            frames = np.arange(len(phase))
            require(np.array_equal(frame_ids[selected], frames)
                    and np.allclose(timestamps[selected], raw['timestamp'], rtol=0, atol=4e-6)
                    and np.array_equal(actions[selected], joint_command_action(raw['joint_command'])),
                    'Dataset order or labels do not match the recorded demonstration')
        # Production takes the first eight outputs of a 32-action prediction.
        # Each episode pads its endpoint with its final command.
        future = np.minimum(frames[:, None] + np.arange(8), len(phase) - 1)
        approaching = np.isin(phase, ('release_contact', 'withdraw_from_contact', 'approach', 'advance'))
        requested = actions[selected, 6][future].mean(axis=1)
        widening_mm = (requested - states[selected, 6]) * APERTURE_SCALE_M * 1000
        reopen[selected] = approaching[future].all(axis=1) & (widening_mm >= 10)
        closing[selected] = approaching & (phase[future] == 'close').any(axis=1)
        phases[selected] = phase
        sources.append(record['source'])
    require(not (reopen & closing).any(), 'Inconsistent grasp windows')
    return dict(reopen=reopen, start_close=closing, other=~(reopen | closing)), phases, episode_ids, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base-weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True, help='Training configuration seed')
    parser.add_argument('--reopen-share', type=float, required=True)
    parser.add_argument('--start-close-share', type=float, required=True)
    args = parser.parse_args()
    require(args.output.suffix == '.npy' and not args.output.exists()
            and not args.output.with_suffix('.json').exists(), 'Use a new .npy output path')
    targets = dict(reopen=args.reopen_share, start_close=args.start_close_share,
                   other=1 - args.reopen_share - args.start_close_share)
    require(all(value > 0 for value in targets.values()), 'Group shares must be positive and sum to one')
    base_hash = sha256(args.base_weights)
    base = np.load(args.base_weights, allow_pickle=False)
    groups, phases, episodes, sources = grasp_groups(args.dataset)
    require(base.shape == phases.shape and np.isfinite(base).all() and (base > 0).all(),
            'Base weights must match the full dataset and be finite and positive')
    train, _ = torch.utils.data.random_split(
        range(len(base)), [.95, .05], torch.Generator().manual_seed(args.seed))
    eligible = np.zeros(len(base), dtype=bool)
    eligible[np.asarray(train.indices)] = True
    denominator = base[eligible].sum()
    weights = base.copy().astype(np.float64)
    report_groups = {}
    for name, mask in groups.items():
        selected = eligible & mask
        fraction = float(base[selected].sum() / denominator)
        require(fraction > 0, f'No eligible anchors in {name}')
        multiplier = targets[name] / fraction
        weights[mask] *= multiplier
        counts = np.bincount(episodes[selected], minlength=len(sources))
        report_groups[name] = dict(total_anchors=int(mask.sum()), eligible_anchors=int(selected.sum()),
                                   base_share=fraction, multiplier=multiplier,
                                   source_count=int(np.count_nonzero(counts)))
    require(np.isfinite(weights).all() and (weights > 0).all(), 'Invalid final weights')
    for name, mask in groups.items():
        share = float(weights[mask & eligible].sum() / weights[eligible].sum())
        require(np.isclose(share, targets[name], rtol=0, atol=1e-9), 'Unexpected training group share')
        report_groups[name]['new_share'] = share
    phase_share = {}
    for phase in np.unique(phases):
        mask = eligible & (phases == phase)
        phase_share[str(phase)] = dict(base=float(base[mask].sum() / denominator),
                                     new=float(weights[mask].sum() / weights[eligible].sum()))
    require(sha256(args.base_weights) == base_hash, 'Base weights changed during preparation')
    np.save(args.output, weights, allow_pickle=False)
    report = dict(kind='grasp_event_training_sampling', dataset=str(args.dataset.resolve()),
                  base_weights_sha256=base_hash, weights_sha256=sha256(args.output),
                  frames=len(base), eligible_anchors=int(eligible.sum()), training_split_seed=args.seed,
                  groups=report_groups, anchor_phase_shares=phase_share,
                  executed_prefix=8, reopen_min_full_width_difference_mm=10,
                  phase_labels_used_only_to_construct_offline_sampling_weights=True,
                  original_dataset_modified=False, independent_cases_read=False,
                  learned_policy_trained=False)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
