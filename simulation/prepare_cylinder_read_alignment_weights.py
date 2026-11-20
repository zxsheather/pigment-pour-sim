"""Reweight the final RGB height adjustment and physical hold in cylinder demonstrations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow
import torch


def prepare(dataset, base_weights, output, seed, alignment_share, hold_share):
    report_path = output.with_suffix('.json')
    if output.suffix != '.npy' or output.exists() or report_path.exists():
        raise ValueError('Choose a new .npy sampling weights path')
    table = arrow.dataset(dataset / 'data', format='parquet').to_table(columns=[
        'index', 'episode_index', 'frame_index', 'task_index', 'action',
    ]).sort_by('index')
    row, episode, frame, task = (np.asarray(table[k]) for k in (
        'index', 'episode_index', 'frame_index', 'task_index',
    ))
    base = np.load(base_weights, allow_pickle=False)
    np.testing.assert_array_equal(row, np.arange(len(base)))
    if base.ndim != 1 or not np.isfinite(base).all() or not (base > 0).all():
        raise ValueError('Expected finite positive weights for every training row')

    phase = np.empty(len(row), dtype='<U48')
    manifest = json.loads((dataset / 'demonstration_provenance.json').read_text())
    np.testing.assert_array_equal(np.unique(episode), np.arange(len(manifest)))
    for index, source in enumerate(manifest):
        with np.load(Path(source['source']) / 'trajectory.npz', allow_pickle=False) as trajectory:
            mask = episode == index
            np.testing.assert_array_equal(frame[mask], np.arange(len(trajectory['phase'])))
            phase[mask] = trajectory['phase']

    action = np.asarray(table['action'].to_pylist())
    presentation = task == 2
    read = presentation & (phase == 'read_stable')
    quiet = ((np.max(np.abs(action[:, :6]), axis=1) <= .002)
             & (np.abs(action[:, 7]) <= .005) & (np.abs(action[:, 8]) <= .015))
    ends = np.r_[np.flatnonzero(np.diff(episode)) + 1, len(row)]
    horizon = row[:, None] + np.arange(8)
    complete = horizon < ends[episode, None]
    frames = np.minimum(horizon, ends[episode, None] - 1)
    hold = (complete & read[frames] & quiet[frames]).all(axis=1)
    next_read = np.minimum.accumulate(np.where(read, row, len(row))[::-1])[::-1]
    # The last measured height corrections are more relevant to reading than
    # the earlier moves that bring the wrist into the head camera view.
    moves_wrist = np.max(np.abs(action[:, :6]), axis=1) > 1e-5
    height_alignment = (phase == 'align_eye_level') & (next_read - row <= 90)
    view_search = (phase == 'find_scale') & (next_read - row <= 130)
    alignment = presentation & (height_alignment | view_search) & moves_wrist & (next_read < ends[episode])
    remainder = presentation & ~(alignment | hold)

    split, _ = torch.utils.data.random_split(range(len(base)), [.95, .05],
                                            torch.Generator().manual_seed(seed))
    fit = np.zeros(len(base), dtype=bool)
    fit[np.asarray(split.indices)] = True
    fit_total = base[fit].sum()
    presentation_share = base[fit & presentation].sum() / fit_total
    targets = (alignment_share, hold_share, presentation_share - alignment_share - hold_share)
    adjusted = base.copy().astype(np.float64)
    groups = dict(final_height_and_view_alignment=alignment, stable_read_hold=hold,
                  other_presentation=remainder)
    stats = {}
    for (name, group), target in zip(groups.items(), targets, strict=True):
        selected = group & fit
        previous_share = base[selected].sum() / fit_total
        if not 0 < previous_share < 1 or not 0 < target < 1:
            raise ValueError(f'Invalid share for {name}: {previous_share} to {target}')
        adjusted[group] *= target / previous_share
        stats[name] = dict(fit_anchors=int(selected.sum()), episodes=int(np.unique(episode[selected]).size),
                           old_share=float(previous_share), new_share=float(target))
    np.testing.assert_array_equal(adjusted[~presentation], base[~presentation])
    np.testing.assert_allclose(adjusted[fit].sum(), fit_total, atol=1e-12, rtol=0.)
    for (_, group), target in zip(groups.items(), targets, strict=True):
        np.testing.assert_allclose(adjusted[group & fit].sum() / fit_total, target, atol=1e-12, rtol=0.)
    np.save(output, adjusted, allow_pickle=False)
    report = dict(kind='stationary_reads_and_final_vertical_alignment_weights', dataset=str(dataset),
                  base_weights=str(base_weights), weights=str(output),
                  sampler_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  frame_count=len(row), fitting_examples=int(fit.sum()), groups=stats,
                  fitting_presentation_share=float(presentation_share),
                  fitting_total_weight_unchanged=True, other_raw_weights_exact=True,
                  height_frames_before_read_max=90, view_search_frames_before_read_max=130,
                  horizon=8, fit_seed=seed,
                  independent_cases_read=False)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base-weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--alignment-share', type=float, required=True)
    parser.add_argument('--hold-share', type=float, required=True)
    args = parser.parse_args()
    result = prepare(args.dataset, args.base_weights, args.output, args.seed, args.alignment_share, args.hold_share)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
