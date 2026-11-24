"""Give verified forward approach labels a specified share of training draws."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset
import pyarrow.parquet

from cylinder_progress_prompts import PROGRESS_PROMPTS


def reweight(dataset, fraction, minimum_forward):
    tasks = pyarrow.parquet.read_table(dataset/'meta/tasks.parquet').to_pandas()
    approach_id = int(tasks.loc[PROGRESS_PROMPTS[0], 'task_index'])
    table = pyarrow.dataset.dataset(dataset/'data', format='parquet').to_table(
        columns=['index', 'task_index', 'action']).sort_by('index')
    weights = np.load(dataset/'frame-sampling-weights.npy', allow_pickle=False).astype(np.float64)
    np.testing.assert_array_equal(np.asarray(table['index']), np.arange(len(weights)))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Expected finite positive baseline weights')
    actions = np.asarray(table['action'].to_pylist())
    if actions.shape != (len(weights), 9) or not np.isfinite(actions).all():
        raise ValueError('Expected finite nine-dimensional joint actions')
    velocity = actions[:, 7]
    selected = (np.asarray(table['task_index']) == approach_id) & (velocity > minimum_forward)
    if not selected.any() or selected.all():
        raise ValueError('Expected both forward approach labels and other actions')
    before = float(weights[selected].sum() / weights.sum())
    if not 0. < before < 1.:
        raise ValueError('Both groups must have nonzero nominal sampling mass')
    weights[selected] *= fraction / before
    weights[~selected] *= (1. - fraction) / (1. - before)
    after = float(weights[selected].sum() / weights.sum())
    np.testing.assert_allclose(after, fraction, atol=1e-12, rtol=0)
    audit = dict(dataset=str(dataset), frames=len(weights), selected_frames=int(selected.sum()),
                 minimum_forward_m_s=minimum_forward,
                 forward_range_m_s=[float(velocity[selected].min()), float(velocity[selected].max())],
                 previous_nominal_mass=before, revised_nominal_mass=after,
                 relative_weights_preserved_within_each_group=True)
    return weights, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--nominal-mass', type=float, required=True)
    parser.add_argument('--minimum-forward', type=float, required=True, help='Forward velocity in metres per second')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if not 0. < args.nominal_mass < 1. or not np.isfinite(args.minimum_forward) or args.minimum_forward <= 0:
        parser.error('Expected a nominal mass in (0, 1) and a positive velocity threshold')
    if args.output.exists() or args.audit.exists() or args.output.resolve() == args.audit.resolve():
        parser.error('Expected two distinct new output paths')
    weights, report = reweight(args.dataset.resolve(), args.nominal_mass, args.minimum_forward)
    with args.output.open('xb') as handle:
        np.save(handle, weights, allow_pickle=False)
    with args.audit.open('x') as handle:
        json.dump(report | {'weights': str(args.output.resolve())}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
