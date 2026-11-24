"""Increase sampling of verified single-finger recovery demonstrations before closing."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cylinder_joint_actions import require
from prepare_cylinder_grasp_weights import grasp_groups, sha256
from recovery_sampler import RecoveryDistributedSampler


RECOVERY_PHASES = ('release_contact', 'withdraw_from_contact', 'approach', 'advance', 'close')


def first_loaded_fingers(source):
    with np.load(Path(source) / 'trajectory.npz', allow_pickle=False) as trajectory:
        return int(trajectory['contacts'][0])


def training_draws(split, path, *, seed, count):
    class SamplerInput:
        hf_dataset = split.dataset
        train_dataset = split

        def __len__(self):
            return len(self.train_dataset)

    sampler = RecoveryDistributedSampler(SamplerInput(), weights_path=path, num_replicas=1, rank=0, seed=seed,
                                        shuffle=True, drop_last=False)
    return np.asarray(split.indices)[np.asarray(list(sampler)[:count])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--base-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--target-share', type=float, required=True)
    args = p.parse_args()
    report_path = args.output.with_suffix('.json')
    require(args.output.suffix == '.npy' and not args.output.exists() and not report_path.exists(),
            'Use a new .npy weights path')
    require(0 < args.target_share < 1, 'Expected a positive probability below one')
    original_hash = sha256(args.base_weights)
    base = np.load(args.base_weights, allow_pickle=False)
    groups, phases, episodes, sources = grasp_groups(args.dataset)
    require(base.shape == episodes.shape and np.isfinite(base).all() and (base > 0).all(),
            'Base weights must match the dataset')
    split, _ = torch.utils.data.random_split(range(len(base)), [.95, .05], torch.Generator().manual_seed(args.seed))
    eligible = np.zeros(len(base), dtype=bool)
    eligible[np.asarray(split.indices)] = True
    initial_single = np.asarray([first_loaded_fingers(source) == 1 for source in sources])[episodes]
    pregrasp = initial_single & np.isin(phases, RECOVERY_PHASES)
    focus = pregrasp & groups['other']
    remaining = groups['other'] & ~focus
    total = base[eligible].sum()
    focus_share = float(base[eligible & focus].sum() / total)
    remaining_share = float(base[eligible & remaining].sum() / total)
    require(0 < focus_share < args.target_share < focus_share + remaining_share,
            'The target must increase contact recovery sampling within the available non-event weight')
    weights = base.copy().astype(np.float64)
    weights[focus] *= args.target_share / focus_share
    weights[remaining] *= (focus_share + remaining_share - args.target_share) / remaining_share
    require(np.isfinite(weights).all() and (weights > 0).all(), 'Expected positive finite recovery weights')
    require(np.array_equal(weights[~groups['other']], base[~groups['other']]), 'Event weights changed')
    require(np.isclose(weights[eligible].sum(), total, rtol=1e-10), 'Training probability mass changed')
    require(np.isclose(weights[eligible & focus].sum() / total, args.target_share, rtol=1e-10),
            'Contact recovery probability differs from the requested share')
    require(sha256(args.base_weights) == original_hash, 'Source sampler weights changed')
    np.save(args.output, weights, allow_pickle=False)
    draws = 512
    before, after = (training_draws(split, path, seed=args.seed, count=draws)
                     for path in (args.base_weights, args.output))
    episode_rows = []
    for i in np.unique(episodes[focus & eligible]):
        selected = episodes == i
        episode_rows.append(dict(episode=int(i), source=sources[i], eligible_focus_anchors=int((selected & focus & eligible).sum()),
                                 before_focus_draws=int((selected & focus)[before].sum()),
                                 after_focus_draws=int((selected & focus)[after].sum())))
    report = dict(kind='single_contact_pregrasp_training_sampling', dataset=str(args.dataset.resolve()),
                  frames=len(episodes), eligible_anchors=int(eligible.sum()), seed=args.seed,
                  base_weights_sha256=original_hash, weights_sha256=sha256(args.output),
                  base_focus_share=focus_share, target_focus_share=args.target_share,
                  pregrasp_phases=list(RECOVERY_PHASES), original_event_weights_exact=True,
                  initial_draw_count=draws, focus_draws_before=int(focus[before].sum()),
                  focus_draws_after=int(focus[after].sum()),
                  all_single_contact_pregrasp_share_before=float(base[pregrasp & eligible].sum() / total),
                  all_single_contact_pregrasp_share_after=float(weights[pregrasp & eligible].sum() / total),
                  episodes=episode_rows, learned_policy_trained=False, independent_cases_read=False)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
