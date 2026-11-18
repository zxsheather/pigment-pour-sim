"""Create auditable frame weights from successful recovery-demo provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-recovery-fraction",type=float,
                   help="Rescale the recovery group to a fixed sampling share while retaining within-group priorities.")
    p.add_argument('--full-start-boost', type=float, default=1.,
                   help='Multiply sampling weights for the start of full demonstrations.')
    p.add_argument('--full-start-seconds', type=float, default=8.,
                   help='Duration of the weighted prefix of each full demonstration.')
    a = p.parse_args()
    if a.target_recovery_fraction is not None and not 0<a.target_recovery_fraction<1:
        p.error("Target recovery fraction must be in (0,1)")
    if (not np.isfinite(a.full_start_boost) or a.full_start_boost < 1.
            or not np.isfinite(a.full_start_seconds) or a.full_start_seconds <= 0.):
        p.error('Full start boost must be at least 1 and the duration must be positive')
    assert not a.output.exists()
    provenance = json.loads((a.dataset/"demonstration_provenance.json").read_text())
    table = arrow.dataset(sorted((a.dataset/"data").rglob("*.parquet")), format="parquet").to_table(
        columns=["index", "episode_index", "timestamp"])
    order = np.argsort(np.asarray(table["index"]))
    assert np.array_equal(np.asarray(table["index"])[order], np.arange(len(order)))
    episodes = np.asarray(table["episode_index"])[order]
    times = np.asarray(table["timestamp"])[order]
    weights = np.ones(len(order), dtype=np.float64)
    recovery = np.zeros(len(order), dtype=bool)
    rows = []
    for episode, demo in enumerate(provenance):
        meta = demo["metadata"]; result = meta["result"]
        initial = result.get("initial_state_provenance")
        # The teacher summary stores initialization explicitly. Never infer
        # recovery membership from a filename or from a failed rollout score.
        if not initial:
            source = Path(demo["source"])/"recovery_source.json"
            if source.exists(): initial = json.loads(source.read_text())
        if not initial: continue
        assert result["success"] and initial["kind"] == "expert_recovery_training_only"
        selected = episodes == episode
        recovery[selected] = True
        weights[selected] = np.where(times[selected] < 2, 64.,
                                     np.where(times[selected] < 6, 8., 2.))
        rows.append(dict(episode=episode, frames=int(selected.sum()), source=demo["source"],
                         starts_holding=initial.get("starts_holding", False)))
    assert rows, "No successful recovery demonstrations found"
    full_start = ~recovery & (times < a.full_start_seconds)
    weights[full_start] *= a.full_start_boost
    group_scale=1.
    if a.target_recovery_fraction is not None:
        group_scale=(weights[~recovery].sum()*a.target_recovery_fraction
                     /((1-a.target_recovery_fraction)*weights[recovery].sum()))
        weights[recovery]*=group_scale
    np.save(a.output, weights, allow_pickle=False)
    report = dict(dataset=str(a.dataset.resolve()), frames=len(weights), recovery_episodes=rows,
        weight_rule="Original frames 1; recovery <2s:64, 2..6s:8, >=6s:2",
        recovery_group_scale=group_scale,target_recovery_sampling_fraction=a.target_recovery_fraction,
        recovery_frame_fraction=float(recovery.mean()),
        expected_recovery_sampling_fraction=float(weights[recovery].sum()/weights.sum()),
        full_start_weight_multiplier=a.full_start_boost, full_start_duration_s=a.full_start_seconds,
        expected_full_start_sampling_fraction=float(weights[full_start].sum()/weights.sum()),
        weights_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        note="Sampling probabilities, not new observations; independent test cases are never read")
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
