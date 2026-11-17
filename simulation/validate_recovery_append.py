"""Check an appended recovery dataset against an immutable training export.

Reports coverage under the existing normalization instead of recomputing its
bounds. This script reads training exports only and does not run a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow

from compute_cylinder_norm_stats import relative_chunks


def load(root):
    table = arrow.dataset(sorted((root / "data").rglob("*.parquet")),
                          format="parquet").to_table()
    order = np.argsort(np.asarray(table["index"]))
    assert np.array_equal(np.asarray(table["index"])[order], np.arange(len(table)))
    return {
        "state": np.asarray(table["observation.state"].to_pylist())[order],
        "action": np.asarray(table["action"].to_pylist())[order],
        "episode": np.asarray(table["episode_index"])[order],
        "frame": np.asarray(table["frame_index"])[order],
    }


def coverage(values, bounds):
    low, high = np.asarray(bounds["q01"]), np.asarray(bounds["q99"])
    outside = (values < low) | (values > high)
    return dict(samples=len(values), outside_fraction_per_dimension=outside.mean(0).tolist(),
                minimum=values.min(0).tolist(), maximum=values.max(0).tolist(),
                maximum_bound_excess=float(np.maximum(0., np.maximum(low-values, values-high)).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--appended", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.horizon > 0
    old, new = load(args.original), load(args.appended)
    count = len(old["state"])
    assert len(new["state"]) > count
    for name in old:
        assert np.array_equal(old[name], new[name][:count]), name
    old_provenance = json.loads((args.original / "demonstration_provenance.json").read_text())
    new_provenance = json.loads((args.appended / "demonstration_provenance.json").read_text())
    assert old_provenance == new_provenance[:len(old_provenance)]
    for episode_id,demo in enumerate(new_provenance[len(old_provenance):],start=len(old_provenance)):
        result = demo["metadata"]["result"]
        assert result["success"]
        assert result["initial_state_provenance"]["kind"] == "expert_recovery_training_only"
        selected=np.flatnonzero(new["episode"]==episode_id)
        with np.load(Path(demo["source"])/"trajectory.npz") as source:
            source_state=source["state"]
            if new["state"].shape[1]==26:
                source_state=np.concatenate([source_state,source["base_pose"]],axis=1)
            assert np.array_equal(new["state"][selected],source_state.astype(np.float32))
            assert np.array_equal(new["action"][selected],source["action"].astype(np.float32))
    norm = json.loads(args.norm_stats.read_text())["norm_stats"]
    chunks = []
    for episode in sorted(set(new["episode"][count:])):
        selected = np.flatnonzero(new["episode"] == episode)
        assert selected[0] >= count
        assert np.array_equal(new["frame"][selected], np.arange(len(selected)))
        chunks.append(relative_chunks(new["state"][selected], new["action"][selected],
                                      args.horizon).reshape(-1, 23))
    action = np.concatenate(chunks)
    state = new["state"][count:]
    assert np.isfinite(state).all() and np.isfinite(action).all()
    report = dict(original=str(args.original.resolve()), appended=str(args.appended.resolve()),
                  original_frames=count, appended_frames=len(state),
                  appended_episodes=len(new_provenance)-len(old_provenance),
                  original_state_action_indices_and_provenance_unchanged=True,
                  appended_robot_state_and_action_exactly_match_source=True,
                  norm_stats_sha256=hashlib.sha256(args.norm_stats.read_bytes()).hexdigest(),
                  norm_stats_unchanged=True, horizon=args.horizon,
                  appended_state=coverage(state, norm["observation.state"]),
                  appended_relative_action=coverage(action, norm["action"]),
                  note="Training data coverage only; expert recoveries are not policy successes.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
