"""Expand only right-XYZ action bounds using the existing training pool.

The original statistics and all other state/action channels remain untouched.
The envelope covers the full 32-action horizon, not only the executed prefix.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--margin-m", type=float, default=1e-5)
    args = parser.parse_args()
    assert args.output.resolve() != args.statistics.resolve()
    assert not args.output.exists(), "Do not overwrite an existing experiment"
    assert args.margin_m >= 0
    config = yaml.safe_load(args.config.read_text())
    layout = config["task"]["dof_config"]
    offset = 0
    for key, width in layout.items():
        if key == "follow_right_ee_cartesian_pos_relative":
            assert width == 3
            break
        offset += width
    else:
        raise ValueError("Expected relative right-XYZ action layout")
    horizon = config["task"]["action_horizon"]
    assert horizon == 32
    original = json.loads(args.statistics.read_text())
    updated = copy.deepcopy(original)
    manifest_bytes = (args.raw / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes)
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    entries = [entry for entry in manifest["episodes"] if entry["accepted"]]
    frames = targets = 0
    for entry in entries:
        path = args.raw / f"episode_{entry['episode']:04d}/trajectory.npz"
        digest.update(path.read_bytes())
        with np.load(path) as trajectory:
            states, actions = trajectory["state"], trajectory["action"]
            assert states.shape == actions.shape == (entry["frames"], 20)
            frames += len(states)
            for frame in range(len(states)):
                delta = actions[frame:frame + horizon, 7:10].astype(np.float64) - states[frame, 7:10]
                assert np.isfinite(delta).all()
                minimum = np.minimum(minimum, delta.min(axis=0))
                maximum = np.maximum(maximum, delta.max(axis=0))
                targets += len(delta)
    action = updated["norm_stats"]["action"]
    old_low = np.asarray(action["q01"][offset:offset + 3])
    old_high = np.asarray(action["q99"][offset:offset + 3])
    low = np.minimum(old_low, minimum - args.margin_m)
    high = np.maximum(old_high, maximum + args.margin_m)
    action["q01"][offset:offset + 3] = low.tolist()
    action["q99"][offset:offset + 3] = high.tolist()
    # Reverting precisely these six values must reconstruct the old file.
    restored = copy.deepcopy(updated)
    restored["norm_stats"]["action"]["q01"][offset:offset + 3] = old_low.tolist()
    restored["norm_stats"]["action"]["q99"][offset:offset + 3] = old_high.tolist()
    assert restored == original
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(updated, indent=2))
    report = {"changed_channels": "right Cartesian action XYZ only",
        "q01_q99_fields_now_contain_training_envelope_for_xyz": True,
        "all_other_statistics_identical": True, "external_data_used": False,
        "episodes": len(entries), "frames": frames, "horizon": horizon,
        "existing_horizon_targets": targets, "margin_m": args.margin_m,
        "right_xyz_layout_offset": offset,
        "old_low": old_low.tolist(), "old_high": old_high.tolist(),
        "observed_low": minimum.tolist(), "observed_high": maximum.tolist(),
        "new_low": low.tolist(), "new_high": high.tolist(),
        "training_data_sha256": digest.hexdigest(),
        "original_statistics_sha256": hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
        "updated_statistics_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
