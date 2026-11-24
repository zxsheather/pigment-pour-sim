"""Measure right-XYZ target clipping, checking against the actual collator.

Only recorded targets in the executed prefix are scored; padded terminal
targets and rotation/gripper round trips are outside this diagnostic.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clip-error-mm", type=float,
                        help="Fail if any recorded right-XYZ label exceeds this round-trip error")
    parser.add_argument("--prefix-steps", type=int, choices=[8, 32], default=8,
                        help="Audit the executed prefix or the full training horizon")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from wall_x.config.loader import load_config
    from wall_x.data.backends.lerobot.build import _build_flat_config
    from wall_x.data.backends.lerobot.loader import (
        PreprocessedDataset, DataCollator, get_data_configs, load_norm_stats)

    cfg = load_config(str(args.config))
    flat, lr_cfg = _build_flat_config(cfg)
    data_cfg = get_data_configs(flat["data"])
    stats = load_norm_stats(flat["norm_stats_path"], data_cfg["key_mappings"],
        dof_config=flat["dof_config"], agent_pos_config=flat["agent_pos_config"])
    dataset = LeRobotDataset(lr_cfg["repo_id"], root=lr_cfg["root"],
        delta_timestamps={"action": [i / 25 for i in range(data_cfg["action_horizon"] - 1)]},
        video_backend="pyav")
    processed = PreprocessedDataset(dataset, flat, stats, data_cfg, lr_cfg, test_only=True)
    collator = DataCollator(flat, data_cfg, stats, lr_cfg)
    offset = 0
    for key, dim in flat["dof_config"].items():
        if key == "follow_right_ee_cartesian_pos_relative":
            assert dim == 3
            break
        offset += dim
    else:
        raise ValueError("Expected relative right Cartesian position in the 26-D layout")
    lo = collator.action_min_stat.numpy()[offset:offset + 3]
    delta = collator.action_delta.numpy()[offset:offset + 3]
    assert np.all(delta > 0)
    hi = lo + delta
    manifest = json.loads((args.raw / "manifest.json").read_text())
    entries = [e for e in manifest["episodes"] if e["accepted"]]
    assert len(dataset) == sum(e["frames"] for e in entries)
    all_errors, by_kind, by_phase, initial = [], {}, {}, []
    global_index = 0
    max_collator_difference = 0.
    for entry in entries:
        folder = args.raw / f"episode_{entry['episode']:04d}"
        with np.load(folder / "trajectory.npz") as raw:
            states, actions, phases = raw["state"], raw["action"], raw["phase"]
            # Check an actual exported sample at each episode's first frame.
            item = processed[global_index]
            expected = actions[:args.prefix_steps, 7:10] - states[0, 7:10]
            np.testing.assert_allclose(item["action"][:args.prefix_steps, offset:offset + 3],
                                       expected, atol=1e-6, rtol=0)
            normalized = collator([item])["action_chunk"][0, :args.prefix_steps, offset:offset + 3].numpy()
            decoded = (normalized + 1) / 2 * delta + lo
            difference = float(np.abs(decoded - np.clip(expected, lo, hi)).max())
            max_collator_difference = max(max_collator_difference, difference)
            assert difference < 1e-6
            for frame in range(len(states)):
                target = actions[frame:frame + args.prefix_steps, 7:10] - states[frame, 7:10]
                error = np.linalg.norm(target - np.clip(target, lo, hi), axis=1) * 1000
                all_errors.extend(error.tolist())
                by_kind.setdefault(entry["kind"], []).extend(error.tolist())
                by_phase.setdefault(str(phases[frame]), []).extend(error.tolist())
                if frame == 0:
                    initial.append({"episode": entry["episode"], "kind": entry["kind"],
                                    "max_position_clip_error_mm": float(error.max())})
            global_index += len(states)

    def summarize(errors):
        return {"recorded_targets": len(errors),
                "targets_above_0_1mm_error": sum(e > .1 for e in errors),
                "max_position_clip_error_mm": max(errors),
                "mean_position_clip_error_mm": float(np.mean(errors))}

    report = {"official_collator_roundtrip_verified": True,
        "episodes_checked": len(entries), "first_actions_per_chunk": args.prefix_steps,
        "right_xyz_layout_offset": offset,
        "max_official_vs_formula_error_m": max_collator_difference,
        "norm_stats_sha256": hashlib.sha256(Path(flat["norm_stats_path"]).read_bytes()).hexdigest(),
        "overall": summarize(all_errors),
        "by_kind": {k: summarize(v) for k, v in by_kind.items()},
        "by_observed_phase": {k: summarize(v) for k, v in by_phase.items()},
        "initial_frames": initial,
        "scope": "Right position only; all accepted training-pool frames, existing future targets; no external data; no model or physics execution"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                     ["official_collator_roundtrip_verified", "episodes_checked", "overall"]}))
    if args.max_clip_error_mm is not None:
        assert report["overall"]["max_position_clip_error_mm"] <= args.max_clip_error_mm, (
            "Right-XYZ action label clipping exceeds the required round-trip tolerance")


if __name__ == "__main__":
    main()
