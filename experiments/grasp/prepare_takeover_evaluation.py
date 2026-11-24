"""Freeze paired correction-data evaluation cases before either model is tested."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    args = parser.parse_args()
    runs = args.runs
    output = runs / "takeover_training_eval_v6"
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"model": "Wall-OSS-0.5", "initial_checkpoint": "train_xyz_envelope_3000_v5/evaluation/adapted",
        "initial_weight_updates": 3000, "additional_weight_updates_each": 3000,
        "optimizer_restarted_for_both": True, "method": "action expert only, FP32 master weights, no LoRA",
        "control_pool_episodes": 32, "augmented_pool_episodes": 41,
        "corrective_episodes": 9, "corrective_frames": 495, "corrective_parent_training_scenes": 2,
        "same_normalizer_both_models": True, "checkpoint_selection": "fixed additional 3000 updates",
        "control_period_s": .04, "execute_steps": 8, "max_chunks": 48, "inference_seed": 0,
        "new_external_seeds": {"clean": 2061, "recovery": 2062},
        "external_data_used_for_training_or_statistics": False,
        "previous_external_results_are_development_evidence": True,
        "seen_prefix_recovery_is_in_sample_and_not_full_ready_pose_pickup": True,
        "slip_recovery_is_not_full_ready_pose_pickup": True}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    internal = read(runs / "reactive_eval_v4/internal_cases.json")
    assert len(internal) == 10
    internal += [{"split": "seen_prefix_recovery", "batch": "takeover_precontact_v5", "episode": e}
                 for e in [4, 13]]
    for case in internal[-2:]:
        meta = read(runs / case["batch"] / f"episode_{case['episode']:04d}/episode.json")
        assert meta["accepted"] and meta["prefix_steps"] == 24
    (output / "internal_cases.json").write_text(json.dumps(internal, indent=2))
    for name, seed, modes, quota, cap in [
        ("takeover_holdout_v6", 2061, ["clean"], 8, 32),
        ("takeover_recovery_holdout_v6", 2062, ["hand_offset", "grasp_miss", "slip"], 2, 32)]:
        with (output / (name + ".log")).open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name("collect_reactive_expert.py")),
                "--output", str(runs / name), "--seed", str(seed), "--no-center", "--episodes", str(cap),
                "--accepted-per-mode", str(quota), "--modes", *modes],
                check=True, stdout=log, stderr=subprocess.STDOUT)
    names = ["reactive_train_v4", "reactive_validation_v4", "reactive_holdout_v4",
             "reactive_recovery_holdout_v4", "xyz_holdout_v5", "xyz_recovery_holdout_v5",
             "takeover_holdout_v6", "takeover_recovery_holdout_v6"]
    accepted = {name: [e for e in read(runs / name / "manifest.json")["episodes"] if e["accepted"]]
                for name in names}
    positions = [{tuple(e["offset_xy_m"]) for e in accepted[name]} for name in names]
    assert all(not positions[i] & positions[j] for i in range(len(names)) for j in range(i + 1, len(names)))
    assert len(accepted[names[-2]]) == 8 and len(accepted[names[-1]]) == 6
    external = [{"split": "external_holdout", "batch": names[-2], "episode": e["episode"]}
                for e in accepted[names[-2]]]
    external += [{"split": "external_recovery_" + e["kind"], "batch": names[-1], "episode": e["episode"]}
                 for e in accepted[names[-1]]]
    (output / "external_cases.json").write_text(json.dumps(external, indent=2))
    (output / "cases_ready.json").write_text(json.dumps({"ready": True, "external_cases": len(external),
        "all_source_positions_disjoint": True, "new_seeds": [2061, 2062]}, indent=2))
    print("Prepared 12 internal probes and 14 paired external cases", flush=True)


if __name__ == "__main__":
    main()
