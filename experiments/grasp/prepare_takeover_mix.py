"""Combine accepted demonstrations while keeping source-scene families together.

Corrections precede the original training scenes; the same four original
internal validation scenes remain at the end. Original files are read-only.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    original = read(args.original / "manifest.json")
    corrections = read(args.corrections / "manifest.json")
    base = [e for e in original["episodes"] if e["accepted"] and e["numerically_valid"]]
    added = [e for e in corrections["episodes"] if e["accepted"] and e["numerically_valid"]]
    assert len(base) == 32 and len(added) == 9
    original_training = base[:28]
    original_validation = base[28:]
    training_groups = {(args.original.name, e["episode"]) for e in original_training}
    validation_groups = {(args.original.name, e["episode"]) for e in original_validation}
    assert not training_groups & validation_groups
    assert all((e["source_batch"], e["source_episode"]) in training_groups for e in added)
    assert corrections["source_split"] == "train_episode only"
    assert not corrections["setup_commands_in_supervision"]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ([(args.corrections, e, "correction", "train") for e in added]
               + [(args.original, e, "original", "train") for e in original_training]
               + [(args.original, e, "original", "internal_validation") for e in original_validation])
    manifest = {"controller": "privileged mixed demonstration teachers", "fps": 25,
        "model_executed": False, "observation_width": 20, "action_width": 20,
        "feature_order": original["feature_order"], "object_pose_in_observation": False,
        "grasp_weld": False, "setup_commands_in_supervision": False,
        "original_episodes": 32, "corrective_episodes": 9,
        "source_family_split": True, "episodes": [], "accepted_episodes": 41}
    groups = []
    for index, (root, entry, role, split) in enumerate(sources):
        source = root / f"episode_{entry['episode']:04d}"
        folder = args.output / f"episode_{index:04d}"
        folder.mkdir()
        for path in source.rglob("*"):
            if not path.is_file() or path.name == "episode.json":
                continue
            destination = folder / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, destination)
            except OSError:
                shutil.copyfile(path, destination)
        copied = copy.deepcopy(entry)
        copied.update(episode=index, mix_role=role, mix_split=split,
                      raw_source_batch=root.name, raw_source_episode=entry["episode"])
        family = ((entry["source_batch"], entry["source_episode"]) if role == "correction"
                  else (root.name, entry["episode"]))
        copied["source_scene_family"] = list(family)
        (folder / "episode.json").write_text(json.dumps(copied, indent=2))
        manifest["episodes"].append(copied)
        groups.append({"episode": index, "role": role, "split": split,
                       "source_family": list(family), "frames": entry["frames"]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {"episodes": 41, "frames": sum(e["frames"] for e in manifest["episodes"]),
        "corrective_episodes": 9, "corrective_frames": sum(e["frames"] for e in added),
        "training_episode_ids": list(range(37)), "validation_episode_ids": list(range(37, 41)),
        "corrective_episode_ids": list(range(9)),
        "training_episode_fraction": 37 / 41,
        "original_training_groups": [list(g) for g in sorted(training_groups)],
        "validation_groups": [list(g) for g in sorted(validation_groups)],
        "no_source_family_overlap": True,
        "source_manifest_sha256": {root.name: hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
                                   for root in [args.original, args.corrections]},
        "episodes_by_group": groups}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ["episodes_by_group", "original_training_groups", "validation_groups"]}))


if __name__ == "__main__":
    main()
