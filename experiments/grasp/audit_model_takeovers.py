"""Audit teacher-only takeover labels and their exported first-frame samples."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image

from audit_expert_data import audit_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/pigment_takeover_precontact_v5")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["source_split"] == "train_episode only"
    assert not manifest["setup_commands_in_supervision"]
    assert not manifest["used_for_xyz_normalization_training"]
    assert manifest["model_prefix_before_recording"]
    entries = manifest["episodes"]
    assert all(e["source_split"] == "train_episode" and
               max(e["prefix_state_error"], e["prefix_bottle_error"]) < 1e-6 for e in entries)
    accepted = [e for e in entries if e["accepted"]]
    paths, positions = audit_batches([args.input])
    assert len(paths) == len(accepted) > 0
    dataset = LeRobotDataset(args.repo_id, root=args.dataset, video_backend="pyav")
    assert len(dataset) == sum(e["frames"] for e in accepted)
    expected_features = {"observation.state", "action", "observation.images.face_view",
                         "observation.images.right_wrist_view", "timestamp", "frame_index",
                         "episode_index", "index", "task_index"}
    assert set(dataset.features) <= expected_features
    groups, index = [], 0
    for exported_index, (entry, folder) in enumerate(zip(accepted, paths)):
        assert entry["max_held_seconds"] >= .9999
        assert entry["max_bottle_bottom_lift_m"] >= .05
        with np.load(folder / "trajectory.npz") as trajectory:
            np.testing.assert_array_equal(trajectory["state"][0], entry["initial_robot_state"])
            row = dataset[index]
            np.testing.assert_array_equal(row["observation.state"].numpy(), trajectory["state"][0])
            np.testing.assert_array_equal(row["action"].numpy(), trajectory["action"][0])
        for camera in ["face_view", "right_wrist_view"]:
            rgb = np.asarray(Image.open(folder / camera / "000000.jpg").convert("RGB"))
            np.testing.assert_allclose(row["observation.images." + camera].numpy(),
                                       rgb.transpose(2, 0, 1) / 255., atol=1e-6)
        groups.append({"exported_episode": exported_index, "source_scene_group": entry["source_episode"],
                       "source_batch": entry["source_batch"], "prefix_steps": entry["prefix_steps"],
                       "raw_episode": entry["episode"]})
        index += entry["frames"]
    report = {"passed": True, "attempts": len(entries), "accepted": len(accepted),
        "frames": len(dataset), "source_scene_groups": len(positions),
        "all_prefixes_replayed_exactly": True, "teacher_labels_only": True,
        "object_truth_excluded_from_features": True, "all_exported_first_frames_match": True,
        "not_used_for_current_training": True,
        "future_validation_must_group_by_source_scene": True,
        "group_mapping": groups}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "group_mapping"}))


if __name__ == "__main__":
    main()
