"""Verify expert timing, supervision, dataset round trip and episode holdout."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image


def audit_batches(roots):
    episodes, offsets = [], set()
    for root in roots:
        manifest = json.loads((root/"manifest.json").read_text())
        assert not manifest["object_pose_in_observation"] and not manifest["grasp_weld"]
        for entry in manifest["episodes"]:
            if not entry["accepted"]:
                continue
            assert entry["numerically_valid"]
            folder = root/f"episode_{entry['episode']:04d}"
            with np.load(folder/"trajectory.npz") as a:
                states, actions, timestamps = a["state"], a["action"], a["timestamp_s"]
                assert states.shape == actions.shape == (entry["frames"], 20)
                assert np.isfinite(states).all() and np.isfinite(actions).all()
                assert abs(timestamps[0]) < 1e-8
                assert np.allclose(np.diff(timestamps), 1/manifest["fps"], atol=1e-8)
                assert ((actions[:, 13] >= 0) & (actions[:, 13] <= 4.5)).all()
                rows = json.loads((folder/"telemetry.json").read_text())
                np.testing.assert_allclose(states[1:, 7:14], [r["right_ee"] for r in rows[:-1]], atol=1e-6)
                np.testing.assert_allclose(actions[:, 7:14], [r["requested_pose"] for r in rows], atol=1e-6)
                np.testing.assert_allclose(timestamps[1:], [r["time_s"] for r in rows[:-1]], atol=1e-8)
                for cam in ["face_view", "right_wrist_view"]:
                    assert len(list((folder/cam).glob("*.jpg"))) == len(states)
            offsets.add(tuple(entry["offset_xy_m"]))
            episodes.append(folder)
    return episodes, offsets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, nargs="+", required=True)
    p.add_argument("--heldout", type=Path, nargs="+", required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--repo-id", default="local/pigment_grasp_train")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    train, train_xy = audit_batches(args.train)
    heldout, heldout_xy = audit_batches(args.heldout)
    assert not train_xy.intersection(heldout_xy), "Train/heldout positions overlap"
    report = {"passed": True, "train_episodes": len(train), "heldout_episodes": len(heldout),
              "distinct_positions": len(train_xy|heldout_xy), "timing": "before-action state paired with issued command",
              "object_pose_in_features": False, "image_state_action_round_trip": False}
    if args.dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(args.repo_id, root=args.dataset, video_backend="pyav")
        row = dataset[0]
        with np.load(train[0]/"trajectory.npz") as raw:
            np.testing.assert_array_equal(row["observation.state"].numpy(), raw["state"][0])
            np.testing.assert_array_equal(row["action"].numpy(), raw["action"][0])
        for cam in ["face_view", "right_wrist_view"]:
            rgb = np.asarray(Image.open(train[0]/cam/"000000.jpg").convert("RGB"))
            np.testing.assert_allclose(row[f"observation.images.{cam}"].numpy(), rgb.transpose(2, 0, 1)/255., atol=1e-6)
        report["image_state_action_round_trip"] = True
        report["dataset_frames"] = len(dataset)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
