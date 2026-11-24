"""Export accepted expert episodes to a local LeRobot v3 image dataset.

This tool never uploads data. Object pose metadata is deliberately excluded
from the model features. Failed episodes are not used as action supervision.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image


NAMES = [f"{side}_{key}" for side in ["left", "right"] for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]] + ["vx", "vy", "vyaw", "lift", "head_pitch", "head_yaw"]
CAMERAS = ["face_view", "right_wrist_view"]


def export(inputs, output, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifests = [json.loads((root/"manifest.json").read_text()) for root in inputs]
    fps = manifests[0]["fps"]
    assert all(m["fps"] == fps for m in manifests)
    features = {
        "observation.state": {"dtype": "float32", "shape": (20,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (20,), "names": NAMES},
        **{f"observation.images.{cam}": {"dtype": "image", "shape": (448, 448, 3), "names": ["height", "width", "channels"]} for cam in CAMERAS},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, robot_type="quanta_x1_sim",
                                    fps=fps, features=features, use_videos=False, image_writer_threads=4)
    accepted = []
    for root, manifest in zip(inputs, manifests):
        for ep in manifest["episodes"]:
            if not ep["accepted"] or not ep["numerically_valid"]:
                continue
            folder = root/f"episode_{ep['episode']:04d}"
            data = np.load(folder/"trajectory.npz")
            assert data["state"].shape == data["action"].shape
            assert data["state"].shape[1] == 20
            assert np.allclose(np.diff(data["timestamp_s"]), 1/fps, atol=1e-8)
            for i, (state, action) in enumerate(zip(data["state"], data["action"])):
                frame = {"observation.state": state.astype(np.float32), "action": action.astype(np.float32),
                         "task": "Use your right hand to pick up the blue bottle from the tray and lift it."}
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = np.asarray(Image.open(folder/cam/f"{i:06d}.jpg").convert("RGB"))
                dataset.add_frame(frame)
            dataset.save_episode()
            accepted.append({"source_batch": root.name, "source_episode": ep["episode"], "frames": len(data["state"])})
    dataset.finalize()
    report = {"format": "LeRobot v3", "library_version": "0.4.3", "repo_id": repo_id,
              "episodes": accepted, "total_frames": sum(e["frames"] for e in accepted),
              "fps": fps, "object_pose_in_features": False, "uploaded": False}
    (output/"export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-id", default="local/pigment_grasp_expert")
    args = p.parse_args()
    export(args.input, args.output, args.repo_id)
