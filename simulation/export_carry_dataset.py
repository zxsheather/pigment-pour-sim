"""Export successful 23D Cartesian robot demonstrations to LeRobot v3."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import DEFAULT_IMAGE_PATH

CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")


class SourceVideoDataset(LeRobotDataset):
    """Keep validated source H.264 bitstreams instead of a second lossy encode.

    Frame decoding/add_frame still performs alignment validation and computes
    dataset statistics. Only the synchronous temporary-video hook is replaced.
    """
    def _encode_temporary_episode_video(self, video_key, episode_index):
        source = self.source_videos[video_key]
        target = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        shutil.copyfile(source, target)
        images = self.root / DEFAULT_IMAGE_PATH.format(
            image_key=video_key, episode_index=episode_index, frame_index=0)
        shutil.rmtree(images.parent)
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=Path, required=True,
                        help="Episode directories containing metadata.json, trajectory.npz and MP4s.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id",default="local/pigment-carry")
    parser.add_argument("--reuse-source-videos", action="store_true",
                        help="Copy validated source H.264 videos without a second lossy encoding.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    validated = []
    for episode in args.episodes:
        metadata = json.loads((episode / "metadata.json").read_text())
        if not metadata["result"]["success"]:
            raise ValueError(f"Refusing to train on failed demonstration: {episode}")
        if metadata.get("robot_type")=="quanta_x1_official_cylinder":
            if metadata["result"].get("grip_profile")!="official" or not metadata["result"].get("geometry_audit_passed"):
                raise ValueError("Cylinder training requires the locked official finger geometry")
        if set(metadata["recorded_cameras"]) != set(CAMERAS):
            raise ValueError(f"All three recorded cameras are required: {episode}")
        with np.load(episode / "trajectory.npz") as data:
            state, action = data["state"].copy(), data["action"].copy()
            timestamps = data["timestamp"].copy()
        count = metadata["frames"]
        if state.shape != (count, 23) or action.shape != state.shape:
            raise ValueError(f"Unexpected vector dimensions: {episode}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Nonfinite demonstration: {episode}")
        if metadata["fps"] != 20 or not np.allclose(timestamps, np.arange(count) / 20):
            raise ValueError(f"Unexpected sampling times: {episode}")
        for camera in CAMERAS:
            with av.open(str(episode / f"{metadata.get('video_prefix','')}{camera}.mp4")) as container:
                stream = container.streams.video[0]
                if stream.frames != count or float(stream.average_rate) != 20:
                    raise ValueError(f"Video/state alignment mismatch: {episode}/{camera}")
                if args.reuse_source_videos and stream.codec_context.name != "h264":
                    raise ValueError("Source-video reuse requires H.264 inputs")
        validated.append((episode, metadata, state, action))
    with av.open(str(validated[0][0] / f"{validated[0][1].get('video_prefix','')}face_view.mp4")) as container:
        sample_image = next(container.decode(video=0)).to_ndarray(format="rgb24")
    names = [f"{side}_{field}_{axis}" for side in ("left", "right")
             for field, axes in (("position", range(3)), ("rotation_row", range(6)),
                                 ("gripper", range(1))) for axis in axes]
    names += ["base_vx", "base_vy", "base_yaw_rate"]
    features = {"observation.state": {"dtype": "float32", "shape": (23,), "names": names},
                "action": {"dtype": "float32", "shape": (23,), "names": names}}
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": sample_image.shape,
            "names": ["height", "width", "channels"],
        }
    dataset_class = SourceVideoDataset if args.reuse_source_videos else LeRobotDataset
    dataset = dataset_class.create(repo_id=args.repo_id, root=args.output,
                                   fps=20, robot_type=validated[0][1].get("robot_type","quanta_x1_planar_carry"),
                                   features=features, use_videos=True, vcodec="h264",
                                   image_writer_threads=3)
    provenance = []
    for episode, metadata, states, actions in validated:
        source_videos = {f"observation.images.{camera}":
                         episode / f"{metadata.get('video_prefix','')}{camera}.mp4"
                         for camera in CAMERAS}
        if args.reuse_source_videos:
            dataset.source_videos = source_videos
        with ExitStack() as stack:
            streams = {camera: iter(stack.enter_context(av.open(str(episode / f"{metadata.get('video_prefix','')}{camera}.mp4")))
                                   .decode(video=0)) for camera in CAMERAS}
            for index, (state, action) in enumerate(zip(states, actions)):
                frame = {"observation.state": state.astype(np.float32),
                         "action": action.astype(np.float32), "task": metadata["instruction"]}
                for camera, stream in streams.items():
                    decoded = next(stream)
                    if abs(float(decoded.time) - index / 20) > 1e-4:
                        raise ValueError(f"Video timestamps do not align: {episode}/{camera}/{index}")
                    image = decoded.to_ndarray(format="rgb24")
                    if image.shape != sample_image.shape:
                        raise ValueError("All camera resolutions must match")
                    frame[f"observation.images.{camera}"] = image
                dataset.add_frame(frame)
            for camera, stream in streams.items():
                if next(stream, None) is not None:
                    raise ValueError(f"Extra video frames: {episode}/{camera}")
        dataset.save_episode(parallel_encoding=False)
        provenance.append({"source": str(episode.resolve()), "metadata": metadata,
                           "video_export": "source_h264_stream_copy" if args.reuse_source_videos else "lerobot_reencode",
                           "source_video_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                                   for key, path in source_videos.items()},
                           "trajectory_sha256": hashlib.sha256(
                               (episode / "trajectory.npz").read_bytes()).hexdigest()})
        print(f"Exported {episode}: {len(states)} frames", flush=True)
    dataset.finalize()
    (args.output / "demonstration_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Open the final artifact through the same reader used by Wall-X.
    reopened = LeRobotDataset(args.repo_id, root=args.output, video_backend="pyav")
    item = reopened[0]
    assert item["action"].shape == (23,)
    for camera in CAMERAS:
        assert item[f"observation.images.{camera}"].shape[0] == 3
    print(json.dumps({"episodes": reopened.num_episodes, "frames": reopened.num_frames,
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
