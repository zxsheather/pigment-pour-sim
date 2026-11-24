"""Export successful Cartesian or reconstructed joint demonstrations to LeRobot v3."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import DEFAULT_IMAGE_PATH

from cylinder_joint_actions import (JOINT_ACTION_NAMES, JOINT_STATE_NAMES,
                                    joint_command_action, joint_observation,
                                    relative_joint_chunks, require)

CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_joint_reconstruction(episode, reconstruction, trajectory, frames):
    report = json.loads((reconstruction / 'summary.json').read_text())
    require(report['passed'] and report['source'] == str(episode.resolve())
            and report['frames_reconstructed'] == frames
            and report['trajectory_sha256'] == file_digest(episode / 'trajectory.npz'),
            'Joint reconstruction does not match the source trajectory')
    with np.load(reconstruction / 'robot_joint_observations.npz') as joints:
        state = joint_observation(joints['right_qpos'], joints['right_qvel'],
                                  trajectory['state'], trajectory['base_pose'])
    return state, joint_command_action(trajectory['joint_command'])


def share_immutable_video(source, target, expected):
    """Share verified bytes after LeRobot has finished writing."""
    require(file_digest(source) == expected and file_digest(target) == expected,
            f'Video bytes changed during append: {target}')
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        link = Path(temporary) / 'verified.mp4'
        os.link(source, link)
        link.replace(target)
    require(os.path.samefile(source, target), f'Video hard link failed: {target}')


class SourceVideoDataset(LeRobotDataset):
    """Keep validated source H.264 bitstreams instead of a second lossy encode.

    Frame decoding/add_frame still performs alignment validation and computes
    dataset statistics. Joint appends use one source file per episode.
    """
    keep_source_files_separate = False

    def _save_episode_video(self, video_key, episode_index, temp_path=None):
        if not self.keep_source_files_separate:
            return super()._save_episode_video(video_key, episode_index, temp_path)
        size_limit = self.meta.info['video_files_size_in_mb']
        # LeRobot opens a new file when the combined size reaches this limit.
        self.meta.info['video_files_size_in_mb'] = 0
        try:
            return super()._save_episode_video(video_key, episode_index, temp_path)
        finally:
            self.meta.info['video_files_size_in_mb'] = size_limit

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
    parser.add_argument("--base-dataset",type=Path,
                        help="Copy an immutable existing dataset, then append new successful episodes.")
    parser.add_argument("--include-base-odometry",action="store_true",
                        help="Append recorded robot base x/y/yaw to the 23D state; action stays 23D.")
    parser.add_argument('--joint-reconstruction', nargs='+', type=Path,
                        help='Validated joint replay directories in episode order for a joint dataset append.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    joint = args.joint_reconstruction is not None
    if joint and (not args.base_dataset or not args.reuse_source_videos or args.include_base_odometry
                  or len(args.joint_reconstruction) != len(args.episodes)):
        parser.error('Joint append requires a joint base, source-video reuse and one reconstruction per episode')
    if joint:
        base_report = json.loads((args.base_dataset / 'joint_export_validation.json').read_text())
        require(base_report['passed'], 'Expected a verified joint base dataset')
        norm = json.loads((args.base_dataset / 'joint_norm_stats.json').read_text())
        horizon = norm['provenance']['horizon']
        original_videos = {p.relative_to(args.base_dataset): file_digest(p)
                           for p in (args.base_dataset / 'videos').rglob('*.mp4')}
        require(bool(original_videos), 'No existing RGB videos')
    validated = []
    for episode_number, episode in enumerate(args.episodes):
        metadata = json.loads((episode / "metadata.json").read_text())
        if not metadata["result"]["success"]:
            raise ValueError(f"Refusing to train on failed demonstration: {episode}")
        if metadata.get("robot_type")=="quanta_x1_official_cylinder":
            if metadata["result"].get("grip_profile")!="official" or not metadata["result"].get("geometry_audit_passed"):
                raise ValueError("Cylinder training requires the locked official finger geometry")
        if set(metadata["recorded_cameras"]) != set(CAMERAS):
            raise ValueError(f"All three recorded cameras are required: {episode}")
        with np.load(episode / "trajectory.npz") as data:
            if joint:
                require(metadata['robot_type'] == 'quanta_x1_official_cylinder', 'Expected official cylinder episode')
                state, action = verify_joint_reconstruction(
                    episode, args.joint_reconstruction[episode_number], data, metadata['frames'])
                for name, values in [('observation.state', state),
                                     ('action', relative_joint_chunks(state, action, horizon))]:
                    bounds = norm['norm_stats'][name]
                    require(np.all(values >= np.asarray(bounds['q01']) - 1e-7)
                            and np.all(values <= np.asarray(bounds['q99']) + 1e-7),
                            f'Joint append would change the existing normalization contract: {episode}/{name}')
            else:
                state, action = data["state"].copy(), data["action"].copy()
            timestamps = data["timestamp"].copy()
            odometry=data["base_pose"].copy() if args.include_base_odometry else None
        count = metadata["frames"]
        if state.shape != (count, 18 if joint else 23) or action.shape != (count, 9 if joint else 23):
            raise ValueError(f"Unexpected vector dimensions: {episode}")
        if odometry is not None:
            if odometry.shape!=(count,3) or not np.isfinite(odometry).all():
                raise ValueError(f"Invalid robot odometry: {episode}")
            state=np.concatenate([state,odometry],axis=1)
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
    state_names=names+(["base_odometry_x","base_odometry_y","base_odometry_yaw"] if args.include_base_odometry else [])
    if joint:
        names, state_names = JOINT_ACTION_NAMES, JOINT_STATE_NAMES
    features = {"observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
                "action": {"dtype": "float32", "shape": (len(names),), "names": names}}
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": sample_image.shape,
            "names": ["height", "width", "channels"],
        }
    dataset_class = SourceVideoDataset if args.reuse_source_videos else LeRobotDataset
    if args.base_dataset:
        # Derived per-frame sampling arrays no longer match after append.
        # Keep the base export immutable and require fresh output weights.
        shutil.copytree(args.base_dataset,args.output,
            ignore=shutil.ignore_patterns("frame-sampling-weights*.npy","frame-sampling-weights*.json",
                                          "joint_export_validation.json") if joint else
                   shutil.ignore_patterns("frame-sampling-weights*.npy","frame-sampling-weights*.json"))
        dataset=dataset_class(args.repo_id,root=args.output,video_backend="pyav",vcodec="h264")
        if joint:
            dataset.keep_source_files_separate = True
        robot_type = 'quanta_x1_official_cylinder_joint_control' if joint else validated[0][1].get('robot_type','quanta_x1_planar_carry')
        assert dataset.meta.info['robot_type'] == robot_type
        for key,feature in features.items():
            assert dataset.features[key]["dtype"]==feature["dtype"]
            assert tuple(dataset.features[key]["shape"])==tuple(feature["shape"])
        assert dataset.fps==20
        dataset.start_image_writer(num_threads=3)
        provenance=json.loads((args.base_dataset/"demonstration_provenance.json").read_text())
        assert dataset.num_episodes==len(provenance)
        if joint:
            sources = [str(item[0].resolve()) for item in validated]
            require(len(set(sources)) == len(sources) and not set(sources) & {item['source'] for item in provenance},
                    'Repeated episode source')
        old_frames=dataset.num_frames
    else:
        dataset = dataset_class.create(repo_id=args.repo_id, root=args.output,
                                   fps=20, robot_type=validated[0][1].get("robot_type","quanta_x1_planar_carry"),
                                   features=features, use_videos=True, vcodec="h264",
                                   image_writer_threads=3)
        provenance=[];old_frames=0
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
                           "observation_state_fields":state_names,
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
    assert reopened.num_episodes==len(provenance)
    assert reopened.num_frames==old_frames+sum(len(item[2]) for item in validated)
    assert len(reopened.hf_dataset)==reopened.num_frames
    item = reopened[0]
    assert item["observation.state"].shape == (len(state_names),)
    assert item["action"].shape == (len(names),)
    for camera in CAMERAS:
        assert item[f"observation.images.{camera}"].shape[0] == 3
    if joint:
        offset = old_frames
        for _, _, states, actions in validated:
            rows = reopened.hf_dataset[offset:offset + len(states)]
            require(np.array_equal(np.asarray(rows['index']), np.arange(offset, offset + len(states)))
                    and np.array_equal(np.stack(rows['observation.state']), states)
                    and np.array_equal(np.stack(rows['action']), actions),
                    'Appended frames differ from reconstructed joint vectors or indices')
            for index in (offset, offset + len(states) - 1):
                item = reopened[index]
                for camera in CAMERAS:
                    require(item[f'observation.images.{camera}'].shape == (3, *sample_image.shape[:2]),
                            'Appended sample has invalid RGB layout')
            offset += len(states)
        shared = []
        for relative, expected in original_videos.items():
            source = args.base_dataset / relative
            share_immutable_video(source, args.output / relative, expected)
            shared.append(dict(path=str(relative), sha256=expected, bytes=source.stat().st_size))
        shared_sources = []
        used_paths = set(original_videos)
        for index, record in enumerate(provenance[-len(validated):], start=len(provenance)-len(validated)):
            for camera in CAMERAS:
                key = f'observation.images.{camera}'
                relative = reopened.meta.get_video_file_path(index, key)
                require(relative not in used_paths, f'Expected a separate source video for {index}/{camera}')
                used_paths.add(relative)
                source = Path(record['source']) / f"{record['metadata'].get('video_prefix', '')}{camera}.mp4"
                sha = record['source_video_sha256'][key]
                share_immutable_video(source, args.output / relative, sha)
                shared_sources.append(dict(episode_index=index, path=str(relative), source=str(source),
                                           sha256=sha, bytes=source.stat().st_size))
        norm_path = args.output / 'joint_norm_stats.json'
        require(file_digest(norm_path) == file_digest(args.base_dataset / norm_path.name), 'Normalization changed')
        report = dict(kind='verified_joint_dataset_append', passed=True,
                      dataset=str(args.output.resolve()), base_dataset=str(args.base_dataset.resolve()),
                      episodes=reopened.num_episodes, frames=reopened.num_frames,
                      robot_state_dimensions=18, action_dimensions=9,
                      episode_provenance=provenance[-len(validated):],
                      base_frame_count=old_frames, shared_existing_videos=shared,
                      shared_appended_source_videos=shared_sources,
                      fixed_normalizer_sha256=file_digest(norm_path),
                      fixed_normalizer_covers_appended_labels=True)
        (args.output / 'joint_export_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({"episodes": reopened.num_episodes, "frames": reopened.num_frames,
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
