"""Validate joint-state reconstruction for every source in an existing export.

Writes small robot-only sidecars. It does not copy videos, modify the dataset,
train a policy, or count privileged replay as a learned task success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    provenance_path = dataset / 'demonstration_provenance.json'
    info_path = dataset / 'meta/info.json'
    provenance_bytes = provenance_path.read_bytes()
    info_bytes = info_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    info = json.loads(info_bytes)
    require(len(provenance) == info['total_episodes'], 'Episode count mismatch')
    require(sum(demo['metadata']['frames'] for demo in provenance)
            == info['total_frames'], 'Source frame count mismatch')
    sources = [Path(demo['source']).resolve() for demo in provenance]
    require(len(set(sources)) == len(sources), 'Duplicate source episode')
    require(not args.output.exists(), 'Output exists')
    # Validate original inputs before creating any output or starting replay.
    for source, demo in zip(sources, provenance):
        require(json.loads((source / 'metadata.json').read_text())
                == demo['metadata'], f'Source metadata changed: {source}')
        require(sha256(source / 'trajectory.npz') == demo['trajectory_sha256'],
                f'Source trajectory changed: {source}')
    args.output.mkdir(parents=True)
    script = Path(__file__).with_name('reconstruct_cylinder_joints.py').resolve()
    started = time.time()
    progress = dict(
        kind='dataset_joint_observation_reconstruction', dataset=str(dataset),
        expected_episodes=len(provenance), expected_frames=info['total_frames'],
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        reconstruction_script_sha256=sha256(script), started=started,
        episodes=[], current_episode=None, current_child_pid=None,
        videos_copied=False, dataset_modified=False, model_training_started=False,
        learned_task_success_measured=False, complete=False, passed=False)

    def save():
        (args.output / 'progress.json').write_text(
            json.dumps(progress, indent=2) + '\n')

    for index, (source, demo) in enumerate(zip(sources, provenance)):
        target = args.output / f'episode_{index:03d}'
        argv = [sys.executable, str(script), '--episode', str(source),
                '--output', str(target)]
        with (args.output / f'episode_{index:03d}.log').open('w') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            progress.update(current_episode=index, current_child_pid=child.pid)
            save()
            code = child.wait()
        report_path = target / 'summary.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        passed = (code == 0 and report.get('passed') is True
                  and report.get('frames_reconstructed') == demo['metadata']['frames']
                  and report.get('trajectory_sha256') == demo['trajectory_sha256']
                  and report.get('source_sha256') == progress['reconstruction_script_sha256'])
        progress['episodes'].append(dict(
            episode_index=index, source=str(source), exit_code=code,
            summary=str(report_path.resolve()), passed=passed,
            frames_reconstructed=report.get('frames_reconstructed', 0),
            robot_state23_max_difference=report.get('robot_state23_max_difference'),
            object_position_max_difference=report.get('object_position_max_difference'),
            object_quaternion_max_difference=report.get('object_quaternion_max_difference')))
        progress.update(current_child_pid=None)
        save()
        print(json.dumps(progress['episodes'][-1]), flush=True)
    inputs_unchanged = (provenance_path.read_bytes() == provenance_bytes
                        and info_path.read_bytes() == info_bytes)
    progress.update(
        current_episode=None, complete=True, finished=time.time(),
        input_provenance_and_info_unchanged=inputs_unchanged,
        reconstructed_frames=sum(row['frames_reconstructed']
                                 for row in progress['episodes']),
        passed=inputs_unchanged and all(row['passed'] for row in progress['episodes']))
    save()
    (args.output / 'summary.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps({key: value for key, value in progress.items()
                      if key != 'episodes'}), flush=True)
    if not progress['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
