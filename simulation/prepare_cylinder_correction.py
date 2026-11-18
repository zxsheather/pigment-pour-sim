"""Verify a teacher correction and preserve the policy's earlier RGB context."""

import argparse
import copy
import json
from pathlib import Path

import av
import imageio.v2 as imageio
import mujoco
import numpy as np

from carry_env import CAMERAS, CONTROL_DT
from cylinder_joint_actions import (
    ARM_SERVO_RATE_RAD_S, JOINT_SERVO_INCREMENT, bounded_joint_command, require,
    servo_increment_targets, with_joint_servo_state,
)
from cylinder_progress_prompts import CylinderProgressPrompts, PROGRESS_PROMPTS, PROGRESS_PROTOCOL
from cylinder_tray_vision import (
    TRAY_CENTER_CUE_PROTOCOL, TRAY_FALLBACK_VIEW, TRAY_VISION_CAMERAS, WRIST_TARGET_CUE_PROTOCOL,
    estimate_from_robot_views, observed_tray_cue_state, tray_grasp_height_m, with_wrist_target_state,
)
from export_carry_dataset import file_digest, verify_joint_reconstruction
from run_wall_cylinder import CylinderTrial


def verify_video(episode, metadata, count):
    require(set(metadata['recorded_cameras']) == set(CAMERAS), 'Expected all three robot camera recordings')
    sources = {}
    for camera in CAMERAS:
        path = episode / f"{metadata.get('video_prefix', '')}{camera}.mp4"
        with av.open(str(path)) as stream:
            video = stream.streams.video[0]
            require(video.codec_context.name == 'h264' and float(video.average_rate) == 20
                    and video.frames == count, 'Unexpected correction camera format')
            decoded = 0
            for i, frame in enumerate(stream.decode(video=0)):
                require(abs(float(frame.time) - i * CONTROL_DT) < 1e-4
                        and (frame.width, frame.height) == (448, 448), 'Correction camera time/shape mismatch')
                decoded += 1
            require(decoded == count, 'Correction video frame count differs from the trajectory')
        sources[camera] = dict(path=str(path), sha256=file_digest(path), frames=count)
    return sources


def restore_correction(trial, episode, prior, snapshot):
    with (np.load(prior/'recovery_physics.npz', allow_pickle=False) as original,
          np.load(episode/'starting_integration_state.npz', allow_pickle=False) as saved):
        np.testing.assert_array_equal(saved['state_spec'], original['state_spec'])
        for key, expected in (
            ('state', original['full_state'][snapshot]),
            ('hold_actuator_ids', original['hold_actuator_ids']),
            ('hold_targets', original['hold_targets'][snapshot]),
            ('applied_base_action', original['applied_base_action'][snapshot]),
        ):
            np.testing.assert_array_equal(saved[key], expected)
        env = trial.env
        mujoco.mj_setState(env.model, env.data, saved['state'], int(saved['state_spec']))
        env.hold_targets = dict(zip(saved['hold_actuator_ids'].tolist(), saved['hold_targets'].tolist(), strict=True))
        env.applied_action[:] = saved['applied_base_action']
        mujoco.mj_forward(env.model, env.data)


def prepare(episode, reconstruction, base, output):
    require(not output.exists(), 'Output exists')
    metadata = json.loads((episode/'metadata.json').read_text())
    result = metadata['result']
    provenance = result['initial_state_provenance']
    require(result['success'] and result['geometry_audit_passed'] and result['grip_profile'] == 'official'
            and result['grip_force_target_n'] is None and provenance['kind'] == 'expert_recovery_training_only'
            and not provenance['starts_holding'] and provenance['source_simulated_seconds'] > 0,
            'Expected a recorded official teacher correction from a live approach')
    prior = Path(provenance['policy_episode']).resolve()
    source = json.loads((prior/'summary.json').read_text())
    aggregate = json.loads((prior.parent/'summary.json').read_text())
    case = source['case']
    protocol = source['tray_cue_protocol']
    require(aggregate['evaluation_complete'] and not aggregate['probe_only']
            and source['progress_protocol'] == PROGRESS_PROTOCOL
            and source['joint_action_representation'] == JOINT_SERVO_INCREMENT
            and protocol in (TRAY_CENTER_CUE_PROTOCOL, WRIST_TARGET_CUE_PROTOCOL)
            and case['seed'] == result['seed'] and case['volume'] == result['expected_volume_ml'],
            'Unexpected policy collection or correction source')
    config = aggregate['controller']
    trial = CylinderTrial(case, config, output, action_space='joint', joint_servo_state=True,
                          progress_protocol=PROGRESS_PROTOCOL, servo_increments=True)
    env = trial.env
    try:
        require(trial.geometry_report['passed'] and trial.geometry_report['custom_pads'] == 0,
                'Official geometry audit failed')
        require(env.data.xpos[env.robot_root, 2] == 0., 'Canonical planar base calibration differs')
        names = set(source['initial_tray_images'])
        require(set(TRAY_VISION_CAMERAS) <= names <= {*TRAY_VISION_CAMERAS, TRAY_FALLBACK_VIEW},
                'Expected the source reset camera images')
        rgb_files = {name: prior/f'initial_tray_{name}.png' for name in sorted(names)}
        estimate = estimate_from_robot_views(env, {name:imageio.imread(p) for name,p in rgb_files.items()})
        require(estimate is not None and estimate == source['initial_tray_cue'],
                'Policy reset cue could not be reconstructed from its earlier images')
        reset_command = bounded_joint_command(env)
        np.testing.assert_array_equal(reset_command, source['progress_initial_joint_command'])
        router = CylinderProgressPrompts(trial.initial_base)
        snapshot = provenance['snapshot_index']
        with (np.load(prior/'trajectory.npz', allow_pickle=False) as t,
              np.load(prior/'model_observations.npz', allow_pickle=False) as observed):
            time = observed['timestamp'][snapshot]
            split = int(np.searchsorted(t['timestamp'], time))
            require(t['timestamp'][split] == time and time == provenance['source_simulated_seconds'],
                    'Model snapshot and teacher start timestamps differ')
            previous = np.vstack((reset_command, t['joint_command'][:split]))
            prefix = with_joint_servo_state(t['robot_joint_state'][:split+1], previous)
            for row in prefix[:-1]:
                router.observe(row)
            require(router.index == 0 and observed['instruction'][snapshot] == PROGRESS_PROMPTS[0],
                    'Expected recovery during approach')
            snapshot_state = observed['state'][snapshot]
        restore_correction(trial, episode, prior, snapshot)
        env.data.time = 0.
        first_command = bounded_joint_command(env)
        restore_router = copy.deepcopy(router)
        with np.load(episode/'trajectory.npz', allow_pickle=False) as raw:
            count = metadata['frames']
            require(metadata['fps'] == 20 and np.allclose(raw['timestamp'], np.arange(count)*CONTROL_DT),
                    'Invalid correction timing')
            joints, first_action = verify_joint_reconstruction(episode, reconstruction, raw, count)
            commands = raw['joint_command']
            previous = np.vstack((first_command, commands[:-1]))
            states = with_joint_servo_state(joints, previous)
            np.testing.assert_array_equal(observed_tray_cue_state(env, states[0], estimate, protocol), snapshot_state)
            tasks = np.asarray([router.observe(s) for s in states],dtype=np.int64)
            require(np.array_equal(np.unique(tasks), np.arange(len(PROGRESS_PROMPTS))), 'Correction misses a task phase')
            active = tasks == 0
            model_states = with_wrist_target_state(states, raw['state'], None, grasp_x_m=env.grasp_offset[0],
                                                    grasp_target_height_m=tray_grasp_height_m(env.config, env.grasp_height))
            model_states[active] = with_wrist_target_state(
                states[active], raw['state'][active], estimate, grasp_x_m=env.grasp_offset[0],
                grasp_target_height_m=tray_grasp_height_m(env.config, env.grasp_height))
            actions = first_action.copy()
            deltas = commands[:, :6] - previous[:, :6]
            require(np.abs(deltas).max() <= ARM_SERVO_RATE_RAD_S * CONTROL_DT + 1e-12, 'Correction exceeds servo rate')
            actions[:, :6] = deltas
            decoded = servo_increment_targets(actions, previous, CONTROL_DT)
            np.testing.assert_allclose(decoded[:, :6], commands[:, :6], atol=2e-9, rtol=0)

        bounds = json.loads((base/'joint_norm_stats.json').read_text())['norm_stats']
        for key, values in [('observation.state',model_states), ('action',actions)]:
            require(np.isfinite(values).all() and np.all(values >= np.asarray(bounds[key]['q01']) - 1e-7)
                    and np.all(values <= np.asarray(bounds[key]['q99']) + 1e-7), f'Correction exceeds {key} normalization')
        videos = verify_video(episode, metadata, count)
        trial.progress_router = restore_router
        trial.progress_changes = []
        trial.update_progress()
        matched_frames = 0
        for task, action in zip(tasks, actions, strict=True):
            require(trial.progress_router.index == task, 'Teacher replay and labeled prompt phases diverge')
            trial.step(action)
            matched_frames += 1
            if trial.done:
                break
        require(trial.success and trial.joint_limit_projected_frames == 0,
                f'Correction failed the full physical replay: {trial.reason}')
        report = dict(kind='certified_on_policy_wrist_target_teacher_correction', source=str(episode),
                      source_model_episode=str(prior), volume_ml=case['volume'], seed=case['seed'],
                      snapshot=snapshot, source_prefix_seconds=time, state_observations_exact_at_snapshot=True,
                      earlier_rgb_sha256={name:file_digest(path) for name,path in rgb_files.items()},
                      earlier_rgb_estimate=estimate, estimator_used_object_truth=False,
                      trajectory_sha256=file_digest(episode/'trajectory.npz'),
                      frames=count, approach_target_frames=int(active.sum()), physically_replayed_frames=matched_frames,
                      geometry_audit_passed=True, teacher_full_task_succeeded=True, model_task_success_measured=False,
                      max_slip_m=trial.max_slip, max_supported_lift_m=trial.max_lift,
                      reading_error_ml=abs(trial.good_readings[-1]-case['volume']),
                      alignment_video=videos, independent_cases_read=False)
        np.savez_compressed(output/'model_training_frames.npz', state=model_states, action=actions, task_index=tasks)
        (output/'certification.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--joint-reconstruction', type=Path, required=True)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.episode.resolve(), args.joint_reconstruction.resolve(),
                            args.base_dataset.resolve(), args.output.resolve())))


if __name__ == '__main__':
    main()
