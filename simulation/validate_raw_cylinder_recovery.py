"""Validate raw recovery data without creating another training export.

Checks restored source state, all video frames and timestamps, robot-only
features, and coverage under the existing training normalization. Coverage is
reported rather than silently changing or clipping the stored labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import yaml

from compute_cylinder_norm_stats import relative_chunks
from validate_recovery_append import coverage
from cylinder_joint_actions import (joint_observation,joint_command_action,relative_joint_chunks,
    with_joint_servo_state,with_joint_wrist_cue_state)


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--joint-reconstruction',type=Path,
        help='Verified robot joint reconstruction, required for a joint-policy config')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite validation')
    metadata=json.loads((args.episode/'metadata.json').read_text())
    result=metadata['result'];provenance=result['initial_state_provenance']
    require(result['success'] and result['geometry_audit_passed'] and result['grip_profile']=='official',
            'Expected successful official-geometry recovery')
    require(provenance['kind']=='expert_recovery_training_only','Expected recovery provenance')
    geometry=json.loads((args.episode/'geometry_audit.json').read_text())
    require(geometry['passed'] and geometry['custom_pads']==0,'Geometry audit failed')
    cfg=yaml.safe_load(args.config.read_text());normpath=Path(cfg['data']['norm_stats_path'])
    joint='follow_right_arm_joint_position_relative' in cfg['task']['dof_config']
    require(joint==(args.joint_reconstruction is not None),
            'Joint configs require a verified joint reconstruction; Cartesian configs do not use it')
    source_episode=Path(provenance['policy_episode'])
    with np.load(args.episode/'starting_integration_state.npz') as start, np.load(
            source_episode/'recovery_physics.npz') as source:
        index=provenance['snapshot_index']
        require(int(start['state_spec'])==int(source['state_spec']),'State specification mismatch')
        for a,b in [('state',source['full_state'][index]),
                    ('hold_actuator_ids',source['hold_actuator_ids']),
                    ('hold_targets',source['hold_targets'][index]),
                    ('applied_base_action',source['applied_base_action'][index])]:
            require(np.array_equal(start[a],b),f'Restored source mismatch: {a}')
    source_observation_width = None
    with np.load(args.episode/'trajectory.npz') as data:
        count=metadata['frames']
        if joint:
            replay=json.loads((args.joint_reconstruction/'summary.json').read_text())
            require(replay['passed'] and replay['source']==str(args.episode.resolve())
                    and replay['frames_reconstructed']==count
                    and replay['trajectory_sha256']==hashlib.sha256((args.episode/'trajectory.npz').read_bytes()).hexdigest(),
                    'Joint reconstruction does not match this recovery')
            with np.load(args.joint_reconstruction/'robot_joint_observations.npz') as joints:
                states=joint_observation(joints['right_qpos'],joints['right_qvel'],data['state'],data['base_pose']).astype(np.float64)
            actions=joint_command_action(data['joint_command']).astype(np.float64)
            state_width,action_width=18,9
            with np.load(source_episode/'model_observations.npz') as observed:
                source_observation_width = observed['state'].shape[1]
                # The restored integration state contains the same joint data
                # even if the source policy used Cartesian observations.
                if source_observation_width == 18:
                    restored = states[0]
                elif source_observation_width == 23:
                    restored = data['state'][0]
                elif source_observation_width == 26:
                    summary_path = source_episode/'summary.json'
                    source_result = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                    if source_result.get('action_space') == 'joint':
                        from export_cylinder_joint_servo_view import initial_command

                        previous, _ = initial_command(args.episode, result, states[0])
                        restored = with_joint_servo_state(states[0], previous)
                        cue = source_result.get('initial_wrist_cue')
                        if cue is not None:
                            restored = with_joint_wrist_cue_state(restored, cue['world_xy_m'])
                    else:
                        restored = np.concatenate([data['state'][0], data['base_pose'][0]]).astype(np.float32)
                else:
                    raise ValueError(f'Unsupported policy observation width: {source_observation_width}')
                require(np.array_equal(restored, observed['state'][index]),
                        'Initial recorded policy observation changed during recovery')
                require(abs(float(observed['timestamp'][index]) - provenance['source_simulated_seconds']) < 1e-6,
                        'Recovery timestamp differs from the recorded policy observation')
        else:
            states=np.concatenate([data['state'],data['base_pose']],axis=1).astype(np.float32).astype(np.float64)
            actions=data['action'].astype(np.float32).astype(np.float64)
            state_width,action_width=26,23
        require(count>0 and states.shape==(count,state_width) and actions.shape==(count,action_width),'Feature shape mismatch')
        require(np.isfinite(states).all() and np.isfinite(actions).all(),'Nonfinite robot features')
        require(metadata['fps']==20 and np.allclose(data['timestamp'],np.arange(count)/20),'Numeric timestamps mismatch')
    videos={}
    cameras=('face_view','left_wrist_view','right_wrist_view')
    require(set(metadata['recorded_cameras'])==set(cameras),'Missing camera')
    for camera in cameras:
        path=args.episode/f"{metadata.get('video_prefix','')}{camera}.mp4"
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            require(stream.frames==count and float(stream.average_rate)==20,'Video metadata mismatch')
            decoded=0
            for i,frame in enumerate(container.decode(video=0)):
                require(abs(float(frame.time)-i/20)<1e-4,'Decoded video timestamps mismatch')
                require((frame.width,frame.height)==(448,448),'Video resolution mismatch')
                decoded+=1
            require(decoded==count,'Decoded video frame count mismatch')
        videos[camera]=dict(frames=count,width=448,height=448,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    norm=json.loads(normpath.read_text())['norm_stats'];horizon=cfg['task']['action_horizon_flow']
    chunks=(relative_joint_chunks(states,actions,horizon) if joint else relative_chunks(states,actions,horizon))
    require(np.isfinite(chunks).all(),'Nonfinite relative labels')
    report=dict(kind='raw_recovery_training_candidate_validation',episode=str(args.episode.resolve()),
        frames=count,robot_state_dimensions=state_width,action_dimensions=action_width,
        action_space='joint' if joint else 'cartesian',
        source_policy_robot_state_dimensions=source_observation_width,
        success_is_privileged_teacher_only=True,source_integration_state_and_controller_exactly_preserved=True,
        no_duplicate_export_created=True,training_not_modified_by_this_check=True,
        source_provenance=provenance,teacher_result=result,video_alignment=videos,
        trajectory_sha256=hashlib.sha256((args.episode/'trajectory.npz').read_bytes()).hexdigest(),
        norm_stats_sha256=hashlib.sha256(normpath.read_bytes()).hexdigest(),horizon=horizon,
        state_coverage=coverage(states,norm['observation.state']),
        horizon_action_coverage=coverage(chunks.reshape(-1,action_width),norm['action']),
        first8_action_coverage=coverage(chunks[:,:8].reshape(-1,action_width),norm['action']),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['data_integrity_passed']=True
    report['normalization_covers_all_labels']=(report['state_coverage']['maximum_bound_excess']==0
        and not any(report['horizon_action_coverage']['outside_fraction_per_dimension']))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(frames=count,teacher_success=True,horizon=horizon,
        normalization_covers_all_labels=report['normalization_covers_all_labels'],
        state_bound_excess=report['state_coverage']['maximum_bound_excess'],
        action_outside_fraction=report['horizon_action_coverage']['outside_fraction_per_dimension'])))


if __name__=='__main__':main()
