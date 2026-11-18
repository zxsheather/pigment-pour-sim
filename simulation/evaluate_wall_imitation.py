"""Diagnose Wall action errors on saved teacher observations by task phase.

This is an offline imitation diagnostic, never a physical task success score.
Phase and teacher actions are used for selecting/scoring samples only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.dataset as arrow
import torch

from cylinder_actions import relative_to_absolute,rotation_from_rows
from cylinder_joint_actions import APERTURE_SCALE_M, absolute_joint_chunk, joint_command_action, require
from run_wall_cylinder import CylinderWallPolicy


def load_joint_episode(dataset, episode, data):
    provenance = json.loads((dataset / 'demonstration_provenance.json').read_text())
    matches = [(i, row) for i, row in enumerate(provenance)
               if Path(row['source']).resolve() == episode.resolve()]
    require(len(matches) == 1, 'Expected exactly one matching episode in joint dataset')
    episode_index, source = matches[0]
    require(hashlib.sha256((episode / 'trajectory.npz').read_bytes()).hexdigest() == source['trajectory_sha256'],
            'Teacher trajectory differs from dataset provenance')
    frames = arrow.dataset(dataset / 'data', format='parquet').to_table(
        filter=arrow.field('episode_index') == episode_index,
        columns=['frame_index', 'timestamp', 'observation.state', 'action']).sort_by('frame_index')
    count = len(data['state'])
    require(np.array_equal(np.asarray(frames['frame_index']), np.arange(count))
            and np.allclose(np.asarray(frames['timestamp']), data['timestamp'], rtol=0, atol=3e-6),
            'Joint dataset frames differ from teacher timeline')
    state = np.asarray(frames['observation.state'].to_pylist(), dtype=np.float32)
    action = np.asarray(frames['action'].to_pylist(), dtype=np.float32)
    require(state.shape == (count, 18) and np.isfinite(state).all()
            and np.array_equal(action, joint_command_action(data['joint_command'])),
            'Expected verified 18D observations and bounded teacher commands')
    return state, action


def joint_errors(absolute, expected, prefix):
    delta = absolute - expected
    arm_error = np.abs(delta[:, :6])
    return dict(prefix_frames=prefix,
                mean_arm_error_first8_rad=float(arm_error[:prefix].mean()),
                max_arm_error_first8_rad=float(arm_error[:prefix].max()),
                mean_arm_error_horizon_rad=float(arm_error.mean()),
                arm_mean_signed_error_first8_rad=delta[:prefix, :6].mean(0).tolist(),
                gripper_width_error_first8_mm=float(np.abs(delta[:prefix, 6]).mean() * APERTURE_SCALE_M * 1000),
                gripper_width_mean_signed_error_first8_mm=float(delta[:prefix, 6].mean() * APERTURE_SCALE_M * 1000),
                base_vx_error_first8_m_s=float(np.abs(delta[:prefix, 7]).mean()),
                base_vx_mean_signed_error_first8_m_s=float(delta[:prefix, 7].mean()),
                first_predicted_right_action=absolute[0].tolist(),
                first_teacher_right_action=expected[0].tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--episode",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--norm-key",default="cylinder_official_v2")
    p.add_argument("--flow-steps",type=int,default=10,choices=range(1,101))
    p.add_argument('--joint-dataset',type=Path,help='Export containing verified joint states for this teacher episode')
    p.add_argument('--frame',type=int,action='append',default=[],
                   help='Explicit teacher frame to evaluate. Default uses the midpoint of each phase.')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    meta=json.loads((a.episode/"metadata.json").read_text())
    data=np.load(a.episode/"trajectory.npz")
    phases=list(dict.fromkeys(data["phase"].tolist()))
    indices=a.frame or [int(f[len(f)//2]) for f in (np.flatnonzero(data['phase']==phase) for phase in phases)]
    require(len(set(indices))==len(indices) and all(0<=i<len(data['state']) for i in indices),
            'Teacher frame indices must be unique and in range')
    if a.joint_dataset:
        joint_states, joint_actions = load_joint_episode(a.joint_dataset,a.episode,data)
    images={i:{} for i in indices}
    for camera in meta["recorded_cameras"]:
        with av.open(str(a.episode/f"{meta.get('video_prefix','')}{camera}.mp4")) as video:
            for index,frame in enumerate(video.decode(video=0)):
                if index in images:
                    require(abs(float(frame.time)-float(data['timestamp'][index]))<1e-4,
                            f'Teacher video timestamp differs at {index}/{camera}')
                    images[index][camera]=frame.to_ndarray(format="rgb24")
    require(all(set(img)==set(meta['recorded_cameras']) for img in images.values()),
            'Missing decoded teacher images')
    torch.manual_seed(47);policy=CylinderWallPolicy(a.model,a.config,a.norm_key,flow_steps=a.flow_steps)
    require((policy.action_space=='joint')==bool(a.joint_dataset),
            'Exactly the joint policy requires --joint-dataset')
    results=[]
    for index in indices:
        if policy.action_space=='joint':
            observation=dict(state=joint_states[index],images=images[index])
            prediction=policy.predict_many([observation],meta['instruction'])[0]
            absolute=absolute_joint_chunk(prediction,observation['state'])
            future=np.minimum(index+np.arange(len(absolute)),len(joint_actions)-1)
            row=dict(phase=str(data['phase'][index]),frame=index,time_s=float(data['timestamp'][index]),
                     **joint_errors(absolute,joint_actions[future],min(8,len(absolute))))
            results.append(row);print(json.dumps(row),flush=True)
            continue
        observation=dict(state=data["state"][index],images=images[index])
        if policy.with_odometry:
            observation["state"]=np.r_[observation["state"],data["base_pose"][index]].astype(np.float32)
        prediction=policy.predict_many([observation],meta["instruction"])[0]
        absolute=relative_to_absolute(prediction,observation["state"][:23])
        future=np.minimum(index+np.arange(len(absolute)),len(data["action"])-1)
        expected=data["action"][future]
        position=np.linalg.norm(absolute[:,10:13]-expected[:,10:13],axis=-1)
        angles=[]
        for pred,target in zip(absolute,expected):
            relative=rotation_from_rows(pred[13:19])@rotation_from_rows(target[13:19]).T
            angles.append(float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
        row=dict(phase=str(data["phase"][index]),frame=index,
                 mean_position_error_mm=float(position.mean()*1000),max_position_error_mm=float(position.max()*1000),
                 mean_rotation_error_deg=float(np.degrees(angles).mean()),
                 mean_gripper_width_error_mm=float(abs(absolute[:,19]-expected[:,19]).mean()*.0975*1000),
                 mean_base_vx_error_m_s=float(abs(absolute[:,20]-expected[:,20]).mean()),
                 first_predicted_right_action=absolute[0,10:23].tolist(),first_teacher_right_action=expected[0,10:23].tolist())
        results.append(row);print(json.dumps(row),flush=True)
    report=dict(kind="offline_teacher_observation_diagnostic",model=str(a.model.resolve()),
                episode=str(a.episode.resolve()),samples=results,flow_steps=a.flow_steps,
                action_space=policy.action_space,
                joint_dataset=str(a.joint_dataset.resolve()) if a.joint_dataset else None,
                physics_advanced=False,task_success_measured=False,
                note="Teacher observations only; does not measure policy-induced drift or task success.")
    a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
