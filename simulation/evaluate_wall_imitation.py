"""Diagnose Wall action errors on saved teacher observations by task phase.

This is an offline imitation diagnostic, never a physical task success score.
Phase and teacher actions are used for selecting/scoring samples only.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch

from cylinder_actions import relative_to_absolute,rotation_from_rows
from run_wall_cylinder import CylinderWallPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--episode",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--norm-key",default="cylinder_official_v2")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    meta=json.loads((a.episode/"metadata.json").read_text())
    data=np.load(a.episode/"trajectory.npz")
    phases=list(dict.fromkeys(data["phase"].tolist()));indices=[]
    for phase in phases:
        selected=np.flatnonzero(data["phase"]==phase)
        indices.append(int(selected[len(selected)//2]))
    images={i:{} for i in indices}
    for camera in meta["recorded_cameras"]:
        with av.open(str(a.episode/f"{meta.get('video_prefix','')}{camera}.mp4")) as video:
            for index,frame in enumerate(video.decode(video=0)):
                if index in images:images[index][camera]=frame.to_ndarray(format="rgb24")
    torch.manual_seed(47);policy=CylinderWallPolicy(a.model,a.config,a.norm_key)
    results=[]
    for index in indices:
        observation=dict(state=data["state"][index],images=images[index])
        prediction=policy.predict_many([observation],meta["instruction"])[0]
        absolute=relative_to_absolute(prediction,observation["state"])
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
                episode=str(a.episode.resolve()),samples=results,
                note="Teacher observations only; does not measure policy-induced drift or task success.")
    a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
