"""Collect state-triggered, physically validated corrective demonstrations."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")

import numpy as np
from PIL import Image

from collect_expert import FEATURE_ORDER,state_vector
from grasp_walloss import GraspEnvironment
from reactive_expert import ReactiveExpert,apply_setup,make_setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    p.add_argument("--scene",type=Path,default=root/"generated/blue_only/scene.xml")
    p.add_argument("--config",type=Path,default=root/"generated/blue_only/scene_config.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--seed",type=int,default=2040)
    p.add_argument("--radius",type=float,default=.02)
    p.add_argument("--modes",nargs="+",default=["clean","hand_offset","grasp_miss","slip"])
    p.add_argument("--max-seconds",type=float,default=8.)
    p.add_argument("--accepted-per-mode",type=int,help="Stop when every requested mode has this many accepted episodes; --episodes remains an attempt cap")
    p.add_argument("--no-center",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(args.seed)
    manifest={"controller":"privileged reactive expert","model_executed":False,"fps":25,
              "observation_width":20,"action_width":20,"feature_order":FEATURE_ORDER,
              "rotation":"extrinsic xyz Euler, radians",
              "action_semantics":"absolute arm-base grasp-center Cartesian targets; gripper motor units",
              "object_pose_in_observation":False,"grasp_weld":False,
              "phase_switch":"alignment geometry and bilateral contact; no fixed approach/close waits",
              "setup_commands_in_supervision":False,
              "teacher_source_sha256":hashlib.sha256(Path(__file__).with_name("reactive_expert.py").read_bytes()).hexdigest(),
              "target_accepted_per_mode":args.accepted_per_mode,
              "episodes":[],"accepted_episodes":0}
    for index in range(args.episodes):
        if args.accepted_per_mode:
            pending=[mode for mode in args.modes if sum(e["accepted"] and e["kind"]==mode for e in manifest["episodes"])<args.accepted_per_mode]
            if not pending:break
            kind=min(pending,key=lambda mode:sum(e["accepted"] and e["kind"]==mode for e in manifest["episodes"]))
        else:
            kind=args.modes[index%len(args.modes)]
        setup=make_setup(kind,rng)
        offset=np.zeros(2) if index==0 and not args.no_center else rng.uniform(-args.radius,args.radius,2)
        folder=args.output/f"episode_{index:04d}";folder.mkdir()
        tree=ET.parse(args.scene);body=tree.getroot().find("worldbody/body[@name='bottle_blue']")
        pos=np.fromstring(body.get("pos"),sep=" ");pos[:2]+=offset
        body.set("pos"," ".join(map(str,pos)));scene=folder/"input_scene.xml";tree.write(scene,encoding="unicode")
        env=GraspEnvironment(scene,args.config,folder,"blue",tcp="grasp",controller_label="REACTIVE EXPERT - NOT MODEL")
        env.step_time=.04
        entry={"episode":index,"seed":args.seed,"kind":kind,"setup":setup,
               "offset_xy_m":offset.tolist(),"accepted":False,"numerically_valid":False,
               "object_pose_for_expert_only":True}
        states=[];actions=[];times=[];phases=[]
        dirs={camera:folder/camera for camera in ["face_view","right_wrist_view"]}
        for d in dirs.values():d.mkdir()
        try:
            entry["initialization"]=apply_setup(env,setup)
            teacher=ReactiveExpert(env)
            for _ in range(round(args.max_seconds*25)):
                state,views=env.observation()
                before=state_vector(state)
                command,phase=teacher.command()
                action=before.copy();action[7:14]=command
                frame=len(states)
                for camera,array in views.items():Image.fromarray(array[0]).save(dirs[camera]/f"{frame:06d}.jpg",quality=90)
                states.append(before);actions.append(action);times.append(float(env.d.time-env.start_time));phases.append(phase)
                row=env.execute(command);row["expert_phase"]=phase;env.capture()
                if env.success:break
            entry.update(accepted=bool(env.success),numerically_valid=True,frames=len(states),
                         simulated_seconds=float(env.d.time-env.start_time),
                         max_bottle_bottom_lift_m=env.max_lift,bilateral_contact_steps=env.contact_frames,
                         max_held_seconds=max((r["held_seconds"] for r in env.telemetry),default=0.),
                         corrective_realignments=teacher.recoveries,
                         phase_counts={phase:phases.count(phase) for phase in sorted(set(phases))})
        except (RuntimeError,AssertionError) as error:
            entry["error"]=str(error)
        finally:
            if states:
                np.savez_compressed(folder/"trajectory.npz",state=np.asarray(states),action=np.asarray(actions),
                                    timestamp_s=np.asarray(times),phase=np.asarray(phases))
            env.capture("final");env.close()
        (folder/"episode.json").write_text(json.dumps(entry,indent=2))
        manifest["episodes"].append(entry);manifest["accepted_episodes"]=sum(e["accepted"] for e in manifest["episodes"])
        (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
        print(json.dumps(entry),flush=True)
        del env


if __name__=="__main__":main()
