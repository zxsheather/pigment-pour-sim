"""Replay saved Cartesian labels through the same generic robot controller.

This validates the data/execution contract, not a learned policy. Saved object
poses and phases are used only for comparison and grasp-reference scoring.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_gripper import audit
from cylinder_actions import cartesian_to_joint_target
from cylinder_reader import read_volume
from inspect_cylinder import CylinderEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--controller-config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.episode/"summary.json").read_text())
    controller=json.loads(a.controller_config.read_text())
    data=np.load(a.episode/"trajectory.npz")
    env=CylinderEnv(source["expected_volume_ml"],image_size=768,
                    friction_scale=source["friction_scale"],mass_scale=source["mass_scale"],
                    **controller["environment"])
    largest_state=0.;largest_object=0.;slip=0.;forbidden=[]
    try:
        geometry=audit(env)
        env.reset_inspection(seed=source["seed"],jitter=source["position_jitter_m"],
                             yaw_jitter=source["yaw_jitter_rad"])
        held=False
        for index,action in enumerate(data["action"]):
            largest_state=max(largest_state,float(abs(env.state_vector()-data["state"][index]).max()))
            largest_object=max(largest_object,float(np.linalg.norm(
                env.data.xpos[env.cylinder_body]-data["cylinder_position"][index])))
            if not held and data["phase"][index]=="lift":
                env.grasp_point_reference=env.grip_point_relative();held=True
            joints,opening,forward,yaw=cartesian_to_joint_target(env,action)
            _,metrics=env.step_joints(joints,opening=opening,forward=forward,yaw=yaw)
            if held:slip=max(slip,metrics["bottle_slip_m"])
            if metrics["forbidden_contacts"]:forbidden.append(dict(frame=index,contacts=metrics["forbidden_contacts"]))
        rgb=env.render("face_view");reading=read_volume(rgb)
        imageio.imwrite(a.output/"final_face.png",rgb)
        # Tight trajectory agreement checks whether labels can reproduce the
        # demonstrated physical path. The replay uses no phase motion planner.
        report=dict(policy="saved_teacher_cartesian_replay",geometry_audit=geometry,
                    frames=len(data["action"]),max_robot_state_deviation=largest_state,
                    max_object_position_deviation_m=largest_object,max_slip_m=slip,
                    forbidden_contacts=forbidden,reading=reading,
                    expected_volume_ml=source["expected_volume_ml"],
                    passed=largest_object<.002 and largest_state<.01 and not forbidden
                           and reading["valid"] and abs(reading["volume_ml"]-source["expected_volume_ml"])<=.5)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report),flush=True)
        if not report["passed"]:raise SystemExit(1)
    finally:env.close()


if __name__=="__main__":main()
