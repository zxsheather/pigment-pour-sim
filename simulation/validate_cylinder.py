"""Validate contact pickup, RGB readings, and intentional observation failures."""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from carry_env import CarryEnv,WaypointTeacher
from cylinder_reader import read_volume
from inspect_cylinder import CylinderEnv,run_inspection


def evaluate_case(job):
    case,output,grip_profile=job
    output.mkdir()
    env=CylinderEnv(case["volume"],image_size=768,grip_profile=grip_profile,
                    friction_scale=case["friction_scale"],mass_scale=case["mass_scale"])
    try:
        with (output/"run.log").open("w") as log,contextlib.redirect_stdout(log):
            result=run_inspection(env,output,seed=case["seed"],jitter=case["jitter"],yaw_jitter=case["yaw_jitter"])
        return dict(case,volume_ml=case["volume"],success=result["success"],
                    reason=result["failure_reason"],reading=result["reading"],
                    error_ml=result.get("absolute_reading_error_ml"),
                    final_tilt_rad=result["final_tilt_rad"],max_slip_m=result["max_slip_m"],
                    max_held_tilt_rad=result["max_held_tilt_rad"],
                    max_contact_loss_seconds=result["max_contact_loss_seconds"],
                    simulated_seconds=result["simulated_seconds"])
    finally:env.close()


def release_saved_episode(case,output,grip_profile):
    env=CylinderEnv(case["volume"],image_size=768,grip_profile=grip_profile,
                    friction_scale=case["friction_scale"],mass_scale=case["mass_scale"])
    try:
        env.reset_inspection(seed=case["seed"],jitter=case["jitter"],yaw_jitter=case["yaw_jitter"])
        with np.load(output/"final_physics.npz") as state:
            env.data.qpos[:]=state["qpos"];env.data.qvel[:]=state["qvel"]
            env.data.ctrl[:]=state["ctrl"];env.data.time=float(state["time"])
        env.hold_targets={a:float(env.data.ctrl[a]) for a in env.position_actuators}
        mujoco.mj_forward(env.model,env.data)
        before=float(env.data.xpos[env.cylinder_body,2])
        q=env.data.qpos[env.arm_addresses].copy()
        for _ in range(30):env.step_joints(q,opening=.045)
        dropped=before-float(env.data.xpos[env.cylinder_body,2]);contacts=env.metrics()["grip_contacts"]
        return dict(drop_m=dropped,contacts=contacts,passed=dropped>.05 and contacts==0,
                    initialization="saved final qpos/qvel/ctrl")
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--suite",choices=("original","heldout","fresh"),default="original")
    parser.add_argument("--grip-profile",choices=("official","flat","v-pad"),default="official")
    parser.add_argument("--workers",type=int,default=1,choices=range(1,9))
    parser.add_argument("--seed-offset",type=int,default=0,help="Use new position/yaw samples without changing the matrix")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[];negative={};reference=None
    if args.suite=="original":
        cases=[dict(volume=volume,seed=seed,jitter=.004,yaw_jitter=0.,friction_scale=1.,mass_scale=1.)
               for volume in (8,15,25,35) for seed in (100,101)]
    elif args.suite=="heldout":
        # Separate seeds, unseen fill levels, wider pose and material/load
        # perturbations. These cases are fixed before running the optimized model.
        cases=[dict(volume=volume,seed=200+index*4+trial,jitter=.008,yaw_jitter=np.deg2rad(8),
                    friction_scale=(.8,.9,1.1,1.2)[trial],mass_scale=(1.2,.9,1.1,.8)[trial])
               for index,volume in enumerate((5,12,22,32,40)) for trial in range(4)]
    else:
        # Fresh cases fixed after discovering the high-fill reading failure;
        # the former heldout matrix is now a regression set, not heldout data.
        cases=[dict(volume=volume,seed=300+index*3+trial,jitter=.010,yaw_jitter=np.deg2rad(10),
                    friction_scale=(.85,1.,1.15)[trial],mass_scale=(1.15,1.,.85)[trial])
               for index,volume in enumerate((6,18,28,40)) for trial in range(3)]
    for case in cases:case["seed"]+=args.seed_offset
    jobs=[(case,args.output/f"volume_{case['volume']}_seed_{case['seed']}",args.grip_profile) for case in cases]
    if args.workers==1:
        for job in jobs:
            result=evaluate_case(job);results.append(result);print(json.dumps(result),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(evaluate_case,jobs):
                results.append(result);print(json.dumps(result),flush=True)
    for case,result,job in zip(cases,results,jobs):
        if result["success"]:
            reference=imageio.imread(job[1]/"final_face.png")
            negative["release"]=release_saved_episode(case,job[1],args.grip_profile)
            break
    if reference is not None:
        negative["blank_image"]=read_volume(np.full_like(reference,255))
        hsv=cv2.cvtColor(reference,cv2.COLOR_RGB2HSV)
        masked=reference.copy()
        masked[(hsv[:,:,0]>=140)&(hsv[:,:,0]<=175)&(hsv[:,:,1]>90)]=255
        negative["missing_anchor"]=read_volume(masked)
        masked_liquid=reference.copy()
        masked_liquid[(hsv[:,:,0]>=95)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>90)]=255
        negative["missing_liquid"]=read_volume(masked_liquid)
        negative["rotated_image"]=read_volume(np.ascontiguousarray(np.rot90(reference)))
        assert all(not negative[k]["valid"] for k in ("blank_image","missing_anchor","missing_liquid","rotated_image"))
        assert negative["release"]["passed"]
    else:
        negative["cylinder_negative_checks_skipped"]={
            "reason":"No successful cylinder episode was available for release and image-negative checks"}
    # Check the shared model-builder hook leaves the old carry task usable.
    carry=CarryEnv()
    try:
        obs=carry.reset(seed=0,goal="B",start_jitter=.03);teacher=WaypointTeacher(carry)
        while not carry.done:obs,info=carry.step(teacher.action(obs))
        assert info["success"]
        negative["carry_regression"]={"success":info["success"],"seconds":info["simulated_seconds"]}
    finally:carry.close()
    report=dict(suite=args.suite,grip_profile=args.grip_profile,workers=args.workers,seed_offset=args.seed_offset,
                episodes=results,successes=sum(row["success"] for row in results),total=len(results),
                negative_checks=negative,
                maximum_slip_m=max(row["max_slip_m"] for row in results),
                maximum_reading_error_ml=max((row["error_ml"] for row in results if row["success"]),default=None))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="episodes"}),flush=True)
    if report["successes"]!=report["total"]:raise SystemExit(1)


if __name__=="__main__":main()
