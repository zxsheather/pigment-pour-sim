"""Validate contact pickup, RGB readings, and intentional observation failures."""
import argparse
import contextlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from carry_env import CarryEnv,WaypointTeacher
from cylinder_reader import read_volume
from inspect_cylinder import CylinderEnv,run_inspection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[];negative={};reference=None
    for volume in (8,15,25,35):
        for seed in (100,101):
            output=args.output/f"volume_{volume}_seed_{seed}";output.mkdir()
            env=CylinderEnv(volume,image_size=768)
            try:
                with (output/"run.log").open("w") as log,contextlib.redirect_stdout(log):
                    result=run_inspection(env,output,seed=seed,jitter=.004)
                results.append(dict(volume_ml=volume,seed=seed,success=result["success"],
                                    reason=result["failure_reason"],reading=result["reading"],
                                    error_ml=result.get("absolute_reading_error_ml"),
                                    final_tilt_rad=result["final_tilt_rad"],max_slip_m=result["max_slip_m"]))
                if result["success"] and reference is None:
                    reference=imageio.imread(output/"final_face.png")
                    before=float(env.data.xpos[env.cylinder_body,2])
                    q=env.data.qpos[env.arm_addresses].copy()
                    for _ in range(30):env.step_joints(q,opening=.045)
                    dropped=before-float(env.data.xpos[env.cylinder_body,2])
                    contacts=env.metrics()["grip_contacts"]
                    negative["release"]={"drop_m":dropped,"contacts":contacts,"passed":dropped>.05 and contacts==0}
                print(json.dumps(results[-1]),flush=True)
            finally:env.close()
    if reference is not None:
        negative["blank_image"]=read_volume(np.full_like(reference,255))
        hsv=cv2.cvtColor(reference,cv2.COLOR_RGB2HSV)
        masked=reference.copy()
        masked[(hsv[:,:,0]>=140)&(hsv[:,:,0]<=175)&(hsv[:,:,1]>90)]=255
        negative["missing_anchor"]=read_volume(masked)
        negative["rotated_image"]=read_volume(np.ascontiguousarray(np.rot90(reference)))
        assert all(not negative[k]["valid"] for k in ("blank_image","missing_anchor","rotated_image"))
        assert negative["release"]["passed"]
    # Check the shared model-builder hook leaves the old carry task usable.
    carry=CarryEnv()
    try:
        obs=carry.reset(seed=0,goal="B",start_jitter=.03);teacher=WaypointTeacher(carry)
        while not carry.done:obs,info=carry.step(teacher.action(obs))
        assert info["success"]
        negative["carry_regression"]={"success":info["success"],"seconds":info["simulated_seconds"]}
    finally:carry.close()
    report=dict(episodes=results,successes=sum(row["success"] for row in results),total=len(results),
                negative_checks=negative,
                maximum_reading_error_ml=max((row["error_ml"] for row in results if row["success"]),default=None))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="episodes"}),flush=True)
    if report["successes"]!=report["total"]:raise SystemExit(1)


if __name__=="__main__":main()
