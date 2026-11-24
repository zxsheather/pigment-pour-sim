"""Bounded controller/solver study with the official geometry kept fixed."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path

from inspect_cylinder import CylinderEnv,run_inspection


def trial(job):
    index,config,root=job
    output=root/f"case_{index:03d}";output.mkdir()
    env=CylinderEnv(25,grasp_x=config["grasp_x"],grasp_height=config["grasp_height"],
                    solver_impratio=config["impratio"],image_size=768)
    try:
        with (output/"run.log").open("w") as log,contextlib.redirect_stdout(log):
            result=run_inspection(env,output,seed=100,jitter=.004,
                grip_opening=config["opening"],read_lateral=config["read_lateral"])
        return dict(case=index,parameters=config,success=result["success"],reason=result["failure_reason"],
                    seconds=result["simulated_seconds"],max_slip_m=result["max_slip_m"],
                    max_tilt_rad=result["max_held_tilt_rad"],error_ml=result.get("absolute_reading_error_ml"))
    finally:env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=8)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    configs=[dict(grasp_x=x,grasp_height=.225,opening=gap,impratio=ratio,read_lateral=.14)
             for x,gap,ratio in itertools.product((.095,.105,.115,.135),(.003,.006,.009),(5.,10.))]
    (args.output/"planned_cases.json").write_text(json.dumps(configs,indent=2)+"\n")
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(trial,[(index,config,args.output) for index,config in enumerate(configs)]):
            results.append(row);print(json.dumps(row),flush=True)
            (args.output/"progress.json").write_text(json.dumps(results,indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps(dict(results=results,
        successes=sum(r["success"] for r in results),total=len(results),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose="Development set only; not independent policy evaluation"),indent=2)+"\n")
