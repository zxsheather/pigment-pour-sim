"""Run the corrective-data comparison, selecting only on validation episodes.

Reproduce scripted initial conditions before policy control. Tabletop pickup
and slip recovery remain separate tasks in every result table.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("experiments/grasp/runs"))
    p.add_argument("--experiment",type=Path,default=Path("experiments/grasp/runs/train_reactive_3000_v4"))
    p.add_argument("--reference",type=Path,default=Path("experiments/grasp/runs/train_action_fp32_1000_v3/evaluation/adapted"))
    p.add_argument("--output",type=Path,default=Path("experiments/grasp/runs/reactive_eval_v4"))
    p.add_argument("--reference-final-device",default="1")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    roots={name:args.runs/f"reactive_{name}_v4" for name in ["train","validation","holdout","recovery_holdout"]}
    accepted={name:[e for e in read(root/"manifest.json")["episodes"] if e["accepted"]] for name,root in roots.items()}
    assert [len(accepted[k]) for k in roots]==[32,8,8,6]
    position_sets=[{tuple(e["offset_xy_m"]) for e in accepted[k]} for k in roots]
    assert all(not position_sets[i]&position_sets[j] for i in range(4) for j in range(i+1,4))

    def case(split,group,entry):
        return {"split":split,"batch":roots[group].name,"episode":entry["episode"],"kind":entry["kind"]}

    train_probes=[next(e for e in accepted["train"] if e["kind"]==kind) for kind in ["clean","hand_offset"]]
    internal=[case("train_episode","train",e) for e in train_probes]+[case("unused_internal_episode","validation",e) for e in accepted["validation"]]
    external=[case("external_holdout","holdout",e) for e in accepted["holdout"]]
    external += [case("external_recovery_"+e["kind"],"recovery_holdout",e) for e in accepted["recovery_holdout"]]
    internal_file=args.output/"internal_cases.json";internal_file.write_text(json.dumps(internal,indent=2))
    external_file=args.output/"external_cases.json";external_file.write_text(json.dumps(external,indent=2))
    protocol={"train_demonstrations":32,"validation_pickups":8,"external_pickups":8,
              "external_recovery_by_kind":{"hand_offset":2,"grasp_miss":2,"slip":2},
              "select_between_new_steps":[1000,3000],"selection_rule":"Most validation stable pickups; tie: lower validation execute8 XYZ error; tie: earlier updates",
              "external_used_for_selection":False,"model_control_after_scripted_initialization_only":True,
              "slip_recovery_is_not_counted_as_tabletop_pickup":True,"seed":0,
              "control_period_s":.04,"execute_steps":8,"max_chunks":48}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    candidates=[("reference_v3",args.reference,"local/pigment_grasp_train",1000),
                ("reactive_001000",args.experiment/"step_001000/evaluation/adapted","local/pigment_grasp_reactive_v4",1000),
                ("reactive_003000",args.experiment/"evaluation/adapted","local/pigment_grasp_reactive_v4",3000)]

    def command(script,checkpoint,norm,output,cases,extra=()):
        return [sys.executable,str(scripts/script),"--source",str(args.source),"--checkpoint",str(checkpoint),
                "--norm-key",norm,"--output",str(output),"--cases",str(cases),*extra]

    def execute(script,checkpoint,norm,output,cases,extra=()):
        result=output/"report.json"
        if result.exists():
            data=read(result)
            if data.get("completed",bool(data.get("records"))):return data
            raise RuntimeError(f"Incomplete existing evaluation: {output}")
        if output.exists():raise RuntimeError(f"Output already exists: {output}")
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command(script,checkpoint,norm,output,cases,extra),check=True,stdout=log,stderr=subprocess.STDOUT)
        return read(result)

    summaries=[]
    for label,checkpoint,norm,steps in candidates:
        deadline=time.monotonic()+7200
        while not (checkpoint/"adaptation.json").exists():
            error_file=args.experiment/"report.json"
            if error_file.exists() and read(error_file).get("error"):raise RuntimeError("Training failed; inspect its report")
            if time.monotonic()>deadline:raise TimeoutError(label)
            time.sleep(10)
        if label=="reactive_001000":
            audit_file=args.output/"input_contract.json"
            if not audit_file.exists():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
                with (args.output/"input_contract.log").open("w") as log:
                    subprocess.run([sys.executable,str(scripts/"audit_observation_contract.py"),
                        "--source",str(args.source),"--checkpoint",str(checkpoint.parent/"baseline"),
                        "--config",str(args.experiment/"train.yml"),"--norm-key",norm,"--phase-frames",
                        "--episode-folder",str(roots["train"]/f"episode_{accepted['train'][0]['episode']:04d}"),
                        "--output",str(audit_file)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            audit=read(audit_file);assert audit["proprioception_matches"] and audit["prefix_tokens_match"]
        opened=execute("evaluate_expert_openloop.py",checkpoint,norm,args.output/("openloop_"+label),internal_file,("--phase-anchors",))
        closed=execute("evaluate_expert_closedloop.py",checkpoint,norm,args.output/("closedloop_"+label),internal_file)
        assert all(c["trial_valid"] for c in closed["cases"])
        row={"label":label,"steps":steps,"openloop":opened["aggregates"],"closedloop":closed["aggregates"]}
        summaries.append(row);(args.output/"comparison.json").write_text(json.dumps(summaries,indent=2))
        print("VALIDATED "+json.dumps(row),flush=True)
    eligible=[s for s in summaries if s["label"]!="reference_v3"]
    selected=max(eligible,key=lambda s:(s["closedloop"]["unused_internal_episode"]["successes"],
                                      -s["openloop"]["unused_internal_episode"]["execute8_xyz_cm"],-s["steps"]))
    choice={"candidate":selected,"rule":protocol["selection_rule"],"external_used_for_selection":False}
    (args.output/"selection.json").write_text(json.dumps(choice,indent=2))
    by_label={label:(checkpoint,norm) for label,checkpoint,norm,_ in candidates}
    jobs=[]
    for label in ["reference_v3",selected["label"]]:
        checkpoint,norm=by_label[label];output=args.output/("final_"+label)
        if (output/"report.json").exists():
            assert read(output/"report.json").get("completed");continue
        env=dict(os.environ)
        if label=="reference_v3":
            env.update(CUDA_VISIBLE_DEVICES=args.reference_final_device,MUJOCO_EGL_DEVICE_ID=args.reference_final_device)
        log=output.with_suffix(".log").open("w")
        proc=subprocess.Popen(command("evaluate_expert_closedloop.py",checkpoint,norm,output,external_file,("--external-holdout",)),
                              env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log,label))
    for proc,log,label in jobs:
        code=proc.wait();log.close()
        if code:raise RuntimeError(f"Final comparison failed: {label}")
    final={label:read(args.output/("final_"+label)/"report.json") for label in ["reference_v3",selected["label"]]}
    assert all(r["completed"] and all(c["trial_valid"] for c in r["cases"]) for r in final.values())
    summary={"completed":True,"selection":choice,"all_trials_valid":True,
             "paired_external":{label:r["aggregates"] for label,r in final.items()}}
    (args.output/"final_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
