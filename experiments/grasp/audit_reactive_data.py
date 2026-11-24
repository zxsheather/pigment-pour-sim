"""Validate corrective supervision and replay physical initial conditions."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL","egl")

import numpy as np

from audit_expert_data import audit_batches
from collect_expert import state_vector
from grasp_walloss import GraspEnvironment
from reactive_expert import apply_setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("experiments/grasp/generated/blue_only/scene_config.json"))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    paths={};positions={};manifests={}
    for split in ["train","validation","holdout"]:
        root=getattr(args,split)
        paths[split],positions[split]=audit_batches([root])
        manifests[split]=json.loads((root/"manifest.json").read_text())
    assert not positions["train"]&positions["validation"]
    assert not positions["train"]&positions["holdout"]
    assert not positions["validation"]&positions["holdout"]
    hashes={m["teacher_source_sha256"] for m in manifests.values()};assert len(hashes)==1
    selected={}
    for entry in manifests["train"]["episodes"]:
        if entry["accepted"]:selected.setdefault(entry["kind"],entry)
    replay=[]
    for kind,entry in selected.items():
        source=args.train/f"episode_{entry['episode']:04d}"
        env=GraspEnvironment(source/"input_scene.xml",args.config,args.output/f"replay_{kind}","blue",tcp="grasp")
        env.step_time=.04
        try:
            apply_setup(env,entry["setup"])
            current=state_vector(env.observation()[0])
            with np.load(source/"trajectory.npz") as raw:expected=raw["state"][0]
            error=float(np.abs(current-expected).max())
            np.testing.assert_allclose(current,expected,atol=1e-5,rtol=0)
            replay.append({"kind":kind,"episode":entry["episode"],"max_initial_state_error":error})
        finally:env.close()
    entries=[e for e in manifests["train"]["episodes"] if e["accepted"]]
    assert len(entries)==32
    assert all(sum(e["kind"]==kind for e in entries)==8 for kind in selected)
    assert all(e["initialization"]["setup_unrecorded"] for e in entries)
    assert all(e["initialization"]["initial_finger_contacts"]<2 for e in entries if e["kind"]=="slip")
    report={"passed":True,"train_attempts":len(manifests["train"]["episodes"]),
            "train_accepted":len(entries),"validation":len(paths["validation"]),"external_holdout":len(paths["holdout"]),
            "frame_counts_by_kind":{kind:[e["frames"] for e in entries if e["kind"]==kind] for kind in selected},
            "initial_state_replay":replay,"setup_actions_in_supervision":False,
            "object_pose_in_features":False,"teacher_source_sha256":next(iter(hashes))}
    (args.output/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
