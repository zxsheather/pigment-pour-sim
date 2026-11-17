"""Collect a privileged teacher recovery from an actual policy-visited state.

Robot/object positions, velocities, controls and solver warm starts are restored
from the recorded integration state. Only the episode clock is restarted. This
generates training data; it never counts as a learned-policy task completion.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from inspect_cylinder import CylinderEnv,run_inspection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-episode",type=Path,required=True)
    p.add_argument("--snapshot-index",type=int,required=True)
    p.add_argument("--controller-config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--record",action="store_true")
    p.add_argument("--start-holding",action="store_true",help="Preserve the existing grasp and teach transport/readout only.")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.policy_episode/"summary.json").read_text());case=source["case"]
    controller=json.loads(a.controller_config.read_text())
    aggregate=json.loads((a.policy_episode.parent/"summary.json").read_text())
    assert aggregate["controller"]["environment"]==controller["environment"],"Physics/controller mismatch"
    env=CylinderEnv(case["volume"],image_size=768,**controller["environment"],
                    friction_scale=case.get("friction_scale",1.),mass_scale=case.get("mass_scale",1.))
    try:
        env.reset_inspection(seed=case["seed"],jitter=case.get("jitter",.004),yaw_jitter=case.get("yaw_jitter",0.))
        original_height=float(env.data.xpos[env.cylinder_body,2])
        with np.load(a.policy_episode/"recovery_physics.npz") as saved:
            index=a.snapshot_index;spec=int(saved["state_spec"])
            assert 0<=index<len(saved["full_state"])
            state=saved["full_state"][index]
            assert len(state)==mujoco.mj_stateSize(env.model,spec)
            mujoco.mj_setState(env.model,env.data,state,spec)
            ids=saved["hold_actuator_ids"].astype(int).tolist()
            assert set(ids)==set(env.hold_targets)
            env.hold_targets=dict(zip(ids,saved["hold_targets"][index].tolist()))
            env.applied_action[:]=saved["applied_base_action"][index]
        mujoco.mj_forward(env.model,env.data)
        source_time=float(env.data.time)
        metrics=env.metrics();axis=env.data.xmat[env.cylinder_body].reshape(3,3)[:,2]
        dof=env.model.joint("cylinder_free").dofadr[0]
        recoverable=(metrics["grip_contacts"]<2 and abs(env.data.xpos[env.cylinder_body,2]-original_height)<.015
                     and axis[2]>np.cos(.10) and np.linalg.norm(env.data.qvel[dof:dof+3])<.02)
        if a.start_holding:
            recoverable=(metrics["grip_contacts"]==2 and env.data.xpos[env.cylinder_body,2]-original_height>.02
                         and axis[2]>np.cos(.12))
        provenance=dict(policy_episode=str(a.policy_episode.resolve()),snapshot_index=index,
                        source_simulated_seconds=source_time,kind="expert_recovery_training_only",
                        physical_state="Recorded policy-visited integration state and controller cache; no pose reset",
                        clock="Restarted at zero for the new demonstration sampling timestamps",
                        teacher_inputs="Simulator object pose and robot state; learned policy inputs unchanged")
        provenance["starts_holding"]=a.start_holding
        (a.output/"recovery_source.json").write_text(json.dumps(provenance,indent=2)+"\n")
        if not recoverable:
            (a.output/"source_rejected.json").write_text(json.dumps(dict(reason="initial_state_not_suitable_for_selected_recovery_mode",
                source_time=source_time,contacts=metrics["grip_contacts"]),indent=2)+"\n")
            raise SystemExit(2)
        np.savez_compressed(a.output/"starting_integration_state.npz",state=state,state_spec=spec,
                            hold_actuator_ids=ids,hold_targets=[env.hold_targets[i] for i in ids],
                            applied_base_action=env.applied_action)
        env.data.time=0.;env.grasp_point_reference=None
        env.grasp_reference=env._bottle_relative_position()
        teacher=dict(controller.get("teacher",{}));teacher.update(record_training=True,training_image_size=448,
                    reset_scene=False,initial_state_provenance=provenance,start_holding=a.start_holding)
        result=run_inspection(env,a.output,seed=case["seed"],jitter=case.get("jitter",.004),
                              yaw_jitter=case.get("yaw_jitter",0.),record=a.record,**teacher)
        if not result["success"]:raise SystemExit(1)
    finally:env.close()


if __name__=="__main__":main()
