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
    p.add_argument("--reading-only",action="store_true",help="Held recovery after the robot already retreated: teach presentation/stopping/readout.")
    p.add_argument("--held-initial-lift-m",type=float,default=0.,
                   help="Optional vertical lift before retracting a partially lifted held cylinder; teacher data only.")
    p.add_argument("--lift-off-recovery",action="store_true",
                   help="Teach a vertical lift from a loaded, near-upright grasp at tray height.")
    p.add_argument("--eye-alignment-target-px",type=float,default=2.,
                   help="Teacher centering target before stable reads; acceptance remains at most2 pixels.")
    p.add_argument("--read-lateral-retries", type=float, nargs="+", default=[],
                   help="Alternate camera-relative lateral positions in metres, tried after an RGB reading failure.")
    a=p.parse_args()
    if a.reading_only and not a.start_holding:p.error("--reading-only requires --start-holding")
    if not np.isfinite(a.held_initial_lift_m) or not 0<=a.held_initial_lift_m<=.10:
        p.error("Held recovery lift must be within [0,0.10] m")
    if a.held_initial_lift_m and (not a.start_holding or a.reading_only):
        p.error("Initial lift requires --start-holding without --reading-only")
    if a.lift_off_recovery and (not a.start_holding or a.reading_only or a.held_initial_lift_m < .04):
        p.error("Lift-off recovery requires --start-holding and an initial vertical lift of at least 0.04 m")
    if not np.isfinite(a.eye_alignment_target_px) or not .5<=a.eye_alignment_target_px<=2.:
        p.error("Eye alignment target must be within [0.5,2] pixels")
    if not all(np.isfinite(view) for view in a.read_lateral_retries):
        p.error("Lateral positions must be finite")
    controller = json.loads(a.controller_config.read_text())
    if not a.read_lateral_retries:
        return collect(a, controller, a.output)

    a.output.mkdir(parents=True, exist_ok=False)
    teacher = controller.get("teacher", {})
    views = dict.fromkeys([teacher.get("read_lateral", .14), *a.read_lateral_retries])
    attempts = []
    for index, view in enumerate(views):
        output = a.output / f"view_{index:02d}"
        variant = controller | {"teacher": teacher | {"read_lateral": view}}
        status = collect(a, variant, output)
        evidence = output / ("source_rejected.json" if status == 2 else "summary.json")
        record = json.loads(evidence.read_text())
        failure = record.get("failure_reason", record.get("reason"))
        attempts.append(dict(output=str(output.resolve()), read_lateral_m=view,
                             exit_code=status, failure_reason=failure))
        temporary = a.output / "view-attempts.json.tmp"
        temporary.write_text(json.dumps(dict(attempts=attempts,
            selected_demonstration=str(output.resolve()) if status == 0 else None), indent=2) + "\n")
        temporary.replace(a.output / "view-attempts.json")
        if status != 1 or not reading_failure(failure):
            return status
    return status


def reading_failure(reason):
    return (reason is not None and
            (reason.startswith(("unreadable:", "unstable_reading:")) or reason in {
                "liquid_boundary_occluded", "eye_alignment_timeout", "unstable_boundary_visibility",
                "eye_alignment_drift", "reading_not_stable", "reading_inaccurate"}))


def collect(a, controller, output):
    output.mkdir(parents=True, exist_ok=False)
    source=json.loads((a.policy_episode/"summary.json").read_text());case=source["case"]
    aggregate=json.loads((a.policy_episode.parent/"summary.json").read_text())
    assert aggregate["controller"]["environment"]==controller["environment"],"Physics/controller mismatch"
    env=CylinderEnv(case["volume"],image_size=768,**controller["environment"],
                    friction_scale=case.get("friction_scale",1.),mass_scale=case.get("mass_scale",1.))
    try:
        env.reset_inspection(seed=case["seed"],jitter=case.get("jitter",.004),yaw_jitter=case.get("yaw_jitter",0.))
        initial_base=env.base_pose().copy()
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
        height=float(env.data.xpos[env.cylinder_body,2]-original_height)
        recoverable=(metrics["grip_contacts"]<2 and abs(height)<.015
                     and axis[2]>np.cos(.10) and np.linalg.norm(env.data.qvel[dof:dof+3])<.02)
        if a.start_holding:
            recoverable=metrics["grip_contacts"]==2 and height>.02 and axis[2]>np.cos(.12)
        if a.lift_off_recovery:
            recoverable=(metrics["grip_contacts"]==2 and abs(height)<=.005 and axis[2]>np.cos(.10)
                         and np.linalg.norm(env.data.qvel[dof:dof+3])<.02
                         and not metrics["forbidden_contacts"])
        if a.reading_only:
            # After retreat the held cylinder can be below its original tray
            # height. That is a useful recovery state, not a task success.
            # Require a loaded grasp, clearance and no forbidden contact;
            # retain the earlier stricter lift guard for transport starts.
            recoverable = (metrics["grip_contacts"]==2 and axis[2]>np.cos(.12)
                and initial_base[1]-env.base_pose()[1]>=.20
                and env.data.xpos[env.cylinder_body,2]>.10 and not metrics["forbidden_contacts"])
        provenance=dict(policy_episode=str(a.policy_episode.resolve()),snapshot_index=index,
                        source_simulated_seconds=source_time,kind="expert_recovery_training_only",
                        physical_state="Recorded policy-visited integration state and controller cache; no pose reset",
                        clock="Restarted at zero for the new demonstration sampling timestamps",
                        teacher_inputs="Simulator object pose and robot state; learned policy inputs unchanged")
        provenance["starts_holding"]=a.start_holding
        provenance["initial_vertical_lift_m"]=a.held_initial_lift_m
        provenance["recovery_stage"]="presentation_stopping_readout" if a.reading_only else "transport_or_grasp"
        if a.lift_off_recovery:
            provenance["recovery_stage"]="lift_off_from_loaded_tray_grasp"
            provenance["initial_cylinder_height_relative_to_tray_reset_m"]=height
            provenance["initial_cylinder_angular_speed_rad_s"]=float(np.linalg.norm(env.data.qvel[dof+3:dof+6]))
        if "source_motion" in source:
            assert source["source_motion"]["kind"]=="bounded_joint_command_prefix_replay_training_only"
            provenance["source_motion"]=source["source_motion"]
            provenance["physical_state"]="Recorded replay-visited integration state and controller cache; no pose reset"
        (output/"recovery_source.json").write_text(json.dumps(provenance,indent=2)+"\n")
        if not recoverable:
            (output/"source_rejected.json").write_text(json.dumps(dict(reason="initial_state_not_suitable_for_selected_recovery_mode",
                source_time=source_time,contacts=metrics["grip_contacts"]),indent=2)+"\n")
            return 2
        np.savez_compressed(output/"starting_integration_state.npz",state=state,state_spec=spec,
                            hold_actuator_ids=ids,hold_targets=[env.hold_targets[i] for i in ids],
                            applied_base_action=env.applied_action)
        env.data.time=0.;env.grasp_point_reference=None
        env.grasp_reference=env._bottle_relative_position()
        teacher=dict(controller.get("teacher",{}));teacher.update(record_training=True,training_image_size=448,
                    reset_scene=False,initial_state_provenance=provenance,start_holding=a.start_holding,
                    skip_retreat=a.reading_only,held_initial_lift_m=a.held_initial_lift_m,
                    eye_alignment_target_px=a.eye_alignment_target_px)
        result=run_inspection(env,output,seed=case["seed"],jitter=case.get("jitter",.004),
                              yaw_jitter=case.get("yaw_jitter",0.),record=a.record,**teacher)
        return 0 if result["success"] else 1
    finally:env.close()


if __name__=="__main__":raise SystemExit(main())
