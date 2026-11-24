from pathlib import Path
import json,sys,hashlib
import mujoco
import numpy as np

root=Path('/data/hlou/pigment-pour-study')
sys.path.insert(0,str(root/'pigment-pour-sim/simulation'))
from inspect_cylinder import CylinderEnv

episode=root/'runs/cylinder-joint-position-bc-v1-step256-single/episode_000'
report=json.loads((episode/'summary.json').read_text())
controller=json.loads((root/'pigment-pour-sim/simulation/configs/cylinder_official_teacher_dev_v1.json').read_text())
case=report['case']
env=CylinderEnv(case['volume'],**controller['environment'],friction_scale=case.get('friction_scale',1.),mass_scale=case.get('mass_scale',1.))
try:
    env.reset_inspection(seed=case['seed'],jitter=case.get('jitter',.004),yaw_jitter=case.get('yaw_jitter',0.))
    assert env.model.opt.timestep==.0005 and env.model.opt.impratio==10
    with np.load(episode/'recovery_physics.npz') as z:
        snapshot=len(z['full_state'])-1
        mujoco.mj_setState(env.model,env.data,z['full_state'][snapshot],int(z['state_spec']))
        env.hold_targets=dict(zip(z['hold_actuator_ids'].astype(int).tolist(),z['hold_targets'][snapshot].tolist()))
        env.applied_action[:]=z['applied_base_action'][snapshot]
    mujoco.mj_forward(env.model,env.data)
    initial=float(env.data.time);maximum=0.;replayed=0
    with np.load(episode/'trajectory.npz') as z:
        start=int(np.argmin(abs(z['timestamp']-initial)))
        assert abs(z['timestamp'][start]-initial)<1e-6
        for frame in range(start,len(z['timestamp'])):
            error=float(abs(env.state_vector()-z['state'][frame]).max())
            maximum=max(maximum,error)
            assert error<1e-6,(frame,error)
            cmd=z['joint_command'][frame]
            env.step_joints(cmd[:6],opening=float(cmd[6]),forward=float(cmd[7]),yaw=float(cmd[8]))
            replayed+=1
    contacts=[]
    for c in env.data.contact[:env.data.ncon]:
        a,b=int(c.geom1),int(c.geom2)
        ba,bb=int(env.model.geom_bodyid[a]),int(env.model.geom_bodyid[b])
        if c.dist<-.001 and ((ba in env.robot_bodies)!=(bb in env.robot_bodies)):
            contacts.append(dict(geom_ids=[a,b],geom_names=[env.model.geom(a).name,env.model.geom(b).name],
                body_names=[env.model.body(ba).name,env.model.body(bb).name],penetration_m=float(-c.dist)))
    metrics=env.metrics()
    assert 'collision:'+str(metrics['forbidden_contacts'])==report['failure_reason']
    result=dict(kind='Recorded bounded-command suffix replay; not another neural task run',
        source_episode=str(episode),restored_snapshot_index=snapshot,source_time_s=initial,
        replayed_frames=replayed,robot_state23_max_error=maximum,final_time_s=float(env.data.time),
        actual_physics_dt=env.model.opt.timestep,actual_impratio=env.model.opt.impratio,
        reproduced_formal_failure_reason=report['failure_reason'],penetrating_contacts=contacts,
        neural_weights_loaded=False,learned_policy_task_success_measured=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (episode.parent/'recorded-collision-replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
finally:env.close()
