import json
from pathlib import Path
import numpy as np

root=Path('/data/hlou/pigment-pour-study')
episode=root/'runs/cylinder-joint-position-bc-v1-step256-single/episode_000'
stats=json.loads((root/'datasets/cylinder-joint-recovery-v1/joint_norm_stats.json').read_text())['norm_stats']
with np.load(episode/'trajectory.npz') as z:
    trajectory={k:z[k] for k in z.files}
with np.load(episode/'model_observations.npz') as z:
    observations={k:z[k] for k in z.files}
chunks=np.load(episode/'predicted_chunks.npy')
state=observations['state']
lo=np.asarray(stats['observation.state']['q01'])
hi=np.asarray(stats['observation.state']['q99'])
outside=(state<lo-1e-7)|(state>hi+1e-7)
action_low=np.asarray(stats['action']['q01'])
action_high=np.asarray(stats['action']['q99'])
prefix=chunks[:,:8,:9]
contacts=trajectory['contacts']
timestamps=trajectory['timestamp']
actions=trajectory['unfiltered_model_action']
projection=trajectory['joint_limit_projection_rad']
end_effector=trajectory['state'][:,10:13]
per_state=[]
for i in range(18):
    bad=np.flatnonzero(outside[:,i])
    per_state.append(dict(index=i,observed_min=float(state[:,i].min()),
        observed_max=float(state[:,i].max()),lower=float(lo[i]),upper=float(hi[i]),
        outside_count=len(bad),first_outside_seconds=(float(observations['timestamp'][bad[0]]) if len(bad) else None)))
per_action=[]
for i in range(9):
    values=prefix[...,i]
    per_action.append(dict(index=i,executed_prefix_prediction_min=float(values.min()),
        executed_prefix_prediction_max=float(values.max()),lower=float(action_low[i]),
        upper=float(action_high[i]),outside_fraction=float(((values<action_low[i]-1e-7)|(values>action_high[i]+1e-7)).mean())))
first_projection=np.flatnonzero(projection>0)
contact_counts={str(int(k)):int(v) for k,v in zip(*np.unique(contacts,return_counts=True))}
windows=[]
for a,b in [(0,5),(5,10),(10,20),(20,40),(40,90)]:
    mask=(timestamps>=a)&(timestamps<b)
    windows.append(dict(seconds=[a,b],frames=int(mask.sum()),
        grip_opening_fraction_quantiles=np.quantile(actions[mask,6],[0,.25,.5,.75,1]).tolist(),
        forward_command_quantiles=np.quantile(actions[mask,7],[0,.25,.5,.75,1]).tolist(),
        contact_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(contacts[mask],return_counts=True))}))
report=dict(scope='CPU analysis of already recorded genuine joint256 model outputs; no extra model run or policy correction',
    model_success=False,model='runs/cylinder-joint-position-bc-v1/checkpoints/0_256',
    observations=len(state),executed_frames=len(timestamps),
    input_state_bounds={'observations_outside':int(outside.any(1).sum()),'fraction':float(outside.any(1).mean()),'per_channel':per_state},
    relative_model_prefix_bounds=per_action,
    contacts=contact_counts,
    joint_limit_projection=dict(frames=int((projection>0).sum()),maximum_rad=float(projection.max()),
        first_seconds=(float(timestamps[first_projection[0]]) if len(first_projection) else None)),
    end_effector_net_displacement_m=(end_effector[-1]-end_effector[0]).tolist(),
    windows=windows,
    interpretation='The recorded run never established a loaded grasp. Base-only retreat is not task success. Input clipping and legal-joint projection are measured symptoms; this observational analysis does not establish their causal role. Keep the current training settings pending later checkpoint evidence.')
target=episode.parent/'recorded-failure-analysis.json'
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['observations','executed_frames','contacts','joint_limit_projection','end_effector_net_displacement_m']},indent=2))
print(json.dumps({'state_outside_channels':[(p['index'],p['outside_count'],p['first_outside_seconds']) for p in per_state if p['outside_count']],
    'relative_prefix_outside_channels':[(p['index'],p['outside_fraction']) for p in per_action if p['outside_fraction']]},indent=2))
