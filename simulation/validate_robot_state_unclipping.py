"""Compare the official state normalizer with the opt-in affine-only version.

Reads actual training states and recorded robot states. No model parameters,
training files or physical acceptance conditions are modified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow
import torch

from wall_x.config.loader import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from robot_state_normalization import enable_unclipped_robot_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    torch.set_num_threads(1)
    cfg=load_config(str(a.config));action,original,*_=load_lerobot_normalizers(cfg)
    affine=copy.deepcopy(original);enable_unclipped_robot_state(affine)
    root=Path(cfg.data.lerobot_config['root'])
    table=arrow.dataset(sorted((root/'data').rglob('*.parquet')),format='parquet').to_table(columns=['observation.state'])
    training=torch.tensor(np.asarray(table['observation.state'].to_pylist()),dtype=torch.float32)[:,None,:]
    key=next(iter(original.min));names=[key]*len(training)
    before=original.normalize_data(training,names);after=affine.normalize_data(training,names)
    if not torch.equal(before,after):raise ValueError('Existing training-state normalization changed')
    metadata=json.loads((a.recovery/'metadata.json').read_text())
    source=Path(metadata['result']['initial_state_provenance']['policy_episode'])
    with np.load(a.recovery/'trajectory.npz') as d:
        recovery=np.c_[d['state'],d['base_pose']].astype(np.float32)
    with np.load(source/'model_observations.npz') as d:
        policy=d['state'].copy();times=d['timestamp'].copy()
    rows=[]
    for label,states in [('successful_teacher_recovery',recovery),('actual_policy_inputs',policy)]:
        x=torch.from_numpy(states)[:,None,:];names=[key]*len(x)
        clipped=original.normalize_data(x,names);unclipped=affine.normalize_data(x,names)
        reconstructed=original.unnormalize_data(unclipped,names)
        error=float((x-reconstructed).abs().max())
        if error>1e-6:raise ValueError('Affine normalization did not retain raw robot state')
        changed=(clipped!=unclipped).squeeze(1)
        indices=torch.nonzero(changed.any(-1)).flatten().tolist()
        rows.append(dict(source=label,frames=len(states),changed_frame_count=len(indices),
            changed_fraction_per_dimension=changed.float().mean(0).tolist(),
            clipped_frame_indices=indices,
            clipped_policy_seconds=times[indices].tolist() if label=='actual_policy_inputs' else None,
            unclipped_min=unclipped.min(0).values.squeeze(0).tolist(),
            unclipped_max=unclipped.max(0).values.squeeze(0).tolist(),
            max_raw_state_reconstruction_error=error))
    report=dict(kind='real_robot_state_normalization_ablation_contract',
        training_dataset=str(root),training_frames=len(training),
        original_training_normalized_states_bit_exact=True,
        new_robot_state_information_retained=True,action_normalizer_not_modified=True,
        observations=rows,model_forward=False,physics_advanced=False,training_modified=False,
        task_success_measured=False,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),Path(__file__).with_name('robot_state_normalization.py')]})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(training_frames=len(training),original_states_bit_exact=True,
        changed_frames=[(r['source'],r['changed_frame_count']) for r in rows])))


if __name__=='__main__':main()
