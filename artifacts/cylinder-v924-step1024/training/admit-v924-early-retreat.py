"""Append certified V718 first-contact withdrawals with explicit sampling of retreat onset and motion."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

stage=Path(__file__).resolve().parent
runs=stage.parent
baseline_dir,=runs.glob('*-v922')
baseline=baseline_dir/'v922-verified-contact-height-inputs.json'
retreat,=runs.glob('*-v917/v917-v718-contact-cameras.json')
previous,=runs.glob('*-v834/admit-v834-direct-advance.py')
helper=runpy.run_path(str(previous))
read,ref,verified,write,append=(helper[key] for key in ('read','ref','verified','write','append'))
output=runs.parent/'datasets/cylinder-joint-v907-plus-early-v718-retreat-v924'
target=runs/'cylinder-joint-early-v718-retreat-bc-v924-a6000'


def approve(manifest, norm, cameras):
    config=Path(sys.modules[append.__module__].__file__).parent/'configs'
    protected={r['seed'] for suite in ('original','dev') for r in read(config/f'cylinder_policy_{suite}_cases.json')}
    seen={row['metadata']['result']['seed'] for row in read(manifest)}
    rows=[]
    for entry in cameras['results']:
        if not entry['qualified']:
            continue
        path=verified(entry['certificate'])
        proof=read(path)
        assert proof['qualified'] and proof['case']==entry['case'] and proof['frames']==entry['frames']
        assert proof['geometry_audit_passed'] and proof['teacher_full_task_succeeded']
        assert not proof['model_task_success_measured'] and not proof['independent_cases_read']
        assert proof['original_development_layout_excluded'] and ref(verified(proof['normalizer']))['sha256']==ref(norm)['sha256']
        assert proof['physically_replayed_frames']==proof['frames'] and not any(proof['normalization_violations'].values())
        assert proof['validated_label_physics']['limit_projected_frames']==0
        arrays=verified(proof['training_arrays'])
        with np.load(Path(proof['source'])/'trajectory.npz',allow_pickle=False) as data:
            assert (data['phase'][:24]=='withdraw_from_contact').all()
            assert (data['task_index'][:24]==0).all()
        seed=proof['case']['seed']
        assert seed not in seen|protected
        seen.add(seed)
        rows.append(dict(seed=seed,proof=str(path),qualified=True,training_arrays=str(arrays),certification=proof))
    assert len(rows)==cameras['qualified_count']
    return rows


def model_weights(previous_weights, rows, config, frames):
    previous=np.load(previous_weights,allow_pickle=False)
    groups=[]
    for row in rows:
        with np.load(row['training_arrays'],allow_pickle=False) as f:
            task=f['task_index'].copy()
        tags=task+2
        tags[0]=0
        tags[[8,16]]=1
        tags[-7:]=-1
        groups.append(tags)
    tags=np.concatenate(groups)
    assert len(previous)+len(tags)==frames
    training,_=torch.utils.data.random_split(range(frames),[.95,.05],
        generator=torch.Generator().manual_seed(config['hyperparams']['seed']))
    indices=np.asarray(training.indices)
    added=indices[indices>=len(previous)]-len(previous)
    mass=.025*previous.sum()/(1.-.025)
    new=np.zeros(len(tags),dtype=np.float64)
    fractions=(.12,.24,.28,.20,.16)
    for task,fraction in enumerate(fractions):
        mask=tags==task
        assert mask[added].any()
        new[mask]=mass*fraction/mask.sum()
    weights=np.r_[previous,new]
    np.testing.assert_array_equal(weights[:len(previous)],previous)
    sampler=stage/'v924-frame-sampling-weights.npy'
    np.save(sampler,weights)
    total=weights[indices].sum()
    names=('first_contact_window','demonstrated_retreat_windows','approach_and_grasp','lift_and_retreat','present_and_read')
    result={name:float(new[added[tags[added]==task]].sum()/total)
            for task,name in enumerate(names)}
    return sampler,sum(result.values()),result


def ready():
    while not baseline.is_file():
        if next(runs.glob('*/candidate-ready.json'),None):
            return False
        pid=(baseline_dir/'admission.pid').read_text().strip()
        status=subprocess.run(['ps','-p',pid,'-o','stat=,args='],capture_output=True,text=True).stdout.strip()
        if not baseline.is_file() and (not status or status.startswith('Z') or 'admit-v922-contact-height.py' not in status):
            raise RuntimeError('The matched retreat-height configuration ended before official validation')
        time.sleep(8)
    return not next(runs.glob('*/candidate-ready.json'),None)


def main():
    if not ready():
        return
    prior=read(baseline)
    assert prior['training_admission_performed'] and prior['data_unchanged']
    assert not prior['optimizer_started'] and not prior['independent_cases_read'] and prior['execution_prefix']==8
    import wall_x.data.backends.lerobot.recovery_sampler as sampler_module
    runtime,=runs.glob('*-v896/fixed-wall-x')
    assert Path(sampler_module.__file__).is_relative_to(runtime)
    import wall_x.model.core.action.cylinder_wrist_pose_loss as wrist
    assert ref(Path(wrist.__file__))==prior['active_loss']
    cfg=yaml.safe_load(verified(prior['config']).read_text())
    data=cfg['data']
    root=Path(data['lerobot_config']['root'])
    norm=Path(data['norm_stats_path'])
    old_weights=Path(data['lerobot_config']['frame_sampling_weights_path'])
    assert ref(old_weights)==prior['sampler']
    cameras=read(retreat)
    assert cameras['teacher_only'] and not cameras['training_admission_performed'] and not cameras['independent_cases_read']
    rows=approve(root/'demonstration_provenance.json',norm,cameras)
    assert len(rows)==6 and len({row['seed'] for row in rows})==6
    bundle=stage/'qualifications.json'
    write(bundle,dict(base_dataset=str(root),results=rows,qualified_count=len(rows),independent_cases_read=False))
    built=append(root,bundle,old_weights,output)
    assert built['base_frames_exact'] and built['normalization_exact'] and built['reader_reopen_passed']
    assert built['new_frames']==sum(r['certification']['frames'] for r in rows) and built['new_episodes']==len(rows)
    sampler,mass,phases=model_weights(old_weights,rows,cfg,built['frames'])
    data['lerobot_config'].update(root=str(output),repo_id=str(output),frame_sampling_weights_path=str(sampler))
    data['norm_stats_path']=str(output/'joint_norm_stats.json')
    cfg['logging']['log_name']=target.name
    cfg['checkpoint']['save_path']=str(target/'checkpoints')
    config=stage/'train-v924.yml'
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    serving=stage/'serving-normalizer-view'
    serving.mkdir()
    shutil.copyfile(norm,serving/'norm_stats.json')
    validator,=runs.glob('*-v779/validate_wall_cylinder_joints.py')
    validation=stage/'official-v924-data-validation.json'
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port=30851',str(validator),
         '--config',str(config),'--output',str(validation)]
    for episode in range(built['episodes']-len(rows),built['episodes']):
        cmd.extend(('--check-episode',str(episode)))
    subprocess.run(cmd,check=True)
    proof=read(validation)
    assert proof['passed'] and proof['wrist_loss_tracking_affine_matches_normalizer']
    assert proof['sampling_weights_match_source_indices'] and proof['checkpoint_serving_normalizer_factory_matches_training']
    write(stage/'v924-verified-early-v718-retreat-inputs.json',dict(base=ref(baseline),camera_index=ref(retreat),
        active_loss=prior['active_loss'],
        wall_sampler_module=ref(Path(sampler_module.__file__)),dataset=built,official_loader=ref(validation),config=ref(config),qualifications=ref(bundle),
        sampler=ref(sampler),architecture=prior['architecture'],serving_normalizer=ref(serving/'norm_stats.json'),
        sampling_by_phase=phases,added_training_fraction=mass,old_sampling_weights_unchanged=True,optimizer_started=False,
        training_admission_performed=True,execution_prefix=8,model_success_measured=False,independent_cases_read=False))
    print(built['episodes'],built['frames'],mass,flush=True)


if __name__=='__main__':
    main()
