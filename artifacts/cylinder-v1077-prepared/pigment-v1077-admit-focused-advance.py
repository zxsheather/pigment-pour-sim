"""Admit certified open-approach demonstrations with audited window sampling."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml


stage=Path(__file__).resolve().parent
runs=stage.parent
previous,=runs.glob('*-v834/admit-v834-direct-advance.py')
helpers=runpy.run_path(str(previous))
read,ref,verified,write,append=(helpers[k] for k in ('read','ref','verified','write','append'))
baseline,=runs.glob('*-v1062/v1062-verified-49ml-readout-inputs.json')
loss_check,=runs.glob('*-v1076/v1076-certified-advance-window-audit.json')
paired,=runs.glob('*-v1064/v1064-verified-open-advance-pair.json')
output=runs.parent/'datasets/cylinder-joint-v1062-plus-matched-advance-v1077'
target=runs/'cylinder-joint-focused-open-advance-bc-v1077-a6000'


def approve(source, audit, root, norm_path):
    config=Path(sys.modules[append.__module__].__file__).parent/'configs'
    protected={row['seed'] for suite in ('original','dev') for row in read(config/f'cylinder_policy_{suite}_cases.json')}
    seen={row['metadata']['result']['seed'] for row in read(root/'demonstration_provenance.json')}
    rows=[]
    for entry in source['results']:
        p=verified(entry['certificate'])
        certificate=read(p)
        trace=read(verified(entry['source']))
        assert entry['qualified'] and certificate['qualified']
        assert trace['native_teacher_passed'] and trace['strict']['success']
        assert trace['strict']['geometry']['passed'] and not trace['strict']['geometry']['custom_pads']
        assert certificate['geometry_audit_passed'] and certificate['teacher_full_task_succeeded']
        assert certificate['original_development_layout_excluded'] and not certificate['independent_cases_read']
        assert certificate['physically_replayed_frames']==certificate['frames']==entry['frames']
        assert ref(verified(certificate['normalizer']))['sha256']==ref(norm_path)['sha256']
        assert not any(certificate['normalization_violations'].values())
        assert certificate['validated_label_physics']['limit_projected_frames']==0
        seed=entry['case']['seed']
        assert seed not in seen|protected
        seen.add(seed)
        labels,=[d for d in audit['results'] if d['case']['seed']==seed]
        assert labels['certificate']==entry['certificate']
        matched,=[q for q in labels['against_archived_policy'] if q['label']=='v950-256' and q['control_tick']==328]
        target_frames=matched['nearby_added_objective_frames']
        extra=labels['additional_effective_frames']
        assert target_frames and set(target_frames)<=set(extra)
        rows.append(dict(seed=seed,proof=str(p),qualified=True,certification=certificate,
                         training_arrays=str(verified(certificate['training_arrays'])),
                         targeted_start_frames=target_frames,newly_supervised_start_frames=extra))
    assert len(rows)==source['qualified_count']==3
    return rows


def allocate(cfg, before_sampler, rows, frames):
    previous=np.load(before_sampler,allow_pickle=False)
    tags=[]
    for row in rows:
        with np.load(row['training_arrays'],allow_pickle=False) as f:
            task=f['task_index'].copy()
        flags=task.astype(np.int64)+2
        assert np.isin(flags,(2,3,4)).all()
        flags[row['newly_supervised_start_frames']]=1
        flags[row['targeted_start_frames']]=0
        assert (task[row['targeted_start_frames']]==0).all()
        flags[-7:]=-1
        tags.append(flags)
    codes=np.concatenate(tags)
    assert len(codes)+len(previous)==frames
    fit,_=torch.utils.data.random_split(range(frames),[.95,.05],
        generator=torch.Generator().manual_seed(cfg['hyperparams']['seed']))
    train=np.asarray(fit.indices)
    included=train[train>=len(previous)]-len(previous)
    mass=.025*previous.sum()/(1.-.025)
    chosen=np.array([.50,.10,.16,.12,.12])
    used=np.asarray([np.any(codes[included]==i) for i in range(len(chosen))])
    assert used[0]
    chosen*=used
    chosen/=chosen.sum()
    extras=np.zeros(len(codes),dtype=np.float64)
    for i,share in enumerate(chosen):
        selection=codes==i
        if share:
            extras[selection]=mass*share/selection.sum()
    weights=np.r_[previous,extras]
    np.testing.assert_array_equal(weights[:len(previous)],previous)
    path=stage/'v1077-weights.npy'
    np.save(path,weights)
    names=('matched_extended_forward','other_extended_forward','remaining_approach','lift_and_retreat','present_and_read')
    total=weights[train].sum()
    fractions={name:float(extras[included[codes[included]==i]].sum()/total) for i,name in enumerate(names)}
    cases=[]
    offset=0
    for row,flags in zip(rows,tags,strict=True):
        active=included[(included>=offset)&(included<offset+len(flags))]
        active=active[codes[active]==0]
        cases.append(dict(seed=row['seed'],nearby_in_training=len(active),nearby_total=sum(flags==0).item(),
                          sample_probability=float(extras[active].sum()/total)))
        offset+=len(flags)
    assert all(a['nearby_in_training'] for a in cases)
    return path,fractions,cases


def main():
    prior,audit,duo=map(read,(baseline,loss_check,paired))
    assert not any(d['independent_cases_read'] for d in (prior,audit,duo))
    assert prior['training_admission_performed'] and not prior['optimizer_started']
    assert audit['prior_training']==ref(baseline) and audit['matching_label_loss_zero']
    assert duo['same_parent_optimizer_sampler_and_data'] and read(verified(duo['official_loader']))['passed']
    cfg=yaml.safe_load(verified(prior['config']).read_text())
    normal=Path(cfg['data']['norm_stats_path'])
    previous_sampler=verified(prior['sampler'])
    root=Path(cfg['data']['lerobot_config']['root'])
    camera=read(verified(audit['source']))
    assert camera['teacher_only'] and not camera['training_admission_performed']
    assert read(verified(prior['official_loader']))['passed']
    rows=approve(camera,audit,root,normal)
    bundle=stage/'qualifications.json'
    write(bundle,dict(base_dataset=str(root),results=rows,qualified_count=len(rows),independent_cases_read=False))
    built=append(root,bundle,previous_sampler,output)
    assert built['base_frames_exact'] and built['normalization_exact'] and built['reader_reopen_passed']
    assert built['new_frames']==1712 and built['new_episodes']==3
    sampler,share,cases=allocate(cfg,previous_sampler,rows,built['frames'])
    cfg['data']['lerobot_config'].update(root=str(output),repo_id=str(output),frame_sampling_weights_path=str(sampler))
    cfg['data']['norm_stats_path']=str(output/'joint_norm_stats.json')
    config=verified(duo['trials']['extended']['architecture'])
    cfg['model']['config_path']=str(config)
    parent=Path(duo['parent'])
    cfg['checkpoint']['resume_from']=str(parent/'model.safetensors')
    cfg['checkpoint']['save_path']=str(target/'checkpoints')
    cfg['logging']['log_name']=target.name
    parameters=stage/'train-v1077.yml'
    parameters.write_text(yaml.safe_dump(cfg,sort_keys=False))
    serving=stage/'serving-normalizer-view'
    serving.mkdir()
    shutil.copyfile(normal,serving/'norm_stats.json')
    validator,=runs.glob('*-v779/validate_wall_cylinder_joints.py')
    proof_file=stage/'official-v1077-data-validation.json'
    argv=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port=31077',str(validator),
          '--config',str(parameters),'--output',str(proof_file)]
    for i in range(built['episodes']-built['new_episodes'],built['episodes']):
        argv.extend(('--check-episode',str(i)))
    subprocess.run(argv,check=True)
    proof=read(proof_file)
    assert proof['passed'] and proof['wrist_loss_tracking_affine_matches_normalizer']
    assert proof['sampling_weights_match_source_indices'] and proof['checkpoint_serving_normalizer_factory_matches_training']
    assert proof['frames']==built['frames'] and proof['episodes']==built['episodes']
    write(stage/'v1077-verified-focused-open-advance.json',dict(
        base=ref(baseline),audit=ref(loss_check),paired_reference=ref(paired),official_loader=ref(proof_file),
        qualified_rows=ref(bundle),config=ref(parameters),architecture=ref(config),sampler=ref(sampler),
        serving_normalizer=ref(serving/'norm_stats.json'),active_loss=prior['active_loss'],dataset=built,
        sampling_fraction_by_new_group=share,nearby_training_windows=cases,old_sampling_weights_unchanged=True,
        training_admission_performed=True,optimizer_started=False,execution_prefix=8,independent_cases_read=False))
    print(built['episodes'],built['frames'],share,flush=True)


if __name__=='__main__' and not next(runs.glob('*/candidate-ready.json'),None):
    main()
