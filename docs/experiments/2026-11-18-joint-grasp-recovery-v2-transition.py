"""Continue from joint768 with the revised data if the first six cases never grasp."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml

root = Path('/data/hlou/pigment-pour-study')
old = root / 'runs/cylinder-joint-position-bc-v1'
new = root / 'runs/cylinder-joint-grasp-recovery-bc-v2'
parent = old / 'checkpoints/0_768'
old_pid = 699975
trainer_pid = 704492
torchrun_pid = 704458
status_file = new / 'transition-status.json'
status = dict(stage='waiting_for_complete_joint768', pid=os.getpid(), old_driver=old_pid,
              old_trainer=trainer_pid, parent=str(parent), new_run=str(new),
              checkpoint_768_required=True, started=time.time())


def save(stage, **fields):
    status.update(stage=stage, updated=time.time(), **fields)
    temp = status_file.with_suffix('.tmp')
    temp.write_text(json.dumps(status, indent=2) + '\n')
    temp.replace(status_file)


def command(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'state=,args='],
                            capture_output=True, text=True).stdout.strip()
    return result if result and not result.startswith('Z') else ''


def run():
    auto = root / 'pigment-pour-sim/simulation/train_and_evaluate_cylinder_joints.py'
    config = new / 'train.yml'
    cfg = yaml.safe_load(config.read_text())
    assert Path(cfg['checkpoint']['resume_from']) == parent / 'model.safetensors'
    assert not (new / 'execution-status.json').exists()
    check = json.loads((new / 'prepared-data-validation.json').read_text())
    assert check['passed'] and check['episodes'] == 53 and check['frames'] == 32362
    assert check['dataset'] == cfg['data']['lerobot_config']['root']
    inputs = json.loads((root / 'results/cylinder-joint-grasp-recovery-v2-data-audit.json').read_text())
    assert inputs['teacher_replay_passed'] and inputs['old_rows_byte_values_equal'] == 31797
    for _ in range(360):
        s = json.loads((old / 'execution-status.json').read_text())
        if (old / 'candidate-ready.json').exists() or s['stage'] == 'candidate_requires_full_suites':
            save('kept_v1_complete_single_candidate'); return
        if s.get('latest_evaluated_step', 0) >= 768:
            break
        assert 'train_and_evaluate_cylinder_joints.py' in command(old_pid)
        time.sleep(5)
    else:
        save('wait_window_expired_v1_left_running'); return
    if s['stage'] != 'training' or s['latest_evaluated_step'] != 768:
        save('kept_v1_eval_stage_advanced', observed_v1_stage=s['stage']); return
    results = []
    for step in range(128, 769, 128):
        output = root / f'runs/cylinder-joint-position-bc-v1-step{step}-single'
        summary = json.loads((output / 'summary.json').read_text())
        case = summary['episodes'][0]
        assert summary['evaluation_complete'] and summary['total'] == 1 and summary['action_space'] == 'joint'
        assert case['geometry_audit']['passed'] and case['geometry_audit']['custom_pads'] == 0
        with np.load(output / 'episode_000/trajectory.npz') as rollout:
            contacts = int(rollout['contacts'].max(initial=0))
        row = dict(step=step, success=case['success'], failure_reason=case['failure_reason'],
                   simulated_seconds=case['simulated_seconds'], held=case['final_task_gates'].get('held', False),
                   max_lift_m=case['max_lifted_m'], max_loaded_finger_contacts=contacts)
        results.append(row)
    if any(r['success'] or r['held'] or r['max_lift_m'] >= .04 or r['max_loaded_finger_contacts'] for r in results):
        save('kept_v1_grasp_progress_observed', results=results); return
    audit = json.loads((old / 'checkpoint-768-weight-audit.json').read_text())
    assert audit['passed'] and Path(audit['checkpoint']) == parent
    assert (parent / 'model.safetensors').stat().st_size > 7 * 1024**3
    assert (parent / 'optimizer.pt').stat().st_size > 1024**3
    assert 'train_and_evaluate_cylinder_joints.py' in command(old_pid)
    assert 'train_fsdp.py' in command(trainer_pid) and 'torch.distributed.run' in command(torchrun_pid)
    s = json.loads((old / 'execution-status.json').read_text())
    if s['stage'] != 'training' or s['latest_evaluated_step'] != 768 or (old / 'candidate-ready.json').exists():
        save('kept_v1_state_changed', results=results, observed_v1_stage=s['stage']); return
    save('stopping_v1_after_complete_checkpoint', results=results,
         rationale='Six full evaluations failed before any loaded contact. Continue the same action schema with the 53-demo data and emphasis on grasp approach.',
         old_checkpoint_retained=str(parent), old_checkpoint_sha256=audit['checkpoint_sha256'])
    os.kill(old_pid, signal.SIGTERM)
    for _ in range(45):
        if not any(command(pid) for pid in (old_pid, trainer_pid, torchrun_pid)):
            break
        time.sleep(2)
    else:
        save('not_launched_v1_process_remains'); return
    log_path = root / 'results/cylinder-joint-grasp-recovery-bc-v2.log'
    argv = [str(root / '.venv-wall/bin/python'), str(auto), '--run', str(new),
            '--parent', str(parent), '--log', str(log_path)]
    with log_path.open('x') as log:
        process = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    save('v2_driver_launched', new_driver_pid=process.pid, argv=argv, log=str(log_path))


try:
    assert not status_file.exists()
    save('waiting_for_complete_joint768')
    run()
except BaseException as exc:
    save('transition_failed', error=f'{type(exc).__name__}: {exc}')
    raise
