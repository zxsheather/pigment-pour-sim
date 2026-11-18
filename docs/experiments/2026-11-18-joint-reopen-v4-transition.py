"""Start the validated reopen-data run if V2 still cannot make loaded contact."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml


ROOT = Path('/data/hlou/pigment-pour-study')
OLD = ROOT / 'runs/cylinder-joint-grasp-recovery-bc-v2'
STAGED = ROOT / 'runs/cylinder-joint-reopen-data-check-v4'
NEW = ROOT / 'runs/cylinder-joint-reopen-bc-v4'
SCRIPT = ROOT / 'pigment-pour-sim/simulation/train_and_evaluate_cylinder_joints.py'
LOG = ROOT / 'results/cylinder-joint-reopen-bc-v4.log'
STATUS = ROOT / 'results/cylinder-joint-reopen-v4-transition.json'
OLD_PIDS = {1422335: 'train_and_evaluate_cylinder_joints.py',
            1426784: 'torch.distributed.run', 1426811: 'train_fsdp.py'}
TARGET_STEP = 512
STATE = dict(pid=os.getpid(), target_step=TARGET_STEP, old_driver_pid=1422335)


def read(path):
    return json.loads(path.read_text())


def save(stage, **values):
    STATE.update(stage=stage, time=time.time(), **values)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, indent=2) + '\n')
    temp.replace(STATUS)


def command(pid):
    row = subprocess.run(['ps', '-p', str(pid), '-o', 'state=,args='],
                         text=True, capture_output=True).stdout.strip()
    return row if row and not row.startswith('Z') else ''


def intact_processes():
    return all(expected in command(pid) and str(OLD) in command(pid)
               for pid, expected in OLD_PIDS.items())


def completed_cases():
    rows = []
    for step in range(128, TARGET_STEP + 1, 128):
        result_dir = ROOT / 'runs' / f'{OLD.name}-step{step}-single'
        report = read(result_dir / 'summary.json')
        assert report['evaluation_complete'] and report['total'] == 1 and report['action_space'] == 'joint'
        episode = report['episodes'][0]
        assert episode['geometry_audit']['passed'] and episode['geometry_audit']['custom_pads'] == 0
        with np.load(result_dir / 'episode_000/trajectory.npz') as trajectory:
            contacts = int(trajectory['contacts'].max(initial=0))
        rows.append(dict(step=step, success=episode['success'], reason=episode['failure_reason'],
                         held=episode['final_task_gates']['held'], max_loaded_contacts=contacts,
                         max_lift_m=episode['max_lifted_m']))
    return rows


def main():
    inputs = read(STAGED / 'summary.json')
    loader = read(STAGED / 'data-validation.json')
    cfg = yaml.safe_load((STAGED / 'train.yml').read_text())
    assert inputs['episodes'] == loader['episodes'] == 59 and inputs['frames'] == loader['frames'] == 36082
    assert loader['passed'] and all(row['demonstration_replay_success'] for row in inputs['decoder_replays'])
    assert not inputs['used_independent_cases'] and not inputs['neural_training_started']
    assert cfg['data']['lerobot_config']['root'] == inputs['dataset'] == loader['dataset']
    assert hashlib.sha256(Path(cfg['data']['norm_stats_path']).read_bytes()).hexdigest() == inputs['norm_stats_sha256']
    assert not NEW.exists() and not LOG.exists()
    save('waiting_for_v2_512_closed_loop')
    for _ in range(540):
        status = read(OLD / 'execution-status.json')
        if (OLD / 'candidate-ready.json').exists() or status['stage'] == 'candidate_requires_full_suites':
            save('kept_v2_complete_single_candidate')
            return
        if status.get('latest_evaluated_step', 0) >= TARGET_STEP:
            break
        assert intact_processes(), 'Expected the live V2 coordinator and trainer'
        time.sleep(5)
    else:
        save('wait_window_expired_v2_left_running')
        return
    rows = completed_cases()
    if any(r['success'] or r['held'] or r['max_loaded_contacts'] or r['max_lift_m'] >= .04 for r in rows):
        save('kept_v2_grasp_progress', results=rows)
        return
    if status['stage'] != 'training' or status['latest_evaluated_step'] != TARGET_STEP:
        save('kept_v2_stage_advanced', observed=status['stage'], results=rows)
        return
    parent = OLD / f'checkpoints/0_{TARGET_STEP}'
    audit = read(OLD / f'checkpoint-{TARGET_STEP}-weight-audit.json')
    assert audit['passed'] and Path(audit['checkpoint']) == parent
    assert (parent / 'model.safetensors').stat().st_size > 7 * 1024**3
    assert hashlib.sha256((parent / 'norm_stats.json').read_bytes()).hexdigest() == inputs['norm_stats_sha256']
    cfg['model']['processor_path'] = str(parent)
    cfg['checkpoint'].update(resume_from=str(parent / 'model.safetensors'), save_path=str(NEW / 'checkpoints'))
    cfg['logging']['log_name'] = NEW.name
    NEW.mkdir()
    (NEW / 'train.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    status = read(OLD / 'execution-status.json')
    if status['stage'] != 'training' or status['latest_evaluated_step'] != TARGET_STEP or (OLD / 'candidate-ready.json').exists():
        save('kept_v2_state_changed', results=rows, observed=status['stage'])
        return
    assert intact_processes()
    save('stopping_v2_after_completed_512', results=rows, parent=str(parent),
         parent_sha256=audit['checkpoint_sha256'])
    os.kill(1422335, signal.SIGTERM)
    for _ in range(45):
        if not any(command(pid) for pid in OLD_PIDS):
            break
        time.sleep(2)
    else:
        save('did_not_start_v4_v2_process_remains')
        return
    argv = [str(ROOT / '.venv-wall/bin/python'), str(SCRIPT), '--run', str(NEW),
            '--parent', str(parent), '--log', str(LOG)]
    with LOG.open('x') as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    save('v4_driver_started', driver_pid=process.pid, argv=argv)


if __name__ == '__main__':
    if STATUS.exists():
        raise SystemExit('Transition already recorded')
    try:
        main()
    except BaseException as exc:
        save('failed', error=f'{type(exc).__name__}: {exc}')
        raise
