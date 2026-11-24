"""Compare identical teacher observations after the first v2 rollout ends."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('/data/hlou/pigment-pour-study')
NEW = ROOT / 'runs/cylinder-joint-grasp-recovery-bc-v2'
OLD = ROOT / 'runs/cylinder-joint-position-bc-v1'
SCRIPTS = ROOT / 'pigment-pour-sim/simulation'
DATA = ROOT / 'datasets/cylinder-joint-grasp-recovery-v2'
LOGGER = ROOT / 'results/cylinder-joint-grasp-recovery-bc-v2.log'
DRIVER_PID = 1422335
PARENT = OLD / 'checkpoints/0_768'
CHECKPOINT = NEW / 'checkpoints/0_128'
ROLLOUT = ROOT / 'runs/cylinder-joint-grasp-recovery-bc-v2-step128-single'
TEACHER = SCRIPTS / 'generated/fixed-official-training-v1/volume_25_seed_27'
OUTPUT = ROOT / 'runs/cylinder-joint-grasp-recovery-v2-step128-teacher-contrast'
FRAMES = [0, 53, 131, 173, 216, 352, 440]
STATE = dict(pid=os.getpid(), stage='waiting_for_complete_neural_128', paired_frames=FRAMES,
             teacher=str(TEACHER), uses_independent_suite=False,
             modifies_training=False, neural_diagnostic_is_not_task_success=True)


def save(stage, **values):
    STATE.update(stage=stage, updated_at=time.time(), **values)
    temp = OUTPUT / 'status.tmp'
    temp.write_text(json.dumps(STATE, indent=2) + '\n')
    temp.replace(OUTPUT / 'status.json')


def driver_status():
    return json.loads((NEW / 'execution-status.json').read_text())


def command(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'args='], text=True, capture_output=True).stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, log_name, timeout):
    with (OUTPUT / log_name).open('x') as log:
        log.write(json.dumps(args) + '\n')
        log.flush()
        env = os.environ | {'PYTHONPATH': str(ROOT / 'wall-x'), 'OPENBLAS_NUM_THREADS': '1',
                            'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
                            'HF_HUB_OFFLINE': '1', 'MUJOCO_GL': 'egl'}
        subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def verify_idle_window():
    state = driver_status()
    assert state['stage'] == 'training' and state['latest_evaluated_step'] == 128
    assert not (NEW / 'candidate-ready.json').exists()
    assert 'train_and_evaluate_cylinder_joints.py' in command(DRIVER_PID)
    children = subprocess.check_output(['ps', '--ppid', str(DRIVER_PID), '-o', 'args='], text=True)
    assert 'train_fsdp.py' in children and 'run_wall_cylinder.py' not in children
    lines = LOGGER.read_text(errors='replace').splitlines()
    current = next(line for line in reversed(lines) if 'epoch ' in line and 'time_current_backward_step' in line)
    step = int(re.search(r'iter\s+(\d+)', current).group(1))
    return step


def main():
    OUTPUT.mkdir()
    save('waiting_for_complete_neural_128')
    for _ in range(360):
        st = driver_status()
        if st.get('latest_evaluated_step', 0) >= 128:
            break
        assert st['driver_pid'] == DRIVER_PID and 'train_and_evaluate_cylinder_joints.py' in command(DRIVER_PID)
        time.sleep(5)
    else:
        save('wait_expired', no_neural_diagnostics_launched=True)
        return
    full = json.loads((ROLLOUT / 'summary.json').read_text())
    assert full['evaluation_complete'] and full['total'] == 1
    save('measuring_completed_rollout', rollout_passed=bool(full['successes']))
    run([str(ROOT / '.venv-sim/bin/python'), str(SCRIPTS / 'audit_cylinder_grasp_approach.py'),
         '--rollouts', str(ROLLOUT), '--dataset', str(DATA), '--reference-volume', '25',
         '--output', str(OUTPUT / 'approach-distance.json')], 'approach.log', 40)
    if full['successes']:
        save('formal_candidate_has_priority', approach_measured=True, neural_diagnostics_launched=False)
        return
    evidence = json.loads((NEW / 'checkpoint-128-weight-audit.json').read_text())
    assert evidence['passed'] and Path(evidence['checkpoint']) == CHECKPOINT
    assert sha(PARENT / 'norm_stats.json') == sha(CHECKPOINT / 'norm_stats.json')
    if verify_idle_window() > 220:
        save('insufficient_gap_before_next_checkpoint', approach_measured=True, neural_diagnostics_launched=False)
        return
    for name, weights, config in [('parent768', PARENT, OLD / 'train.yml'),
                                  ('v2_128', CHECKPOINT, NEW / 'train.yml')]:
        step = verify_idle_window()
        if step > 237:
            save('insufficient_gap_before_next_checkpoint', approach_measured=True, next_diagnostic=name)
            return
        save(f'diagnose_{name}', approach_measured=True, trainer_iter=step)
        argv = [str(ROOT / '.venv-wall/bin/python'), str(SCRIPTS / 'evaluate_wall_imitation.py'),
                '--model', str(weights), '--config', str(config), '--episode', str(TEACHER),
                '--joint-dataset', str(DATA), '--norm-key', 'cylinder_joint_v1',
                '--output', str(OUTPUT / f'{name}.json')]
        for frame in FRAMES:
            argv += ['--frame', str(frame)]
        run(argv, f'{name}.log', 105)
    reports = [json.loads((OUTPUT / f'{n}.json').read_text()) for n in ('parent768', 'v2_128')]
    assert all(r['action_space'] == 'joint' and not r['physics_advanced'] for r in reports)
    keys = ['mean_arm_error_first8_rad', 'gripper_width_mean_signed_error_first8_mm',
            'base_vx_mean_signed_error_first8_m_s']
    rows = []
    for frame, a, b in zip(FRAMES, reports[0]['samples'], reports[1]['samples'], strict=True):
        assert frame == a['frame'] == b['frame'] and a['phase'] == b['phase']
        assert a['first_teacher_right_action'] == b['first_teacher_right_action']
        rows.append(dict(frame=frame, phase=a['phase'], parent768={k:a[k] for k in keys},
                         v2_128={k:b[k] for k in keys}))
    (OUTPUT / 'comparison.json').write_text(json.dumps({'scope':'paired saved teacher observations',
        'task_success_measured':False,'rows':rows},indent=2)+'\n')
    save('complete', approach_measured=True, paired_neural_diagnostic_complete=True)


try:
    main()
except BaseException as exc:
    if (OUTPUT / 'status.json').exists():
        save('failed', error=f'{type(exc).__name__}: {exc}')
    raise
