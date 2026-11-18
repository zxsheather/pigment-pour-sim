"""Start the staged grasp-focused sampling run if V4 still misses the grasp."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml


ROOT = Path('/data/hlou/pigment-pour-study')
OLD = ROOT / 'runs/cylinder-joint-reopen-bc-v4'
STAGED = ROOT / 'runs/cylinder-joint-grasp-sampling-data-check-v5'
NEW = ROOT / 'runs/cylinder-joint-grasp-sampling-bc-v5'
SCRIPTS = ROOT / 'pigment-pour-sim/simulation'
LOG = ROOT / 'results/cylinder-joint-grasp-sampling-bc-v5.log'
STATUS = ROOT / 'results/cylinder-joint-grasp-sampling-v5-transition.json'
DIAGNOSTIC = ROOT / 'runs/cylinder-joint-reopen-v4-step128-diagnostics'
OLD_PIDS = {1868039: 'train_and_evaluate_cylinder_joints.py',
            1872486: 'torch.distributed.run', 1872691: 'train_fsdp.py'}
TARGET_STEP = 256
STATE = dict(pid=os.getpid(), target_step=TARGET_STEP, old_driver_pid=1868039)


def read(path):
    return json.loads(path.read_text())


def save(stage, **values):
    STATE.update(stage=stage, updated_at=time.time(), **values)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, indent=2) + '\n')
    temp.replace(STATUS)


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
    except FileNotFoundError:
        return ''


def intact():
    return all(marker in command(pid) and str(OLD) in command(pid) for pid, marker in OLD_PIDS.items())


def main():
    input_audit = read(STAGED / 'grasp-weights.json')
    loader = read(STAGED / 'data-validation.json')
    cfg = yaml.safe_load((STAGED / 'train.yml').read_text())
    weights = Path(cfg['data']['lerobot_config']['frame_sampling_weights_path'])
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == input_audit['weights_sha256']
    assert loader['passed'] and loader['sampling_weights_match_source_indices']
    assert loader['episodes'] == 59 and loader['frames'] == input_audit['frames'] == 36082
    dataset = Path(loader['dataset'])
    assert str(dataset) == cfg['data']['lerobot_config']['root'] == input_audit['dataset']
    paired = read(DIAGNOSTIC / 'comparison.json')
    recovery_starts = [r for r in paired['rows'] if r['frame'] == 0
                       and 'cylinder-joint-v2-reopen-recovery-v1' in r['episode']]
    errors = [r['trained']['gripper_width_mean_signed_error_first8_mm'] for r in recovery_starts]
    assert len(errors) == 3 and all(r['matched_robot_state_and_rgb'] for r in recovery_starts)
    assert not paired['task_success_measured']
    assert not NEW.exists() and not LOG.exists()
    if sum(error < -10 for error in errors) < 2:
        save('kept_v4_reopen_bias_not_repeated', matched_reopen_errors_mm=errors)
        return
    save('waiting_for_v4_256_closed_loop', matched_reopen_errors_mm=errors)
    for _ in range(540):
        state = read(OLD / 'execution-status.json')
        if (OLD / 'candidate-ready.json').exists() or state['stage'] == 'candidate_requires_full_suites':
            save('kept_v4_full_single_candidate')
            return
        if state.get('latest_evaluated_step', 0) >= TARGET_STEP:
            break
        assert intact(), 'V4 coordinator or trainer is missing'
        time.sleep(5)
    else:
        save('wait_expired_v4_left_running')
        return
    if state['stage'] != 'training' or state['latest_evaluated_step'] != TARGET_STEP:
        save('kept_v4_stage_advanced', observed=state['stage'])
        return
    rollout = ROOT / 'runs' / f'{OLD.name}-step{TARGET_STEP}-single'
    summary = read(rollout / 'summary.json')
    assert summary['evaluation_complete'] and summary['action_space'] == 'joint' and summary['total'] == 1
    case = summary['episodes'][0]
    assert case['geometry_audit']['passed'] and case['geometry_audit']['custom_pads'] == 0
    if case['success'] or case['final_task_gates']['held'] or case['max_lifted_m'] >= .04:
        save('kept_v4_physical_grasp_progress')
        return
    output = STATUS.with_name('cylinder-joint-reopen-v4-step256-approach.json')
    subprocess.run([str(ROOT / '.venv-sim/bin/python'), str(SCRIPTS / 'audit_cylinder_grasp_approach.py'),
        '--rollouts', str(rollout), '--dataset', str(dataset), '--reference-volume', '25',
        '--output', str(output)], cwd=ROOT, check=True, timeout=45,
        env=os.environ | dict(MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1'))
    first = read(DIAGNOSTIC / 'approach.json')['policy_results'][0]
    second = read(output)['policy_results'][0]
    if any(row['maximum_recorded_loaded_contacts'] or row['nearest']['grasp_distance_m'] <= .05
           for row in (first, second)):
        save('kept_v4_contact_or_close_approach', approach=str(output))
        return
    parent = OLD / f'checkpoints/0_{TARGET_STEP}'
    audit = read(OLD / f'checkpoint-{TARGET_STEP}-weight-audit.json')
    assert audit['passed'] and Path(audit['checkpoint']) == parent
    assert (parent / 'model.safetensors').stat().st_size > 7 * 1024**3
    norm = Path(cfg['data']['norm_stats_path'])
    assert norm.read_bytes() == (parent / 'norm_stats.json').read_bytes()
    cfg['model']['processor_path'] = str(parent)
    cfg['checkpoint'].update(resume_from=str(parent / 'model.safetensors'), save_path=str(NEW / 'checkpoints'))
    cfg['logging']['log_name'] = NEW.name
    NEW.mkdir()
    (NEW / 'train.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    state = read(OLD / 'execution-status.json')
    if state['stage'] != 'training' or state['latest_evaluated_step'] != TARGET_STEP or (OLD / 'candidate-ready.json').exists():
        save('kept_v4_state_changed', observed=state['stage'])
        return
    assert intact()
    save('stopping_v4_after_completed_256', parent=str(parent), parent_sha256=audit['checkpoint_sha256'],
         approach=str(output), nearest_distances_m=[r['nearest']['grasp_distance_m'] for r in (first, second)])
    os.kill(1868039, signal.SIGTERM)
    for _ in range(45):
        if not any(command(pid) for pid in OLD_PIDS):
            break
        time.sleep(2)
    else:
        save('did_not_start_v5_v4_process_remains')
        return
    argv = [str(ROOT / '.venv-wall/bin/python'), str(SCRIPTS / 'train_and_evaluate_cylinder_joints.py'),
            '--run', str(NEW), '--parent', str(parent), '--log', str(LOG)]
    with LOG.open('x') as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    save('v5_driver_started', driver_pid=process.pid, argv=argv)


if __name__ == '__main__':
    if STATUS.exists():
        raise SystemExit('Transition already recorded')
    try:
        main()
    except BaseException as exc:
        save('failed', error=f'{type(exc).__name__}: {exc}')
        raise
