"""Start a checked joint-policy run after two pre-grasp failures."""
import argparse
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
FIRST_APPROACH = None
OLD_PIDS = {1868039: 'train_and_evaluate_cylinder_joints.py',
            1872486: 'torch.distributed.run', 1872691: 'train_fsdp.py'}
TARGET_STEP = 256
STATE = dict(pid=os.getpid(), target_step=TARGET_STEP, old_driver_pid=1868039)


def read(path):
    for attempt in range(4):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(.15)


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


def configure():
    global OLD, STAGED, NEW, LOG, STATUS, DIAGNOSTIC, FIRST_APPROACH, OLD_PIDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-run', type=Path)
    parser.add_argument('--staged-run', type=Path)
    parser.add_argument('--new-run', type=Path)
    first = parser.add_mutually_exclusive_group()
    first.add_argument('--diagnostics', type=Path)
    first.add_argument('--first-approach', type=Path, help='Use a verified step-128 approach report for a data-only change')
    parser.add_argument('--status', type=Path)
    parser.add_argument('--trainer-pid', type=int)
    args = parser.parse_args()
    overrides = (args.active_run, args.staged_run, args.new_run, args.diagnostics or args.first_approach,
                 args.status, args.trainer_pid)
    if all(value is None for value in overrides):
        return
    if any(value is None for value in overrides):
        parser.error('Pass all run overrides together')
    OLD, STAGED, NEW, evidence, STATUS = (path.resolve() for path in overrides[:-1])
    assert OLD.parent == STAGED.parent == NEW.parent == ROOT / 'runs'
    assert STATUS.parent == ROOT / 'results'
    if args.first_approach:
        assert evidence.parent == ROOT / 'results'
        FIRST_APPROACH = evidence
    else:
        assert evidence.parent == ROOT / 'runs'
        DIAGNOSTIC = evidence
    LOG = ROOT / 'results' / f'{NEW.name}.log'
    active = read(OLD / 'execution-status.json')
    OLD_PIDS = {active['driver_pid']: 'train_and_evaluate_cylinder_joints.py',
                active['training_pid']: 'torch.distributed.run', args.trainer_pid: 'train_fsdp.py'}
    assert intact()
    STATE.update(old_driver_pid=active['driver_pid'], active_run=str(OLD), staged_run=str(STAGED))


def first_rollout():
    path = FIRST_APPROACH or DIAGNOSTIC / 'approach.json'
    report = read(path)
    assert not report['new_policy_inference_run'] and not report['independent_cases_read']
    row, = report['policy_results']
    if FIRST_APPROACH:
        rollout = ROOT / 'runs' / f'{OLD.name}-step128-single'
        summary = read(rollout / 'summary.json')
        case, = summary['episodes']
        assert row['rollout'] == str(rollout) and row['robot_observations_reproduced_exactly']
        assert summary['evaluation_complete'] and summary['action_space'] == 'joint' and summary['total'] == 1
        assert case['geometry_audit']['passed'] and case['geometry_audit']['custom_pads'] == 0
        assert not case['success'] and not case['final_task_gates']['held'] and case['max_lifted_m'] < .04
        assert row['failure_reason'] == case['failure_reason']
        assert read(OLD / 'checkpoint-128-weight-audit.json')['passed']
    return row


def main():
    input_audit = read(STAGED / 'grasp-weights.json')
    loader = read(STAGED / 'data-validation.json')
    cfg = yaml.safe_load((STAGED / 'train.yml').read_text())
    weights = Path(cfg['data']['lerobot_config']['frame_sampling_weights_path'])
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == input_audit['weights_sha256']
    assert loader['passed'] and loader['sampling_weights_match_source_indices']
    assert loader['frames'] == input_audit['frames']
    if (STAGED / 'summary.json').exists():
        dataset_audit = read(STAGED / 'summary.json')
        assert dataset_audit['frames'] == loader['frames'] and dataset_audit['episodes'] == loader['episodes']
        assert dataset_audit['data_loader_passed']
        assert all(row['demonstration_replay_success'] and row['geometry_audit_passed']
                   for row in dataset_audit['teacher_decoder_replays'])
    else:
        assert loader['episodes'] == 59 and loader['frames'] == 36082
    dataset = Path(loader['dataset'])
    assert str(dataset) == cfg['data']['lerobot_config']['root'] == input_audit['dataset']
    assert not NEW.exists() and not LOG.exists()
    if FIRST_APPROACH:
        first = first_rollout()
        evidence = dict(first_approach=str(FIRST_APPROACH), first_nearest_m=first['nearest']['grasp_distance_m'])
        if first['maximum_recorded_loaded_contacts'] or evidence['first_nearest_m'] <= .05:
            save('kept_active_contact_or_close_approach', **evidence)
            return
    else:
        paired = read(DIAGNOSTIC / 'comparison.json')
        recovery_starts = [r for r in paired['rows'] if r['frame'] == 0
                           and 'cylinder-joint-v2-reopen-recovery-v1' in r['episode']]
        errors = [r['trained']['gripper_width_mean_signed_error_first8_mm'] for r in recovery_starts]
        assert len(errors) == 3 and all(r['matched_robot_state_and_rgb'] for r in recovery_starts)
        assert not paired['task_success_measured']
        evidence = dict(matched_reopen_errors_mm=errors)
        if sum(error < -10 for error in errors) < 2:
            save('kept_active_reopen_bias_not_repeated', **evidence)
            return
        first = first_rollout()
    save('waiting_for_active_256_closed_loop', **evidence)
    for _ in range(540):
        state = read(OLD / 'execution-status.json')
        if (OLD / 'candidate-ready.json').exists() or state['stage'] == 'candidate_requires_full_suites':
            save('kept_active_full_single_candidate')
            return
        if state.get('latest_evaluated_step', 0) >= TARGET_STEP:
            break
        assert intact(), 'Active coordinator or trainer is missing'
        time.sleep(5)
    else:
        save('wait_expired_active_left_running')
        return
    if state['stage'] != 'training' or state['latest_evaluated_step'] != TARGET_STEP:
        save('kept_active_stage_advanced', observed=state['stage'])
        return
    rollout = ROOT / 'runs' / f'{OLD.name}-step{TARGET_STEP}-single'
    summary = read(rollout / 'summary.json')
    assert summary['evaluation_complete'] and summary['action_space'] == 'joint' and summary['total'] == 1
    case = summary['episodes'][0]
    assert case['geometry_audit']['passed'] and case['geometry_audit']['custom_pads'] == 0
    if case['success'] or case['final_task_gates']['held'] or case['max_lifted_m'] >= .04:
        save('kept_active_physical_grasp_progress')
        return
    output = STATUS.with_name(f"{OLD.name.replace('-bc-', '-')}-step256-approach.json")
    subprocess.run([str(ROOT / '.venv-sim/bin/python'), str(SCRIPTS / 'audit_cylinder_grasp_approach.py'),
        '--rollouts', str(rollout), '--dataset', str(dataset), '--reference-volume', '25',
        '--output', str(output)], cwd=ROOT, check=True, timeout=45,
        env=os.environ | dict(MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1'))
    second = read(output)['policy_results'][0]
    if any(row['maximum_recorded_loaded_contacts'] or row['nearest']['grasp_distance_m'] <= .05
           for row in (first, second)):
        save('kept_active_contact_or_close_approach', approach=str(output))
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
        save('kept_active_state_changed', observed=state['stage'])
        return
    assert intact()
    save('stopping_active_after_completed_256', parent=str(parent), parent_sha256=audit['checkpoint_sha256'],
         approach=str(output), nearest_distances_m=[r['nearest']['grasp_distance_m'] for r in (first, second)])
    os.kill(STATE['old_driver_pid'], signal.SIGTERM)
    for _ in range(45):
        if not any(command(pid) for pid in OLD_PIDS):
            break
        time.sleep(2)
    else:
        save('new_run_skipped_active_process_remains')
        return
    argv = [str(ROOT / '.venv-wall/bin/python'), str(SCRIPTS / 'train_and_evaluate_cylinder_joints.py'),
            '--run', str(NEW), '--parent', str(parent), '--log', str(LOG)]
    with LOG.open('x') as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    save('new_driver_started', driver_pid=process.pid, argv=argv)


if __name__ == '__main__':
    configure()
    if STATUS.exists():
        raise SystemExit('Transition already recorded')
    try:
        main()
    except BaseException as exc:
        save('failed', error=f'{type(exc).__name__}: {exc}')
        raise
