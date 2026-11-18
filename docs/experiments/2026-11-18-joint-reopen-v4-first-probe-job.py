"""Measure a first joint rollout and compare predictions on fixed teacher inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('/data/hlou/pigment-pour-study')
RUN = ROOT / 'runs/cylinder-joint-reopen-bc-v4'
PARENT = ROOT / 'runs/cylinder-joint-grasp-recovery-bc-v2/checkpoints/0_512'
DATASET = ROOT / 'datasets/cylinder-joint-reopen-recovery-v4'
SCRIPTS = ROOT / 'pigment-pour-sim/simulation'
OUTPUT = ROOT / 'runs/cylinder-joint-reopen-v4-step128-diagnostics'
SELECTIONS = ROOT / 'results/2026-11-18-joint-reopen-v4-teacher-frames.json'
BASELINE = ROOT / 'results/cylinder-joint-reopen-v4-teacher-parent512.json'
TRAIN_LOG = ROOT / 'results/cylinder-joint-reopen-bc-v4.log'
DRIVER_PID = 1868039
TRAINER_PID = 1872691
STATE = dict(pid=os.getpid(), first_evaluation_step=128)
ENV = os.environ | dict(PYTHONPATH=str(ROOT / 'wall-x'), MUJOCO_GL='egl',
    HF_HUB_OFFLINE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')


def read(path):
    return json.loads(path.read_text())


def save(stage, **values):
    STATE.update(stage=stage, updated_at=time.time(), **values)
    temporary = OUTPUT / 'status.tmp'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(OUTPUT / 'status.json')


def command(pid):
    proc = Path(f'/proc/{pid}/cmdline')
    try:
        return proc.read_bytes().decode(errors='replace')
    except FileNotFoundError:
        return ''


def call(venv, script, args, log_name, timeout=None):
    argv = [str(ROOT / venv / 'bin/python'), str(SCRIPTS / script), *map(str, args)]
    with (OUTPUT / log_name).open('x') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=timeout, cwd=ROOT, env=ENV)


def compare(before, after):
    assert before['flow_steps'] == after['flow_steps'] == 10
    assert before['flow_seed'] == after['flow_seed'] == 47
    assert before['action_space'] == after['action_space'] == 'joint'
    assert not any(r['task_success_measured'] or r['physics_advanced'] for r in (before, after))
    metrics = ('mean_arm_error_first8_rad', 'mean_predicted_gripper_width_first8_mm',
               'mean_teacher_gripper_width_first8_mm', 'gripper_width_mean_signed_error_first8_mm',
               'base_vx_mean_signed_error_first8_m_s')
    rows = []
    for a, b in zip(before['episodes'], after['episodes'], strict=True):
        assert a['episode'] == b['episode']
        for x, y in zip(a['samples'], b['samples'], strict=True):
            keys = ('frame', 'phase', 'robot_state_sha256', 'image_sha256', 'first_teacher_right_action')
            assert all(x[key] == y[key] for key in keys), 'Training observations differ'
            rows.append(dict(episode=a['episode'], frame=x['frame'], phase=x['phase'],
                matched_robot_state_and_rgb=True, observed_width_mm=x['observed_gripper_width_mm'],
                parent={key: x[key] for key in metrics}, trained={key: y[key] for key in metrics}))
    assert len(rows) == 13
    return dict(kind='paired_teacher_observation_comparison', rows=rows,
                models=[before['model'], after['model']], task_success_measured=False)


def configure():
    global RUN, PARENT, DATASET, OUTPUT, BASELINE, TRAIN_LOG, DRIVER_PID, TRAINER_PID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--trainer-pid', type=int)
    args = parser.parse_args()
    if args.run is None:
        if any(x is not None for x in (args.output, args.baseline, args.trainer_pid)):
            parser.error('Pass --run with --output, --baseline and --trainer-pid')
        return
    if any(x is None for x in (args.output, args.baseline, args.trainer_pid)):
        parser.error('A run override requires --output, --baseline and --trainer-pid')
    RUN, OUTPUT, BASELINE = (x.resolve() for x in (args.run, args.output, args.baseline))
    assert RUN.parent == OUTPUT.parent == ROOT / 'runs'
    status = read(RUN / 'execution-status.json')
    validation = read(RUN / 'data-validation.json')
    assert validation['passed'] and validation['sampling_weights_match_source_indices']
    PARENT = Path(status['parent']).resolve()
    DATASET = Path(validation['dataset']).resolve()
    DRIVER_PID, TRAINER_PID = status['driver_pid'], args.trainer_pid
    TRAIN_LOG = ROOT / 'results' / f'{RUN.name}.log'
    assert str(RUN) in command(DRIVER_PID) and str(RUN) in command(TRAINER_PID)
    assert Path(read(BASELINE)['model']).resolve() == PARENT
    STATE.update(run=str(RUN), dataset=str(DATASET), trainer_pid=TRAINER_PID)


def main():
    configure()
    OUTPUT.mkdir()
    save('waiting_for_completed_neural_rollout')
    for _ in range(540):
        status = read(RUN / 'execution-status.json')
        if status.get('latest_evaluated_step', 0) >= 128:
            break
        assert 'train_and_evaluate_cylinder_joints.py' in command(DRIVER_PID), 'Training coordinator exited'
        time.sleep(5)
    else:
        save('wait_expired_without_starting_diagnostics')
        return
    rollout = ROOT / 'runs' / f'{RUN.name}-step128-single'
    summary = read(rollout / 'summary.json')
    assert summary['evaluation_complete'] and summary['total'] == 1
    save('measuring_completed_neural_rollout', full_single_pass=bool(summary['successes']))
    call('.venv-sim', 'audit_cylinder_grasp_approach.py',
         ['--rollouts', rollout, '--dataset', DATASET, '--reference-volume', '25',
          '--output', OUTPUT / 'approach.json'], 'approach.log', timeout=45)
    if summary['successes']:
        candidate = RUN / 'candidate-ready.json'
        for _ in range(45):
            if candidate.exists() and not command(TRAINER_PID) and not command(status['training_pid']):
                break
            time.sleep(3)
        else:
            save('candidate_needs_acceptance_after_gpu_frees')
            return
        target = ROOT / 'runs' / f'{RUN.name}-step128-acceptance'
        save('running_guarded_acceptance', output=str(target))
        call('.venv-wall', 'evaluate_cylinder_joint_candidate.py',
             ['--candidate', candidate, '--output', target], 'acceptance.log')
        save('acceptance_finished_needs_goal_audit')
        return
    status = read(RUN / 'execution-status.json')
    iteration = int(re.findall(r'\biter\s+(\d+)\s*/', TRAIN_LOG.read_text())[-1])
    if status['stage'] != 'training' or status['latest_evaluated_step'] != 128 or iteration > 225:
        save('rollout_measured_no_diagnostic_gap', iteration=iteration)
        return
    assert 'train_fsdp.py' in command(TRAINER_PID)
    assert 'train_and_evaluate_cylinder_joints.py' in command(DRIVER_PID)
    assert not command(next(row['pid'] for row in status['history'] if row['stage']=='neural_evaluation_128'))
    checkpoint = RUN / 'checkpoints/0_128'
    assert read(RUN / 'checkpoint-128-weight-audit.json')['passed']
    assert (PARENT / 'norm_stats.json').read_bytes() == (checkpoint / 'norm_stats.json').read_bytes()
    assert BASELINE.is_file(), 'Parent teacher diagnostic missing'
    before = read(BASELINE)
    assert Path(before['model']) == PARENT
    save('running_trained_teacher_diagnostic', iteration=iteration,
         baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest())
    trained = OUTPUT / 'trained.json'
    call('.venv-wall', 'evaluate_wall_imitation.py',
         ['--model', checkpoint, '--config', RUN / 'train.yml', '--norm-key', 'cylinder_joint_v1',
          '--joint-dataset', DATASET, '--selections', SELECTIONS, '--output', trained],
         'trained.log', timeout=155)
    comparison = compare(before, read(trained))
    (OUTPUT / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    save('complete', paired_rows=len(comparison['rows']))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUTPUT.joinpath('status.json').exists():
            save('failed', error=f'{type(exc).__name__}: {exc}')
        raise
