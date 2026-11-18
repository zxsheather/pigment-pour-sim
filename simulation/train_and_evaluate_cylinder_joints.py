"""Train a Wall cylinder policy and evaluate complete checkpoints.

Keeps parent, latest and best development-case weights, one latest optimizer,
and all small experiment records. A full single-case candidate stops training
but still requires original8/development12/independent12; it is not goal success.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from audit_cylinder_grasp_approach import measure_rollout
from cylinder_joint_actions import JOINT_ACTION_LAYOUT, JOINT_STATE_LAYOUT, require


def development_score(successes, case, approach):
    gates = case['final_task_gates']
    stages = ('held', 'lifted', 'retreated', 'close_to_head')
    completed = next((i for i, gate in enumerate(stages) if not gates[gate]), len(stages))
    if completed == 0:
        return (successes, completed, approach['maximum_recorded_loaded_contacts'],
                -approach['nearest']['grasp_distance_m'])
    return (successes, completed, -case['max_slip_m'],
            min(.15, case['max_lifted_m']) if completed >= 2 else 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--data-port', type=int, default=29946)
    parser.add_argument('--train-port', type=int, default=29948)
    parser.add_argument('--action-space', choices=('joint', 'cartesian'), default='joint')
    parser.add_argument('--training-wall-root', type=Path,
                        help='Wall-X source copy used by the trainer subprocess.')
    args = parser.parse_args()
    run, parent = args.run.resolve(), args.parent.resolve()
    config = run/'train.yml'
    cfg = yaml.safe_load(config.read_text())
    action_space = args.action_space
    if action_space == 'joint':
        require(list(cfg['task']['dof_config'].items()) == list(JOINT_ACTION_LAYOUT.items())
                and list(cfg['task']['agent_pos_config'].items()) == list(JOINT_STATE_LAYOUT.items()),
                'Expected the joint action/state schema')
        validator = 'validate_wall_cylinder_joints.py'
        norm_key = 'cylinder_joint_v1'
    else:
        require(cfg['task']['dof_config'].get('follow_right_ee_cartesian_pos_relative') == 3
                and cfg['task']['dof_config'].get('follow_right_ee_rotation_6D_relative') == 6
                and cfg['task']['agent_pos_config'].get('base_odometry') == 3,
                'Expected Cartesian targets with base odometry')
        validator = 'validate_wall_data.py'
        norm_key = 'cylinder_cartesian_v1'
    steps = cfg['hyperparams']['scheduler']['num_training_steps']
    interval = cfg['logging']['save_interval']
    require(steps > interval and (steps-1) % interval == 0, 'Final checkpoint must be scheduled')
    resume = Path(cfg['checkpoint']['resume_from']).resolve()
    resume_step = 0
    if resume.is_dir():
        import torch
        for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pt',
                     'global_step.pth', 'current_epoch.pth'):
            require((resume/name).is_file(), f'Missing resume checkpoint component: {name}')
        resume_step = torch.load(resume/'global_step.pth', map_location='cpu', weights_only=True)['global_step']
        require(0 < resume_step < steps-1 and resume_step % interval == 0 and resume.name == f'0_{resume_step}',
                'Expected a complete scheduled checkpoint from epoch zero')
        earlier = resume.parent.parent/f'checkpoint-{resume_step}-weight-audit.json'
        proof = json.loads(earlier.read_text())
        require(proof['passed'] and proof['checkpoint'] == str(resume), 'Resume checkpoint audit failed')
    else:
        require(resume == parent/'model.safetensors', 'Parent/config mismatch')
    require(not (run/'execution-status.json').exists(), 'Existing execution must be inspected, not restarted')
    scripts = Path(__file__).resolve().parent
    root = Path.cwd()
    python = str(root/'.venv-wall/bin/python')
    os.environ.update(PYTHONPATH=str(root/'wall-x'), MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1',
                       TOKENIZERS_PARALLELISM='false')
    trainer_entry = Path('wall-x/wall_x/trainer/fsdp_trainer/train_fsdp.py')
    trainer_env = None
    if args.training_wall_root is not None:
        training_wall = args.training_wall_root.resolve()
        trainer_entry = training_wall / 'wall_x/trainer/fsdp_trainer/train_fsdp.py'
        require(trainer_entry.is_file(), 'Missing training entry point in the Wall-X source copy')
        trainer_env = os.environ | dict(PYTHONPATH=str(training_wall))
    state = dict(stage='data_validation', started=time.time(), driver_pid=os.getpid(),
                 parent=str(parent), action_space=action_space, history=[], goal_complete=False,
                 resumed_checkpoint=str(resume) if resume_step else None, resumed_step=resume_step)
    trainer = None
    active_call = None

    def save():
        (run/'execution-status.json').write_text(json.dumps(state, indent=2)+'\n')

    def call(argv, stage):
        nonlocal active_call
        state['stage'] = stage
        print('START', argv, flush=True)
        process = subprocess.Popen(argv)
        active_call = process
        row = dict(stage=stage, argv=argv, pid=process.pid, started=time.time())
        state['history'].append(row);save()
        row.update(exit_code=process.wait(), finished=time.time());save()
        active_call = None
        require(row['exit_code'] == 0, f'{stage} failed; inspect the log before any restart')

    def stop_trainer():
        if trainer is None or trainer.poll() is not None:
            return
        trainer.send_signal(signal.SIGINT)
        try:trainer.wait(timeout=45)
        except subprocess.TimeoutExpired:
            trainer.terminate()
            try:trainer.wait(timeout=10)
            except subprocess.TimeoutExpired:trainer.kill();trainer.wait(timeout=5)
        state['training_exit_code'] = trainer.returncode
        save()

    def interrupted(signum, frame):
        raise SystemExit(128+signum)

    for sig in (signal.SIGTERM, signal.SIGINT):signal.signal(sig, interrupted)
    save()
    try:
        episodes=json.loads((Path(cfg['data']['lerobot_config']['root'])/'meta/info.json').read_text())['total_episodes']
        call([python, '-m', 'torch.distributed.run', '--nproc_per_node=1',
              f'--master_port={args.data_port}', str(scripts/validator),
              '--config', str(config), '--check-episode', '0', '--check-episode', str(episodes//2),
              '--check-episode', str(episodes-1), '--output', str(run/'data-validation.json')], 'data_validation')
        training = [python, '-m', 'torch.distributed.run', '--nproc_per_node=1',
                    f'--master_port={args.train_port}', str(trainer_entry),
                    '--config', str(config)]
        trainer = subprocess.Popen(training, env=trainer_env)
        state.update(stage='training', training_pid=trainer.pid, training_argv=training);save()
        visited = []
        best = None
        best_score = None
        for step in range(resume_step + interval, steps, interval):
            while f'at global_step={step}' not in args.log.read_text(errors='replace'):
                if trainer.poll() is not None:
                    raise RuntimeError(f'Training ended before expected checkpoint{step}; exit={trainer.returncode}')
                time.sleep(10)
            checkpoint = run/f'checkpoints/0_{step}'
            output = root/f'runs/{run.name}-step{step}-single'
            audit = [python, str(scripts/'audit_wall_checkpoint.py'), '--base', str(parent),
                     '--checkpoint', str(checkpoint), '--output', str(run/f'checkpoint-{step}-weight-audit.json')]
            if action_space == 'cartesian':
                audit.append('--allow-new-proprio-projection')
            call(audit,
                 f'audit_{step}')
            call([python, str(scripts/'run_wall_cylinder.py'), '--model', str(checkpoint),
                  '--config', str(config), '--norm-key', norm_key,
                  '--controller-config', str(scripts/'configs/cylinder_official_teacher_dev_v1.json'),
                  '--volume', '25', '--seed', '100', '--execute-steps', '8', '--max-seconds', '90',
                  '--record', '--output', str(output)], f'neural_evaluation_{step}')
            report = json.loads((output/'summary.json').read_text())
            require(report['evaluation_complete'] and report['action_space'] == action_space
                    and report['total'] == 1 and report['gripper_open_confirmations'] == 0,
                    'Incomplete or unexpected policy evaluation')
            approach = measure_rollout(output)
            (output/'grasp-approach-audit.json').write_text(json.dumps(approach, indent=2)+'\n')
            score = development_score(report['successes'], report['episodes'][0], approach)
            if best_score is None or score > best_score:
                best, best_score = checkpoint, score
            visited.append(checkpoint)
            (run/'best-development-case.json').write_text(json.dumps(dict(model=str(best), score=best_score,
                scope='One development case only; all full-suite gates still required',
                ranking='Success, sequential task stages, then contacts and grasp distance before holding or slip and lift after holding'),
                indent=2)+'\n')
            active = subprocess.run(['ps', '-eo', 'args='], capture_output=True, text=True).stdout
            for previous in visited[:-1]:
                if any('python' in line and str(previous) in line for line in active.splitlines()):continue
                for name in ('model.safetensors', 'optimizer.pt'):
                    file = previous/name
                    if name == 'model.safetensors' and previous == best:continue
                    if file.exists() and not file.is_symlink():
                        row = dict(path=str(file), bytes=file.stat().st_size, time=time.time(),
                                   reason='Completed evaluated checkpoint superseded; latest optimizer retained')
                        file.unlink()
                        with (run/'checkpoint-retention.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            state.update(stage='training', latest_evaluated_step=step, best_model=str(best));save()
            if report['successes']:
                (run/'candidate-ready.json').write_text(json.dumps(dict(model=str(checkpoint), step=step, action_space=action_space,
                    single_summary=str(output/'summary.json'), goal_complete=False,
                    requires=['original8', 'development12', 'untouched_independent12']), indent=2)+'\n')
                state['stage'] = 'candidate_requires_full_suites';save()
                stop_trainer()
                return
        state['training_exit_code'] = trainer.wait(timeout=60)
        require(trainer.returncode == 0, 'Training did not finish cleanly')
        state.update(stage='curve_complete_no_full_single_success', finished=time.time());save()
    finally:
        if active_call is not None and active_call.poll() is None:
            active_call.send_signal(signal.SIGINT)
            try:active_call.wait(timeout=45)
            except subprocess.TimeoutExpired:
                active_call.terminate()
                try:active_call.wait(timeout=10)
                except subprocess.TimeoutExpired:active_call.kill();active_call.wait(timeout=5)
        stop_trainer()


if __name__ == '__main__':
    main()
