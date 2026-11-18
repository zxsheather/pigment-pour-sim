"""Run the complete original8/development12/independent12 neural acceptance.

Requires a genuinely successful, audited single-case candidate. The untouched
suite is reserved in a durable ledger only after both development suites pass.
No training, teacher control, modified task gates, or video frame cache is used.
This script produces evidence; it never marks the Codex goal complete.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SDK_COMMIT = '63610199bbd70eafd1fc087a015ec5c74a0c906c'
FINGER_HASHES = {
    '3e43838c5023aa6bda5d6a339535d806c74461548c6c97befc8c0bf05be166ea': 3,
    '86c16517b8d624d0349615a0f11d7a1573ac48bb98ff112ab979d9e3e2cdc039': 2,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def canonical_case(case):
    defaults = dict(jitter=.004, yaw_jitter=0., friction_scale=1., mass_scale=1.)
    return defaults | case


def case_set_digest(cases):
    """Treat reordered/reformatted copies as the same consumed test cases."""
    normalized = []
    for case in cases:
        case = canonical_case(case)
        require(set(case) == {'volume', 'seed', 'jitter', 'yaw_jitter', 'friction_scale', 'mass_scale'},
                'Unexpected case fields')
        require(int(case['seed']) == case['seed'], 'Seed must be an integer')
        values = {key: int(value) if key == 'seed' else float(value) for key, value in case.items()}
        require(all(math.isfinite(value) for value in values.values()), 'Nonfinite case parameters')
        normalized.append(json.dumps(values, sort_keys=True, allow_nan=False))
    require(len(set(normalized)) == len(normalized), 'Duplicate cases in suite')
    return hashlib.sha256(json.dumps(sorted(normalized)).encode()).hexdigest()


def verify_report(report, cases, model, config, controller, source_hashes, *, action_space='joint'):
    require(report['evaluation_complete'] and not report['probe_only'], 'Incomplete/probe evaluation')
    require(action_space in ('joint', 'cartesian') and report['action_space'] == action_space,
            'Unexpected neural action space')
    require(report['model'] == str(model) and report['config'] == str(config), 'Candidate/config mismatch')
    require(report['controller'] == controller, 'Physical controller configuration changed')
    settings = dict(execution_prefix=8, policy_batch_size=1, flow_steps=10, flow_noise='random',
                    action_filter_seconds=0., temporal_ensemble_predictions=0,
                    prediction_samples_per_observation=1, state_embedding_precision='training',
                    gripper_open_confirmations=0, unclipped_robot_state=False)
    require(all(report[k] == v for k, v in settings.items()), 'Unexpected neural execution settings')
    expected_ik_limit = None if action_space == 'joint' else .002
    require(report['ik_max_residual'] == expected_ik_limit, 'Unexpected inverse kinematics setting')
    require(report['robot_servo_envelope'] == dict(cartesian_speed_limit_m_s=0., angular_speed_limit_rad_s=0.),
            'Unexpected extra servo envelope')
    require(report['total'] == len(cases) == len(report['episodes']), 'Case count mismatch')
    require(report['successes'] == len(cases), 'A full task case failed; later suites will not run')
    for case, result in zip(cases, report['episodes']):
        require(result['policy'] == 'wall_oss_full_cylinder' and result['action_space'] == action_space,
                'Expected genuine neural policy output')
        if action_space == 'cartesian':
            ik = result['ik_projection']
            require(ik['applicable'] and ik['max_allowed_weighted_residual'] == expected_ik_limit
                    and 0 <= ik['max_attempted_residual'] <= expected_ik_limit,
                    'Cartesian inverse kinematics gate failed')
        require(canonical_case(result['case']) == canonical_case(case), 'Case order/parameters changed')
        require(result['success'] and not result['evaluation_interrupted']
                and result['failure_reason'] is None, 'Case did not complete successfully')
        require(result['source_sha256'] == source_hashes, 'Execution source changed since candidate')
        geometry = result['geometry_audit']
        require(geometry['passed'] and geometry['custom_pads'] == 0
                and geometry['source_commit'] == SDK_COMMIT, 'Official geometry audit failed')
        require(len(geometry['fingers']) == 4, 'Incomplete finger audit')
        require(all(sum(finger['sha256'] == sha for finger in geometry['fingers']) == 2
                    for sha in FINGER_HASHES), 'Expected the two official finger meshes on both arms')
        for finger in geometry['fingers']:
            require(finger['sha256'] in FINGER_HASHES
                    and finger['convex_solids'] == finger['compiled_contact_geoms'] == FINGER_HASHES[finger['sha256']]
                    and finger['urdf_effort_limit_n'] == 20., 'Finger geometry/force changed')
        require(geometry['contact_model']['friction'] == [1., .005, .0001]
                and geometry['contact_model']['condim'] == 3
                and geometry['contact_model']['noslip_iterations'] == 0, 'Contact model changed')
        gates = result['final_task_gates']
        require(all(gates[k] for k in ('held', 'lifted', 'retreated', 'close_to_head', 'upright', 'still')),
                'Incomplete physical task gates')
        require(result['max_lifted_m'] >= .04 and result['max_slip_m'] <= .025
                and result['max_held_tilt_rad'] <= .12 and result['final_tilt_rad'] <= math.radians(3)
                and gates['base_retreat_m'] >= .20 and gates['camera_distance_m'] < .45
                and gates['linear_speed_m_s'] <= .005 and gates['angular_speed_rad_s'] <= .03
                and result['simulated_seconds'] <= 90.000001, 'Physical numeric gate failed')
        last = result['reading_history'][-3:]
        require(len(last) == 3 and all(x['valid'] and abs(x['eye_alignment_error_px']) <= 2
                and x.get('boundary_support_fraction', 0) >= .5 for x in last), 'Three valid RGB reads required')
        require(all(b['time'] - a['time'] >= .25 - 1e-6 for a, b in zip(last, last[1:])),
                'RGB reads are too close in time')
        volumes = [x['volume_ml'] for x in last]
        require(max(volumes) - min(volumes) <= .4
                and abs(volumes[-1] - case['volume']) <= .5, 'RGB measurement gate failed')


def reserve_independent(ledger, case_set_sha, model_sha, output, *, suite_file_sha=None):
    """Reserve before launching so interruption cannot silently reuse test data."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        require(not any(row.get('case_set_sha256', row.get('suite_sha256')) == case_set_sha for row in rows),
                'Independent suite was already reserved/used; audit that run or define a fresh final suite before tuning')
        stream.write(json.dumps(dict(case_set_sha256=case_set_sha, suite_sha256=suite_file_sha,
                     model_sha256=model_sha,
                     output=str(output), reserved_at=time.time(), status='reserved_before_neural_launch')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True, help='Integrated driver candidate-ready.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='Check the genuine candidate without any suite run/ledger mutation')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    candidate_path = args.candidate.resolve()
    candidate = json.loads(candidate_path.read_text())
    run = candidate_path.parent
    model = Path(candidate['model']).resolve()
    config = run / 'train.yml'
    scripts = Path(__file__).resolve().parent
    root = scripts.parents[1]
    controller_path = scripts / 'configs/cylinder_official_teacher_dev_v1.json'
    controller = json.loads(controller_path.read_text())
    single = json.loads(Path(candidate['single_summary']).read_text())
    action_space = single['action_space']
    require(candidate.get('action_space', 'joint') == action_space, 'Candidate action space changed')
    require(candidate.get('goal_complete') is False, 'Expected a provisional candidate')
    require(candidate['requires'] == ['original8', 'development12', 'untouched_independent12'], 'Acceptance scope mismatch')
    audit = json.loads((run / f"checkpoint-{candidate['step']}-weight-audit.json").read_text())
    require(audit['passed'] and audit['checkpoint'] == str(model), 'Successful weight audit required')
    hashes = single['episodes'][0]['source_sha256']
    require(all(digest(scripts / name) == sha for name, sha in hashes.items()), 'Candidate execution source is stale')
    verify_report(single, [dict(volume=25., seed=100)], model, config, controller, hashes, action_space=action_space)
    from audit_wall_checkpoint import saved_checkpoint_position

    require(model.parent == run / 'checkpoints', 'Unexpected checkpoint location')
    _, saved_step = saved_checkpoint_position(model)
    require(saved_step == candidate['step'], 'Candidate step differs from saved checkpoint metadata')
    require((model / 'model.safetensors').is_file(), 'Candidate weights missing')
    require(not args.output.exists(), 'Output already exists; inspect it rather than silently rerunning suites')
    require(audit['checkpoint_sha256'] == digest(model / 'model.safetensors'), 'Audited weights changed')
    if args.validate_only:
        print(json.dumps(dict(candidate_valid=True, neural_suites_started=False, independent_reserved=False)))
        return
    processes = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
    require(not any(str(root) in line and 'python' in line
                    and any(marker in line for marker in ('train_fsdp.py', 'run_wall_cylinder.py'))
                    for line in processes),
            'Project neural training/evaluation is still running; inspect/wait before full acceptance')
    acceptance_lock = (root / 'runs/cylinder-neural-acceptance.lock').open('a')
    fcntl.flock(acceptance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    suites = [('original8', 8, 'cylinder_policy_original_cases.json'),
              ('development12', 12, 'cylinder_policy_dev_cases.json'),
              ('untouched_independent12', 12, 'cylinder_policy_independent_cases.json')]
    protocol = [(name, count, scripts / 'configs' / file) for name, count, file in suites]
    suite_hashes = {name: digest(path) for name, _, path in protocol}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(model=str(model), model_sha256=audit['checkpoint_sha256'], suite_sha256=suite_hashes,
                 action_space=action_space,
                 source_sha256=hashes, acceptance_source_sha256=digest(Path(__file__)),
                 config_sha256=digest(config), controller_sha256=digest(controller_path),
                 status='running', completed_suites=[], goal_complete=False)
    def save():
        (output / 'acceptance-status.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    env = os.environ | dict(PYTHONPATH=str(root / 'wall-x'), MUJOCO_GL='egl',
                            OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    try:
        for name, count, case_path in protocol:
            require(digest(case_path) == suite_hashes[name], 'Suite changed during acceptance')
            require(digest(config) == state['config_sha256']
                    and digest(controller_path) == state['controller_sha256'],
                    'Training/controller configuration changed during acceptance')
            cases = json.loads(case_path.read_text())
            require(len(cases) == count, 'Suite size changed')
            if name == 'untouched_independent12':
                state['independent_case_set_sha256'] = case_set_digest(cases)
                reserve_independent(root / 'runs/cylinder-policy-independent-suite-ledger.jsonl',
                                    state['independent_case_set_sha256'], state['model_sha256'], output / name,
                                    suite_file_sha=suite_hashes[name])
            state['active_suite'] = name
            save()
            norm_key = 'cylinder_joint_v1' if action_space == 'joint' else 'cylinder_cartesian_v1'
            argv = [sys.executable, str(scripts / 'run_wall_cylinder.py'), '--model', str(model),
                    '--config', str(config), '--norm-key', norm_key,
                    '--controller-config', str(controller_path), '--cases', str(case_path),
                    '--execute-steps', '8', '--max-seconds', '90', '--policy-batch-size', '1',
                    '--output', str(output / name)]
            with (output / f'{name}.log').open('w') as log:
                subprocess.run(argv, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((output / name / 'summary.json').read_text())
            verify_report(report, cases, model, config, controller, hashes, action_space=action_space)
            state['completed_suites'].append(dict(name=name, successes=count, total=count,
                summary=str(output / name / 'summary.json'), summary_sha256=digest(output / name / 'summary.json')))
            save()
        require(digest(model / 'model.safetensors') == state['model_sha256'], 'Weights changed during acceptance')
        state.update(status='all_32_neural_cases_passed_requires_final_audit', finished=time.time())
        save()
    except BaseException as error:
        state.update(status='stopped_without_full_acceptance', error=f'{type(error).__name__}: {error}')
        save()
        raise
    finally:
        acceptance_lock.close()


if __name__ == '__main__':
    main()
