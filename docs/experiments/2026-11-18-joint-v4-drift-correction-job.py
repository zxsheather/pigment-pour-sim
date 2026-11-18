"""Collect and check teacher corrections from V4-derived training states."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/data/hlou/pigment-pour-study')
SIM = ROOT / 'pigment-pour-sim/simulation'
PREFIX = ROOT / 'runs/cylinder-joint-v4-drift-prefix-v1'
RAW = SIM / 'generated/cylinder-joint-v4-drift-recovery-v1'
RESULT = ROOT / 'runs/cylinder-joint-v4-drift-recovery-v1-validation'
TRAIN = ROOT / 'runs/cylinder-joint-reopen-bc-v4/train.yml'
PLANS = [('volume_15_seed_9021', (1,)), ('volume_25_seed_9023', (4,)),
         ('volume_35_seed_9025', (6, 5))]
ENV = os.environ | dict(MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
STATE = dict(pid=os.getpid(), stage='collecting', attempts=[])


def read(path):
    return json.loads(path.read_text())


def save():
    STATE['updated_at'] = time.time()
    temp = RESULT / 'status.tmp'
    temp.write_text(json.dumps(STATE, indent=2) + '\n')
    temp.replace(RESULT / 'status.json')


def call(venv, script, arguments, log):
    argv = [str(ROOT / venv / 'bin/python'), str(SIM / script), *map(str, arguments)]
    with log.open('x') as output:
        output.write(json.dumps(argv) + '\n')
        output.flush()
        return subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=300,
                              cwd=ROOT, env=ENV).returncode


def configure():
    global PREFIX, RAW, RESULT, PLANS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=PREFIX)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--result', type=Path, default=RESULT)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    PREFIX, RAW, RESULT = (path.resolve() for path in (args.prefix, args.raw, args.result))
    if args.plan:
        PLANS = [(row['case'], row['snapshots']) for row in read(args.plan)]
    assert PLANS and len({case for case, _ in PLANS}) == len(PLANS)
    for case, snapshots in PLANS:
        assert Path(case).name == case and snapshots and min(snapshots) >= 0
        assert len(set(snapshots)) == len(snapshots)
    STATE.update(prefix=str(PREFIX), raw_directory=str(RAW))


def main():
    RAW.mkdir()
    RESULT.mkdir()
    for case, snapshots in PLANS:
        source = PREFIX / case / 'episode_000'
        summary = read(source / 'summary.json')
        assert summary['geometry_audit']['passed']
        for index in snapshots:
            snap = summary['snapshots'][index]
            assert snap['loaded_contacts'] == 0
            tag = f'{case}_snapshot_{index:02d}'
            dest = RAW / tag
            logs = RESULT / tag
            logs.mkdir()
            row = dict(case=case, source_snapshot=snap, destination=str(dest))
            STATE['attempts'].append(row)
            save()
            row['teaching_exit'] = call('.venv-sim', 'recover_cylinder_demonstration.py', [
                '--policy-episode', source, '--snapshot-index', index, '--controller-config',
                SIM / 'configs/cylinder_official_teacher_dev_v1.json', '--output', dest], logs / 'teach.log')
            if row['teaching_exit']:
                summary_path = dest / 'summary.json'
                if summary_path.exists():
                    row['teacher_failure_reason'] = read(summary_path)['failure_reason']
                save()
                continue
            reconstruct = logs / 'joint-reconstruction'
            row['reconstruction_exit'] = call('.venv-sim', 'reconstruct_cylinder_joints.py',
                ['--episode', dest, '--output', reconstruct], logs / 'reconstruct.log')
            if row['reconstruction_exit']:
                save()
                continue
            validation = logs / 'validation.json'
            row['validation_exit'] = call('.venv-wall', 'validate_raw_cylinder_recovery.py',
                ['--episode', dest, '--config', TRAIN, '--joint-reconstruction', reconstruct,
                 '--output', validation], logs / 'validate.log')
            if not row['validation_exit']:
                verified = read(validation)
                row.update(frames=verified['frames'], validation=str(validation),
                           normalization_covers_all_labels=verified['normalization_covers_all_labels'],
                           data_integrity_passed=verified['data_integrity_passed'],
                           max_slip_m=verified['teacher_result']['max_slip_m'])
                save()
                if row['data_integrity_passed'] and row['normalization_covers_all_labels']:
                    break
    STATE.update(stage='complete', learned_policy_task_success_measured=False,
                 active_training_dataset_modified=False, independent_cases_read=False)
    save()


if __name__ == '__main__':
    configure()
    if RAW.exists() or RESULT.exists():
        raise FileExistsError('Recovery job requires new output directories')
    try:
        main()
    except BaseException as exc:
        if RESULT.exists():
            STATE.update(stage='failed', error=f'{type(exc).__name__}: {exc}')
            save()
        raise
