"""Audit training demonstrations through the joint policy's physical evaluator."""
import argparse
import hashlib
import json
from pathlib import Path

from validate_cylinder_joint_execution import joint_replay


def write_summary(output, report):
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(output / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    provenance_file = dataset / 'demonstration_provenance.json'
    provenance_bytes = provenance_file.read_bytes()
    provenance = json.loads(provenance_bytes)
    info = json.loads((dataset / 'meta/info.json').read_text())
    if not provenance or len(provenance) != info['total_episodes']:
        raise ValueError('Dataset episode metadata does not match demonstration provenance')
    if any(not item['metadata']['result']['success'] for item in provenance):
        raise ValueError('Audit expects demonstrations that passed the source teacher')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='recorded_teacher_joint_decoder_dataset_audit', dataset=str(dataset),
                  total=len(provenance), rows=[], status='running', source_provenance_sha256=
                  hashlib.sha256(provenance_bytes).hexdigest(), neural_weights_loaded=False,
                  learned_policy_task_success_measured=False, independent_cases_read=False)
    write_summary(output, report)
    for index, demo in enumerate(provenance):
        target = output / f'episode-{index:03d}'
        result = demo['metadata']['result']
        row = dict(episode=index, source=demo['source'], seed=result['seed'],
                   volume_ml=result['expected_volume_ml'],
                   recovery=bool(result.get('initial_state_provenance')))
        error = None
        try:
            replay = joint_replay(dataset, index, target)
        except ValueError as exc:
            replay_file = target / 'decoder-replay.json'
            if not replay_file.is_file():
                # An infrastructure/data failure is distinct from a physical loss.
                report.update(status='error', error=str(exc), error_episode=index)
                write_summary(output, report)
                raise
            error = str(exc)
            replay = json.loads(replay_file.read_text())
        row.update(passed=error is None and replay['geometry_audit_passed'], error=error,
                   geometry_audit_passed=replay['geometry_audit_passed'],
                   failure_reason=replay['failure_reason'], final_gates=replay['final_task_gates'],
                   max_slip_m=replay['max_slip_m'], max_lift_m=replay['max_lifted_m'],
                   joint_limit_projected_frames=replay['joint_limit_projected_frames'],
                   relative_decode_max_difference=replay['relative_decode_max_difference'],
                   report=str(target / 'decoder-replay.json'))
        report['rows'].append(row)
        write_summary(output, report)
        print(json.dumps(row), flush=True)
    if provenance_file.read_bytes() != provenance_bytes:
        raise ValueError('Demonstration provenance changed during the audit')
    report.update(status='complete', passed=sum(row['passed'] for row in report['rows']))
    write_summary(output, report)
    print(json.dumps(dict(output=str(output), passed=report['passed'], total=report['total'])), flush=True)


if __name__ == '__main__':
    main()
