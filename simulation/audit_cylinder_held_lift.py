"""Replay a saved joint rollout and measure lift while both fingers carry load."""
import argparse
import json
from pathlib import Path

import numpy as np

from cylinder_joint_actions import require, robot_joint_observation
from run_wall_cylinder import CylinderTrial


def audit(rollout, output):
    recorded = json.loads((rollout / 'summary.json').read_text())
    require(recorded['evaluation_complete'] and recorded['action_space'] == 'joint'
            and recorded['total'] == 1 and not recorded['probe_only'],
            'Expected a complete single joint rollout')
    case = recorded['episodes'][0]
    trace = rollout / 'episode_000/trajectory.npz'
    duration = case['simulated_seconds']
    max_seconds = duration if case['failure_reason'] == 'timeout' else duration + 1.
    trial = CylinderTrial(case['case'], recorded['controller'], output,
                          action_space='joint', max_seconds=max_seconds)
    max_supported_lift = 0.
    raw_peak = dict(height_m=0., time_s=0., loaded_fingers=0, established_hold=False)
    contact_counts = [0, 0, 0]
    try:
        require(trial.geometry_report['passed'] and trial.geometry_report['custom_pads'] == 0,
                'Expected official collision geometry')
        with np.load(trace, allow_pickle=False) as data:
            for action, state, bounded in zip(data['action'], data['robot_joint_state'],
                                               data['joint_command'], strict=True):
                require(not trial.done, 'Replay stopped before the recorded final command')
                np.testing.assert_array_equal(robot_joint_observation(trial.env), state)
                trial.step(action)
                np.testing.assert_array_equal(trial.rows[-1]['joint_command'], bounded)
                metrics = trial.env.metrics()
                fingers = metrics['grip_contacts']
                contact_counts[fingers] += 1
                elevation = float(trial.env.data.xpos[trial.env.cylinder_body, 2] - trial.initial_cylinder[2])
                if elevation > raw_peak['height_m']:
                    raw_peak = dict(height_m=elevation, time_s=float(trial.env.data.time),
                                    loaded_fingers=fingers, established_hold=trial.held,
                                    tilt_rad=metrics['bottle_tilt_rad'])
                if trial.held and fingers == 2 and not metrics['forbidden_contacts']:
                    max_supported_lift = max(max_supported_lift, elevation)
        require(trial.reason == case['failure_reason'] and trial.success == case['success'],
                'Terminal outcome did not reproduce')
        require(abs(raw_peak['height_m'] - case['max_lifted_m']) < 1e-9,
                'Recorded height peak did not reproduce')
        result = dict(rollout=str(rollout.resolve()), terminal_reason=trial.reason, raw_peak=raw_peak,
                      max_supported_lift_m=max_supported_lift, established_hold=trial.held,
                      loaded_finger_counts_after_command=contact_counts,
                      observations_and_bounded_commands_exact=True, checkpoint_inference_rerun=False)
        (output / 'held-lift-audit.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        trial.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.rollout, args.output)))


if __name__ == '__main__':
    main()
