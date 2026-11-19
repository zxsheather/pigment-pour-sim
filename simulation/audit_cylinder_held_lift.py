"""Replay recorded commands and measure lift while both fingers carry load."""
import argparse
import json
from pathlib import Path

import numpy as np

from carry_env import CONTROL_DT
from cylinder_joint_actions import require, robot_joint_observation
from run_wall_cylinder import CylinderTrial


def audit(rollout, output):
    recorded = json.loads((rollout / 'summary.json').read_text())
    joint = recorded['action_space'] == 'joint'
    backoff = recorded['action_space'] == 'cartesian' and recorded.get('cartesian_ik_backoff', False)
    require(recorded['evaluation_complete'] and (joint or backoff)
            and recorded['total'] == 1 and not recorded['probe_only'],
            'Expected a complete single joint or diagnostic Cartesian rollout')
    case = recorded['episodes'][0]
    trace = rollout / 'episode_000/trajectory.npz'
    duration = case['simulated_seconds']
    max_seconds = duration if case['failure_reason'] == 'timeout' else duration + 1.
    trial = CylinderTrial(case['case'], recorded['controller'], output,
                          action_space=recorded['action_space'], max_seconds=max_seconds,
                          cartesian_ik_backoff=backoff, ik_max_residual=recorded['ik_max_residual'] if backoff else .002)
    max_supported_lift = 0.
    raw_peak = dict(height_m=0., time_s=0., loaded_fingers=0, established_hold=False)
    contact_counts = [0, 0, 0]
    double_frames = longest_double = 0
    first_hold = first_supported_40mm = None
    closest = (float('inf'), 0.)
    try:
        require(trial.geometry_report['passed'] and trial.geometry_report['custom_pads'] == 0,
                'Expected official collision geometry')
        with np.load(trace, allow_pickle=False) as data:
            state_key = 'robot_joint_state' if joint else 'state'
            backoff_trace = {key: data[key] for key in ('ik_residual', 'requested_ik_residual', 'pose_fraction')} if backoff else {}
            increments = zip(data['action'], data[state_key], data['joint_command'], strict=True)
            for index, (action, state, bounded) in enumerate(increments):
                require(not trial.done, 'Replay stopped before the recorded final command')
                observed = robot_joint_observation(trial.env) if joint else trial.env.state_vector()
                np.testing.assert_array_equal(observed, state)
                trial.step(action)
                np.testing.assert_array_equal(trial.rows[-1]['joint_command'], bounded)
                if backoff:
                    require(all(trial.rows[-1][key] == data_column[index] for key, data_column in backoff_trace.items()),
                            'Recorded IK projection did not reproduce')
                metrics = trial.env.metrics()
                fingers = metrics['grip_contacts']
                contact_counts[fingers] += 1
                double_frames = double_frames + 1 if fingers == 2 else 0
                longest_double = max(longest_double, double_frames)
                time = float(trial.env.data.time)
                if backoff:
                    distance = float(np.linalg.norm(trial.env.grip_point_relative() - trial.env.grasp_offset))
                    closest = min(closest, (distance, time))
                if trial.held and first_hold is None:
                    first_hold = time
                elevation = float(trial.env.data.xpos[trial.env.cylinder_body, 2] - trial.initial_cylinder[2])
                if elevation > raw_peak['height_m']:
                    raw_peak = dict(height_m=elevation, time_s=time,
                                    loaded_fingers=fingers, established_hold=trial.held,
                                    tilt_rad=metrics['bottle_tilt_rad'])
                if trial.held and fingers == 2 and not metrics['forbidden_contacts']:
                    max_supported_lift = max(max_supported_lift, elevation)
                    if elevation >= .04 and first_supported_40mm is None:
                        first_supported_40mm = time
        require(trial.reason == case['failure_reason'] and trial.success == case['success'],
                'Terminal outcome did not reproduce')
        require(abs(raw_peak['height_m'] - case['max_lifted_m']) < 1e-9,
                'Recorded height peak did not reproduce')
        result = dict(rollout=str(rollout.resolve()), terminal_reason=trial.reason, raw_peak=raw_peak,
                      max_supported_lift_m=max_supported_lift, established_hold=trial.held,
                      loaded_finger_counts_after_command=contact_counts,
                      longest_consecutive_double_load_frames=longest_double,
                      longest_consecutive_double_load_seconds=longest_double * CONTROL_DT,
                      first_sustained_hold_s=first_hold, first_supported_lift_40mm_s=first_supported_40mm,
                      observations_and_bounded_commands_exact=True, checkpoint_inference_rerun=False)
        if backoff:
            result.update(ik_backoff_trace_exact=True,
                          nearest_tool=dict(distance_m=closest[0], time_s=closest[1]) if sum(contact_counts) else None)
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
