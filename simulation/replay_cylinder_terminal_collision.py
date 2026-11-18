"""Reproduce a joint rollout's last collision and identify both contact bodies."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from cylinder_joint_actions import require, robot_joint_observation
from inspect_cylinder import CylinderEnv


def replay(rollout):
    summary = json.loads((rollout / 'summary.json').read_text())
    require(summary['evaluation_complete'] and summary['action_space'] == 'joint' and summary['total'] == 1,
            'Expected a complete single joint-policy rollout')
    result = summary['episodes'][0]
    require(result['geometry_audit']['passed'] and result['geometry_audit']['custom_pads'] == 0,
            'Expected official finger geometry')
    require((result['failure_reason'] or '').startswith('collision:'), 'Expected a terminal collision')
    case = result['case']
    episode = rollout / 'episode_000'
    env = CylinderEnv(case['volume'], **summary['controller']['environment'],
                      friction_scale=case.get('friction_scale', 1.),
                      mass_scale=case.get('mass_scale', 1.))
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0.))
        with np.load(episode / 'recovery_physics.npz', allow_pickle=False) as states:
            snapshot = len(states['full_state']) - 1
            spec = int(states['state_spec'])
            mujoco.mj_setState(env.model, env.data, states['full_state'][snapshot], spec)
            actuator_ids = states['hold_actuator_ids'].tolist()
            require(set(actuator_ids) == set(env.hold_targets), 'Servo cache differs')
            env.hold_targets = dict(zip(actuator_ids, states['hold_targets'][snapshot].tolist()))
            env.applied_action[:] = states['applied_base_action'][snapshot]
        mujoco.mj_forward(env.model, env.data)
        source_time = float(env.data.time)
        with np.load(episode / 'trajectory.npz', allow_pickle=False) as trace:
            times = trace['timestamp']
            first = int(np.argmin(abs(times - source_time)))
            require(abs(times[first] - source_time) < 1e-8, 'Snapshot is out of alignment')
            commands = trace['joint_command'][first:]
            observed = trace['robot_joint_state'][first:]
        for command, state in zip(commands, observed, strict=True):
            np.testing.assert_array_equal(robot_joint_observation(env), state)
            _, metrics = env.step_joints(command[:6], opening=float(command[6]),
                                         forward=float(command[7]), yaw=float(command[8]))
        require(abs(env.data.time - result['simulated_seconds']) < 1e-8, 'Terminal time differs')
        require('collision:' + str(metrics['forbidden_contacts']) == result['failure_reason'],
                'Recorded collision did not reproduce')
        contacts = []
        for contact in env.data.contact[:env.data.ncon]:
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(env.model.geom_bodyid[g]) for g in geoms]
            names = [env.model.geom(g).name for g in geoms]
            if contact.dist < -.001 and (names in metrics['forbidden_contacts']
                                         or names[::-1] in metrics['forbidden_contacts']):
                contacts.append(dict(geom_ids=geoms, geom_names=names,
                    body_names=[env.model.body(b).name for b in bodies], penetration_m=-float(contact.dist)))
        require(bool(contacts), 'No corresponding forbidden contact')
        return dict(kind='recorded_joint_terminal_collision_replay', rollout=str(rollout.resolve()),
                    snapshot_index=snapshot, source_time_s=source_time, replayed_frames=len(commands),
                    final_time_s=float(env.data.time), joint_observations_exact=True,
                    recorded_failure_reason=result['failure_reason'], penetrating_contacts=contacts,
                    physics_timestep_s=float(env.model.opt.timestep), solver_impratio=float(env.model.opt.impratio),
                    neural_weights_loaded=False, independent_cases_read=False,
                    learned_policy_task_success_measured=False)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists')
    report = replay(args.rollout)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
