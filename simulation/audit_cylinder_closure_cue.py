"""Measure how well reset RGB and wrist kinematics locate the cylinder at closure."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow

from cylinder_joint_actions import APERTURE_SCALE_M, require, with_joint_servo_state
from cylinder_progress_prompts import CylinderProgressPrompts, PROGRESS_PROMPTS, PROGRESS_PROTOCOL
from cylinder_tray_vision import (
    WRIST_TARGET_CUE_INDICES, WRIST_TARGET_CUE_PROTOCOL, WRIST_TARGET_NEAR_SCALE_M,
    tray_grasp_height_m, with_wrist_target_state,
)


# Lateral, axial and vertical tolerances in millimetres. This grid is a
# sensitivity analysis, not a set of allowed commands or task score gates.
GATES_MM = ((5., 20., 10.), (10., 20., 10.), (20., 20., 10.),
            (5., 30., 20.), (10., 30., 20.), (20., 30., 20.))
DWELL_GATE_MM = (5., 20., 10.)
DWELL_CANDIDATES = ((1., 5), (2.5, 5), (5., 5), (1., 10), (2.5, 10), (5., 10))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrist_error_mm(states, robot):
    """Project the causal RGB offset into the wrist's lateral and axial axes."""
    cue = states[:, WRIST_TARGET_CUE_INDICES]
    require(set(np.unique(cue[:, -1])) <= {0., 1.}, 'Invalid RGB availability flag')
    body_error = 1000 * WRIST_TARGET_NEAR_SCALE_M * np.sinh(cue[:, :3])
    first, second = robot[:, 13:16], robot[:, 16:19]
    rotation = np.stack((first, second, np.cross(first, second)), axis=-2)
    tool_error = np.einsum('nij,ni->nj', rotation, body_error)
    error = np.column_stack((tool_error[:, 1], tool_error[:, 0], body_error[:, 2]))
    error[cue[:, -1] == 0.] = np.nan
    return error


def wrist_speed_mm_s(states, robot, grasp_x_m, times):
    """Recover tool-tip speed using consecutive measured poses at 20 Hz."""
    first, second = robot[:, 13:16], robot[:, 16:19]
    shaft = np.stack((first[:, 0], second[:, 0], np.cross(first, second)[:, 0]), axis=-1)
    tip = robot[:, 10:13] + grasp_x_m * shaft
    heading = states[:, 17].astype(np.float64)
    c, s = np.cos(heading), np.sin(heading)
    x, y, z = tip.T
    world = np.column_stack((states[:, 15] + c * x - s * y, states[:, 16] + s * x + c * y, z))
    step = np.diff(times)
    require(np.allclose(step, .05, rtol=0., atol=1e-7), 'Expected consecutive 20 Hz source poses')
    return np.r_[np.inf, 1000 * np.linalg.norm(np.diff(world, axis=0), axis=1) / step]


def gate_events(ready, close, narrow, actual):
    eligible = np.flatnonzero(ready[:narrow+1])
    hit = int(eligible[0]) if len(eligible) else None
    return dict(
        ready_at_close=bool(ready[close]), ready_at_first_narrow=bool(ready[narrow]),
        first_gate_frames_relative_to_close=None if hit is None else hit-close,
        first_gate_true_tool_error_mm=None if hit is None else actual[hit, [1, 0, 2]].tolist())


def dwell_mask(spatially_ready, speed, limit, ticks):
    qualifying = spatially_ready & (speed <= limit)
    # Count only the current tick and already observed ticks.
    consecutive = np.convolve(qualifying.astype(int), np.ones(ticks, dtype=int))[:len(speed)]
    return consecutive == ticks


def summarize(source, result, state, action, task):
    """Use source phase and grip geometry only as retrospective labels."""
    with np.load(source/'trajectory.npz', allow_pickle=False) as raw:
        n = len(raw['phase'])
        require(n == len(state) == len(action) == len(task), 'Frame count differs from the teacher')
        np.testing.assert_array_equal(raw['base_pose'].astype(np.float32), state[:, 15:18])
        np.testing.assert_array_equal((2 * raw['joint_command'][:, 6] / APERTURE_SCALE_M).astype(np.float32),
                                      action[:, 6])
        closes = np.flatnonzero(raw['phase'] == 'close')
        if not len(closes):
            return None
        close = int(closes[0])
        narrower = closes[action[closes, 6] * APERTURE_SCALE_M <= .020]
        if not len(narrower):
            return None
        narrow = int(narrower[0])
        measured_gap = state[:, 6] * APERTURE_SCALE_M
        opened = np.flatnonzero((measured_gap >= .040) & (np.arange(n) < close))
        if not len(opened):
            return None
        prefix = np.arange(n) >= opened[0]
        robot = raw['state']
        grasp_x_m = result['controller_parameters']['grasp_x_m']
        observed = wrist_error_mm(state, robot)
        actual = 1000 * (raw['grip_point_relative'] - [grasp_x_m, 0., 0.])
        speed = wrist_speed_mm_s(state, robot, grasp_x_m, raw['timestamp'])
        require(np.isfinite(actual).all(), 'Invalid true grip coordinates')
        rows = {}
        for tag, i in (('close_start', close), ('first_narrow_target', narrow)):
            valid = np.isfinite(observed[i]).all()
            rows[tag] = dict(frame=i, observed_error_mm=observed[i].tolist() if valid else None,
                             measured_tip_speed_mm_s=float(speed[i]),
                             true_tool_error_mm=actual[i, [1, 0, 2]].tolist())
        approach = prefix & (task == 0)
        gates = [gate_events(np.all(np.abs(observed) <= limits, axis=1) & approach, close, narrow, actual)
                 for limits in GATES_MM]
        spatially_ready = np.all(np.abs(observed) <= DWELL_GATE_MM, axis=1) & approach
        dwell = [gate_events(dwell_mask(spatially_ready, speed, limit, ticks), close, narrow, actual)
                 for limit, ticks in DWELL_CANDIDATES]
        return dict(source=str(source), seed=result['seed'], events=rows, gates=gates, dwell_gates=dwell)


def gate_counts(rows, key, index):
    entries = [row[key][index] for row in rows]
    start = [e['first_gate_frames_relative_to_close'] for e in entries
             if e['first_gate_frames_relative_to_close'] is not None]
    return dict(ready_at_close=sum(e['ready_at_close'] for e in entries),
                ready_at_first_narrow=sum(e['ready_at_first_narrow'] for e in entries),
                opens_at_least_8_frames_early=sum(t <= -8 for t in start),
                first_gate_relative_frame_quantiles=np.percentile(start, (0, 50, 100)).tolist() if start else [],
                never_opens_by_first_narrow=len(entries)-len(start))


def summarize_groups(rows, total):
    events = {}
    for tag in ('close_start', 'first_narrow_target'):
        valid = [r['events'][tag] for r in rows if r['events'][tag]['observed_error_mm'] is not None]
        observed = np.asarray([r['observed_error_mm'] for r in valid]).reshape(-1, 3)
        actual = np.asarray([r['true_tool_error_mm'] for r in valid]).reshape(-1, 3)
        events[tag] = dict(
            cue_episode_count=len(valid),
            absolute_observed_error_mm_quantiles=np.percentile(np.abs(observed), (50, 90, 100), axis=0).tolist()
            if valid else [],
            absolute_true_tool_error_mm_quantiles=np.percentile(np.abs(actual), (50, 90, 100), axis=0).tolist()
            if valid else [],
            absolute_lateral_estimation_error_mm_quantiles=np.percentile(
                np.abs(observed[:, 0] - actual[:, 0]), (50, 90, 100)).tolist() if valid else [])
    gates = [dict(tolerances_mm=limit, **gate_counts(rows, 'gates', i)) for i, limit in enumerate(GATES_MM)]
    dwell = [dict(tolerances_mm=DWELL_GATE_MM, max_tool_speed_mm_s=speed, elapsed_ticks=ticks,
                  **gate_counts(rows, 'dwell_gates', i)) for i, (speed, ticks) in enumerate(DWELL_CANDIDATES)]
    return dict(total_episodes=total, open_then_close_episodes=len(rows), events=events, gates=gates,
                dwell_gates=dwell)


def audit_rollout(folder):
    from inspect_cylinder import CylinderEnv

    summary = json.loads((folder/'summary.json').read_text())
    episode, = summary['episodes']
    require(summary['evaluation_complete'] and episode['action_space'] == 'joint'
            and episode['progress_protocol'] == PROGRESS_PROTOCOL
            and episode['tray_cue_protocol'] == WRIST_TARGET_CUE_PROTOCOL
            and episode['geometry_audit']['passed'] and episode['geometry_audit']['custom_pads'] == 0,
            'Expected a complete official-geometry rollout with the original RGB cue')
    case = episode['case']
    env = CylinderEnv(case['volume'], mass_scale=case.get('mass_scale', 1.),
                      friction_scale=case.get('friction_scale', 1.), **summary['controller']['environment'])
    try:
        env.reset_inspection(seed=case['seed'], jitter=case.get('jitter', .004),
                             yaw_jitter=case.get('yaw_jitter', 0.))
        target_height = tray_grasp_height_m(env.config, env.grasp_height) - env.data.xpos[env.robot_root, 2]
        router = CylinderProgressPrompts(env.base_pose())
        grasp_x_m = env.grasp_offset[0]
    finally:
        env.close()

    folder = folder/'episode_000'
    trajectory = folder/'trajectory.npz'
    observations = folder/'model_observations.npz'
    with (np.load(trajectory, allow_pickle=False) as raw, np.load(observations, allow_pickle=False) as samples):
        times = raw['timestamp']
        robot = raw['state']
        commands = raw['joint_command']
        previous = np.vstack((episode['progress_initial_joint_command'], commands[:-1]))
        servo = with_joint_servo_state(raw['robot_joint_state'], previous)
        tasks = np.array([router.observe(row) for row in servo])
        state = with_wrist_target_state(servo, robot, episode['initial_tray_cue'],
                                        grasp_x_m=grasp_x_m, grasp_target_height_m=target_height)
        state[np.ix_(tasks != 0, WRIST_TARGET_CUE_INDICES)] = 0.
        indices = np.searchsorted(times, samples['timestamp'])
        np.testing.assert_array_equal(times[indices], samples['timestamp'])
        np.testing.assert_array_equal(np.asarray(PROGRESS_PROMPTS)[tasks[indices]], samples['instruction'])
        np.testing.assert_array_equal(state[indices], samples['state'])
        observed = wrist_error_mm(state, robot)
        speed = wrist_speed_mm_s(state, robot, grasp_x_m, times)
        opened = np.flatnonzero(state[:, 6] * APERTURE_SCALE_M >= .040)
        after_open = np.arange(len(state)) >= opened[0] if len(opened) else np.zeros(len(state), dtype=bool)
        approach = after_open & (tasks == 0)
        spatial = np.all(np.abs(observed) <= DWELL_GATE_MM, axis=1) & approach
        dwell = dwell_mask(spatial, speed, 2.5, 10)
        requested = raw['unfiltered_model_action'][:, 6] * APERTURE_SCALE_M
        next_opening = 2 * commands[:, 6]

        def event(mask):
            candidates = np.flatnonzero(mask)
            if not len(candidates):
                return None
            i = candidates[0]
            return dict(time_s=float(times[i]), observed_error_mm=observed[i].tolist()
                        if np.isfinite(observed[i]).all() else None, dwell_gate_open=bool(dwell[i]))

        return dict(rollout=str(folder.parent), failure_reason=episode['failure_reason'], case=case,
                    trajectory_sha256=sha256(trajectory), observation_sha256=sha256(observations),
                    reconstructed_observations=len(indices), observed_state_and_prompt_exact=True,
                    first_spatially_ready=event(spatial), first_ready_after_dwell=event(dwell),
                    first_narrow_request_after_open=event(after_open & (requested <= .020)),
                    first_applied_narrow_after_open=event(after_open & (next_opening <= .020)))


def audit(dataset, reserve_path):
    provenance_path = dataset/'demonstration_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    table = arrow.dataset(dataset/'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'frame_index', 'task_index', 'observation.state', 'action']
    ).sort_by('index')
    np.testing.assert_array_equal(np.asarray(table['index']), np.arange(len(table)))
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    tasks = np.asarray(table['task_index'])
    episodes = np.asarray(table['episode_index'])
    frames = np.asarray(table['frame_index'])
    np.testing.assert_array_equal(np.unique(episodes), np.arange(len(provenance)))
    rows = []
    training_seeds = set()
    for i, entry in enumerate(provenance):
        source = Path(entry['source'])
        result = entry['metadata']['result']
        training_seeds.add(result['seed'])
        require(result['success'] and result['grip_profile'] == 'official' and result['geometry_audit_passed']
                and sha256(source/'trajectory.npz') == entry['trajectory_sha256'], 'Uncertified teacher')
        chosen = episodes == i
        np.testing.assert_array_equal(frames[chosen], np.arange(entry['metadata']['frames']))
        row = summarize(source, result, states[chosen], actions[chosen], tasks[chosen])
        if row is not None:
            rows.append(row)

    reserve = json.loads(reserve_path.read_text())
    require(reserve['used_for_any_training'] is False and reserve['independent_cases_read'] is False,
            'Expected the reserved teacher diagnostics')
    reserved = []
    for entry in reserve['results']:
        source = Path(entry['source'])
        result = json.loads((source/'summary.json').read_text())
        require(entry['qualified'] and result['seed'] == entry['seed'] and result['seed'] not in training_seeds,
                'Expected a reserved seed excluded from this dataset')
        with np.load(entry['training_arrays'], allow_pickle=False) as recorded:
            row = summarize(source, result, recorded['state'], recorded['action'], recorded['task_index'])
        require(row is not None, 'Missing reserved closing transition')
        reserved.append(row)
    return dict(kind='observable_reset_rgb_closure_cue_diagnostic',
                input_sha256=dict(dataset=sha256(provenance_path), reserve=sha256(reserve_path)),
                observed_coordinates=['tool_lateral', 'tool_axial', 'vertical'],
                true_coordinates=['tool_lateral', 'tool_axial', 'tool_vertical'],
                error_quantiles=[50, 90, 100], timing_quantiles=[0, 50, 100], independent_cases_read=False,
                new_training_or_formal_evaluation=False,
                training_teachers=summarize_groups(rows, len(provenance)),
                reserved_teachers=summarize_groups(reserved, len(reserve['results'])),
                training_rows=rows, reserved_rows=reserved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reserve', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, action='append', default=[])
    args = parser.parse_args()
    report = audit(args.dataset, args.reserve)
    report['model_rollouts'] = [audit_rollout(path) for path in args.rollout]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('training_teachers', 'reserved_teachers')}, indent=2))


if __name__ == '__main__':
    main()
