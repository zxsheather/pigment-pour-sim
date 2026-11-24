"""Compare pre-grasp rollout states and commands with the demonstration envelope."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow
import yaml

from cylinder_joint_actions import (
    APERTURE_SCALE_M, JOINT_ACTION_WIDTH, JOINT_STATE_NAMES, JOINT_STATE_WIDTH, require,
)


def pregrasp_demo_rows(dataset):
    table = arrow.dataset(dataset / 'data', format='parquet').to_table(columns=[
        'index', 'episode_index', 'frame_index', 'timestamp', 'observation.state', 'action'])
    table = table.sort_by('index')
    require(np.array_equal(np.asarray(table['index']), np.arange(len(table))),
            'Expected contiguous training data indices')
    episodes = np.asarray(table['episode_index'])
    frames = np.asarray(table['frame_index'])
    timestamps = np.asarray(table['timestamp'])
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float64)
    require(states.shape == (len(table), JOINT_STATE_WIDTH) and actions.shape == (len(table), JOINT_ACTION_WIDTH)
            and np.isfinite(states).all() and np.isfinite(actions).all(), 'Invalid exported joint data')
    provenance = json.loads((dataset / 'demonstration_provenance.json').read_text())
    require(np.array_equal(np.unique(episodes), np.arange(len(provenance))), 'Provenance episode mismatch')
    pregrasp = np.zeros(len(table), dtype=bool)
    for index, entry in enumerate(provenance):
        selected = episodes == index
        path = Path(entry['source']) / 'trajectory.npz'
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['trajectory_sha256'],
                'Recorded demonstration differs from the export provenance')
        with np.load(path, allow_pickle=False) as raw:
            phase = raw['phase']
            require(np.array_equal(frames[selected], np.arange(len(phase)))
                    and np.allclose(timestamps[selected], raw['timestamp'], rtol=0, atol=4e-6),
                    'Demonstration frame order differs from the exported table')
            pregrasp[selected] = np.isin(phase, ['approach', 'advance', 'close'])
    require(pregrasp.any(), 'Missing demonstrated pre-grasp phases')
    return states[pregrasp], actions[pregrasp], int(np.unique(episodes[pregrasp]).size)


def measure(rollout, lower, upper, norm_lower, norm_upper):
    summary = json.loads((rollout / 'summary.json').read_text())
    require(summary['evaluation_complete'] and summary['action_space'] == 'joint' and summary['total'] == 1,
            'Expected a completed neural joint rollout')
    episode = summary['episodes'][0]
    require(episode['geometry_audit']['passed'] and episode['geometry_audit']['custom_pads'] == 0,
            'Expected official finger geometry')
    directory = rollout / 'episode_000'
    with (np.load(directory / 'model_observations.npz', allow_pickle=False) as observations,
          np.load(directory / 'trajectory.npz', allow_pickle=False) as trajectory):
        seconds, states = observations['timestamp'], observations['state']
        times, commands, contacts = trajectory['timestamp'], trajectory['joint_command'], trajectory['contacts']
    require(states.shape == (len(seconds), JOINT_STATE_WIDTH)
            and commands.shape == (len(times), JOINT_ACTION_WIDTH) and len(contacts) == len(times)
            and np.isfinite(states).all() and np.isfinite(commands).all()
            and len(seconds) and len(times), 'Invalid rollout observations or commands')
    loaded = times[contacts > 0]
    end = float(loaded[0]) if loaded.size else float(episode['simulated_seconds'])
    before = seconds < end
    require(before.any(), 'No measured pre-contact state')
    states, seconds = states[before], seconds[before]
    outside_observed = (states < lower - 1e-4) | (states > upper + 1e-4)
    outside_normalized = (states < norm_lower - 1e-6) | (states > norm_upper + 1e-6)
    first_outside = {name: float(seconds[np.flatnonzero(outside_observed[:, j])[0]])
                     for j, name in enumerate(JOINT_STATE_NAMES) if outside_observed[:, j].any()}
    windows = []
    for horizon in (5., 10., 15.):
        command_window = commands[times < min(horizon, end)]
        observed_window = states[seconds < horizon]
        windows.append(dict(seconds=horizon, before_contact_frames=len(command_window),
                            reverse_fraction=float((command_window[:, 7] < -.01).mean()),
                            observed_aperture_min_mm=float(observed_window[:, 6].min() * APERTURE_SCALE_M * 1000),
                            states_outside_normalizer=int(outside_normalized[seconds < horizon].any(axis=1).sum())))
    return dict(rollout=str(rollout.resolve()), failure_reason=episode['failure_reason'],
                first_recorded_loaded_contact_s=float(loaded[0]) if loaded.size else None,
                first_outside_observed_demonstration_range_s=first_outside,
                windows=windows, comparisons_are_coordinatewise=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rollouts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = Path(cfg['data']['lerobot_config']['root'])
    norm = json.loads(Path(cfg['data']['norm_stats_path']).read_text())['norm_stats']['observation.state']
    states, actions, episodes = pregrasp_demo_rows(dataset)
    lo, hi = states.min(axis=0), states.max(axis=0)
    norm_lower, norm_upper = np.asarray(norm['q01']), np.asarray(norm['q99'])
    report = dict(kind='pregrasp_observation_and_command_support', dataset=str(dataset),
                  demonstration_phases=['approach', 'advance', 'close'], demonstration_frames=len(states),
                  demonstration_episodes=episodes, demonstration_base_forward_range_m_s=[
                      float(actions[:, 7].min()), float(actions[:, 7].max())],
                  demonstration_reverse_fraction_below_0_01_m_s=float((actions[:, 7] < -.01).mean()),
                  observed_state_ranges={name: [float(lo[i]), float(hi[i])]
                                         for i, name in enumerate(JOINT_STATE_NAMES)},
                  trials=[measure(path, lo, hi, norm_lower, norm_upper) for path in args.rollouts],
                  no_new_policy_inference=True, independent_cases_read=False)
    with args.output.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
