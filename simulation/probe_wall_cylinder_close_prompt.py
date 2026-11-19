"""Compare Wall prompts on the same reserved RGB, robot state and flow noise."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import yaml

from audit_cylinder_closure_cue import sha256
from carry_env import CAMERAS, CONTROL_DT
from cylinder_joint_actions import APERTURE_SCALE_M, servo_increment_targets
from cylinder_progress_prompts import PROGRESS_PROMPTS
from cylinder_wrist_pose_loss import wrist_markers
from run_wall_cylinder import CylinderWallPolicy


CLOSE_PROMPT = ('Close the parallel fingers around the measuring cylinder. '
                'Hold the wrist position steady and keep the cylinder upright.')
NOISE_SEEDS = (47, 229, 417)
HORIZON = 8
# The official inspection servo caps each finger at 0.03 m/s.
FINGER_RATE_M_S = .03


def read(path):
    return json.loads(path.read_text())


def read_images(source, indices):
    images = {index: {} for index in indices}
    for camera in CAMERAS:
        with av.open(str(source/f'training_{camera}.mp4')) as video:
            for i, frame in enumerate(video.decode(video=0)):
                if i in images:
                    if abs(float(frame.time) - i * CONTROL_DT) >= 1e-5:
                        raise ValueError('RGB timestamp differs from the recorded control tick')
                    images[i][camera] = frame.to_ndarray(format='rgb24')
                if i == max(images):
                    break
    if any(set(frame) != set(CAMERAS) for frame in images.values()):
        raise ValueError('Missing a recorded camera image')
    return images


def kinematics(outputs, previous, teacher, calibration):
    applied = previous.copy()
    low, high = np.asarray(calibration['joint_limits_rad'])
    q = []
    for prediction in outputs[:HORIZON, :9]:
        target = servo_increment_targets(prediction, applied, CONTROL_DT)
        applied[:6] = np.clip(target[:6], low, high)
        requested_half_width = np.clip(target[6], 0., 1.) * APERTURE_SCALE_M / 2
        half_width_delta = np.clip(requested_half_width - applied[6], -FINGER_RATE_M_S * CONTROL_DT,
                                   FINGER_RATE_M_S * CONTROL_DT)
        applied[6] += half_width_delta
        q.append(applied[:6].copy())
    reference = wrist_markers(torch.from_numpy(teacher[:, :6]), calibration).numpy()
    forecast = wrist_markers(torch.from_numpy(np.asarray(q)), calibration).numpy()
    offset = forecast[:, 0] - reference[:, 0]
    lateral_axis = (reference[:, 2] - reference[:, 0]) / calibration['orientation_marker_offset_m']
    lateral = np.sum(lateral_axis * offset, axis=1)
    return dict(lateral_rms_mm=float(1000 * np.sqrt(np.mean(lateral**2))),
                tip_rms_mm=float(1000 * np.sqrt(np.mean(np.sum(offset**2, axis=1)))),
                final_applied_aperture_mm=float(2000 * applied[6]))


def probe(args):
    reserve = read(args.reserve)
    baseline = read(args.matched_observations)
    dataset = Path(yaml.safe_load(args.config.read_text())['data']['lerobot_config']['root'])
    provenance = dataset/'demonstration_provenance.json'
    fit_seeds = {row['metadata']['result']['seed'] for row in read(provenance)}
    reserve_seeds = {row['seed'] for row in reserve['results']}
    if (reserve['used_for_any_training'] or reserve['independent_cases_read']
            or fit_seeds & reserve_seeds or Path(baseline['source']).resolve() != args.reserve.resolve()):
        raise ValueError('Expected the same reserved teacher observations, with seeds absent from training')
    calibration = read(args.calibration)['config']
    raw_forecasts = {}
    records = []
    policy = CylinderWallPolicy(args.model, args.config, 'cylinder_joint_v1')
    for entry in reserve['results']:
        source = Path(entry['source'])
        scenes = [row for row in baseline['frames'] if row['seed'] == entry['seed']]
        images = read_images(source, {row['frame'] for row in scenes})
        with (np.load(source/'trajectory.npz', allow_pickle=False) as trace,
              np.load(entry['training_arrays'], allow_pickle=False) as labels):
            for row in scenes:
                frame = row['frame']
                state = labels['state'][frame]
                teacher = trace['joint_command'][frame:frame+HORIZON]
                phase = int(labels['task_index'][frame])
                if phase != row['task_index']:
                    raise ValueError('Recorded prompt phase differs')
                previous = trace['joint_command'][frame-1]
                teacher_aperture = 2000 * teacher[-1, 6]
                np.testing.assert_allclose(teacher_aperture, row['teacher_executed_aperture_mm'][-1], atol=1e-5)
                np.testing.assert_allclose(2000 * previous[6], (state[6]+state[24])*APERTURE_SCALE_M*1000, atol=1e-5)
                modes = dict(original=PROGRESS_PROMPTS[phase], steady_close=CLOSE_PROMPT)
                result = dict(seed=entry['seed'], frame=frame, window=row['window'],
                              teacher_final_aperture_mm=float(teacher_aperture), predictions={})
                for tag, instruction in modes.items():
                    metrics = []
                    for noise in NOISE_SEEDS:
                        torch.manual_seed(noise)
                        observation = dict(state=state, images=images[frame], instruction=instruction)
                        outputs, = policy.predict_many([observation])
                        raw_forecasts[f"{entry['seed']}_{frame}_{tag}_{noise}"] = outputs
                        metrics.append(dict(noise_seed=noise, **kinematics(outputs, previous, teacher, calibration)))
                    result['predictions'][tag] = metrics
                records.append(result)
                print(entry['seed'], row['window'],
                      {tag: [round(x['final_applied_aperture_mm'], 2) for x in metrics]
                       for tag, metrics in result['predictions'].items()}, flush=True)
    forecast = args.output.with_suffix('.npz')
    np.savez_compressed(forecast, **raw_forecasts)
    return dict(kind='paired_close_language_probe_on_reserved_teacher_observations',
                model=str(args.model), training_provenance_sha256=sha256(provenance),
                config_sha256=sha256(args.config), reserve_sha256=sha256(args.reserve),
                raw_forecasts_path=str(forecast), raw_forecasts_sha256=sha256(forecast),
                comparison_prompt=CLOSE_PROMPT, independent_cases_read=False, simulator_advanced=False,
                new_training_or_formal_evaluation=False, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('model', 'config', 'reserve', 'matched-observations', 'calibration', 'output'):
        parser.add_argument(f'--{arg}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    report = probe(args)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
