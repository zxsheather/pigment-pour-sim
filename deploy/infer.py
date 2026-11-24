"""Local-only inference. This module never imports the robot SDK."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from deploy.artifacts import ROOT, WALLX_REVISION, verify_checkpoint


STATE_SIZES = {'follow1_pos': 7, 'follow2_pos': 7, 'lift': 1, 'head_pos': 2,
               'velocity_decomposed': 3, 'velocity_decomposed_odom': 3}


def validate_observation(observation):
    for key, size in STATE_SIZES.items():
        value = np.asarray(observation['state'][key])
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid state: {key}')
    for key in ['face_view', 'right_wrist_view']:
        value = np.asarray(observation['views'][key])
        if value.dtype != np.uint8 or value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 3:
            raise ValueError(f'{key} must be RGB uint8 [1,H,W,3]')
    if not observation.get('instruction'):
        raise ValueError('Instruction is required')


def build(checkpoint, wallx, device='cuda:0', seed=1):
    # Wall-X must be the pinned checkout, patched by `artifacts source`.
    import subprocess
    revision = subprocess.check_output(['git', '-C', str(wallx), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != WALLX_REVISION:
        raise ValueError('Unverified Wall-X source revision')
    os.environ['WALLX_VISION_ATTN_IMPLEMENTATION'] = 'sdpa'
    os.environ['WALLX_SMOOTH_ACTION'] = 'false'
    sys.path.insert(0, str(Path(wallx).resolve()))
    sys.path.insert(0, str(ROOT / 'experiments/grasp'))
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available() or not device.startswith('cuda:'):
        raise RuntimeError('This tested inference profile requires an NVIDIA CUDA GPU.')
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    from walloss_runtime import build_policy
    return build_policy(InferConfig(checkpoint_path=str(Path(checkpoint).resolve()),
                                   robot_type='ex001', norm_key='local/pigment_grasp_reactive_v4',
                                   cam_names=['face_view', 'right_wrist_view'], model_device=device,
                                   robot_action_end_ratio=1., robot_action_interpolate_multiplier=1,
                                   smooth_action=False))


def predict(policy, observation):
    validate_observation(observation)
    import torch
    started = time.monotonic()
    with torch.inference_mode():
        output = policy.infer(observation)
    action = np.asarray(output['follow2_pos'], dtype=np.float64)
    if action.shape != (32, 7) or not np.isfinite(action).all():
        raise ValueError(f'Invalid predicted right-arm chunk: {action.shape}')
    return {'follow2_pos': action.tolist(), 'inference_seconds': time.monotonic() - started,
            'load_audit': policy.model_wrapper.load_audit}


def load_observation(path):
    path = Path(path)
    data = json.loads(path.read_text())
    data['state'] = {k: np.asarray(v, dtype=np.float32) for k, v in data['state'].items()}
    data['views'] = {k: np.asarray(Image.open(path.parent / v).convert('RGB'))[None]
                     for k, v in data['views'].items()}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'deploy/checkpoint')
    parser.add_argument('--wallx', type=Path, default=ROOT / 'deploy/vendor/wall-x')
    parser.add_argument('--observation', type=Path, default=ROOT / 'deploy/samples/observation.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'deploy/outputs/offline_prediction.json')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint, full=False)
    policy = build(args.checkpoint, args.wallx, args.device)
    result = predict(policy, load_observation(args.observation))
    result['source'] = 'offline observation; no robot commands sent'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'saved': str(args.output), 'action_shape': [32, 7],
                      'inference_seconds': result['inference_seconds']}))


if __name__ == '__main__':
    main()
