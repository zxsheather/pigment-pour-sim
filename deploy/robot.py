"""Observe, preview, or send ONE bounded right-arm target through X2Robot SDK."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from deploy.artifacts import ROOT, verify_checkpoint
from deploy.guards import check_unchanged, model_observation, proposed_step, validate_config
from deploy.sdk_bridge import SDKBridge


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def save_capture(raw, destination):
    destination.mkdir(parents=True, exist_ok=True)
    data = {k: jsonable(v) for k, v in raw.items() if k != 'views'}
    for key, img in raw.get('views', {}).items():
        Image.fromarray(img).save(destination / f'{key}.png')
    data['timestamp_note'] = 'Local monotonic receipt time; SDK camera synchronization/freshness must be verified on hardware.'
    (destination / 'raw_sdk_observation.json').write_text(json.dumps(data, indent=2) + '\n')


def commission(bridge, policy, config, instruction, execute=False, allow_gripper=False):
    from deploy.infer import predict
    validate_config(config)
    original = bridge.read(cameras=True)
    observation = model_observation(original, config, instruction)
    result = predict(policy, observation)
    fresh = bridge.read(cameras=False)
    check_unchanged(original, fresh, config)
    matrix, grip, report = proposed_step(result['follow2_pos'][0], fresh, config, allow_gripper)
    result['proposal'] = report
    result['commands_sent'] = 0
    if execute:
        # No file I/O or user interaction between this check and the single SDK write.
        age = time.monotonic() - fresh['read_finished_monotonic']
        if not 0 <= age <= config['limits']['max_dispatch_age_seconds']:
            raise ValueError('Dispatch deadline expired; no command sent')
        bridge.write_right(matrix, grip)
        result['commands_sent'] = 1 + int(grip is not None)
    return original, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['observe', 'preview', 'step'])
    parser.add_argument('--config', type=Path, default=ROOT / 'deploy/robot.local.json')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'deploy/checkpoint')
    parser.add_argument('--wallx', type=Path, default=ROOT / 'deploy/vendor/wall-x')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--instruction', default='Use your right hand to pick up the blue bottle from the tray and lift it.')
    parser.add_argument('--output', type=Path, default=ROOT / 'deploy/outputs/robot')
    parser.add_argument('--execute', action='store_true', help='Required with step; sends one bounded target')
    parser.add_argument('--allow-gripper', action='store_true', help='Also permit a bounded right gripper write')
    args = parser.parse_args()
    if args.execute != (args.mode == 'step') or (args.mode == 'observe' and args.allow_gripper):
        parser.error('Only step --execute can write; observe does not accept --allow-gripper')
    config = json.loads(args.config.read_text())
    policy = None
    if args.mode != 'observe':
        validate_config(config)
        verify_checkpoint(args.checkpoint, full=False)
        from deploy.infer import build
        policy = build(args.checkpoint, args.wallx, args.device)
    bridge = SDKBridge.connect(config['endpoint'], config['camera_array_order'])
    if args.mode == 'observe':
        save_capture(bridge.read(), args.output)
        print(f'Saved raw SDK observation: {args.output}; zero write commands')
        return
    raw, result = commission(bridge, policy, config, args.instruction,
                             execute=args.execute, allow_gripper=args.allow_gripper)
    save_capture(raw, args.output)
    (args.output / 'prediction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'mode': args.mode, 'commands_sent': result['commands_sent'], 'proposal': result['proposal']}))


if __name__ == '__main__':
    main()
