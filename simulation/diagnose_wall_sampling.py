"""Measure flow-sampling variation at identical recorded policy observations.

Runs the real network without advancing physics. Variation is not task error;
this diagnostic alone cannot select an execution controller or prove success.
"""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from carry_env import CAMERAS
from cylinder_actions import relative_to_absolute, rotation_from_rows
from cylinder_joint_actions import absolute_joint_chunk,APERTURE_SCALE_M
from run_wall_cylinder import CylinderWallPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--norm-key", default="cylinder_official_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or not 1 <= args.batch_size <= args.samples:
        parser.error("Need at least two samples and a valid batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    policy = CylinderWallPolicy(args.model, args.config, args.norm_key)
    recorded = np.load(args.episode / "model_observations.npz")
    rows = []
    for index in args.snapshots:
        state = recorded["state"][index]
        folder = args.episode / "model_observations" / f"chunk_{index:04d}"
        images = {name: imageio.imread(folder / f"{name}.png") for name in CAMERAS}
        observation = dict(state=state, images=images)
        chunks = []
        for start in range(0, args.samples, args.batch_size):
            batch = [observation] * min(args.batch_size, args.samples-start)
            chunks.extend(policy.predict_many(batch))
        chunks=np.asarray(chunks)
        absolute = np.stack([(absolute_joint_chunk(chunk,state) if policy.action_space=='joint'
                              else relative_to_absolute(chunk,state[:23])) for chunk in chunks])
        np.save(args.output / f"snapshot_{index:03d}_raw_chunks.npy", chunks)
        np.save(args.output / f"snapshot_{index:03d}_actions.npy", absolute)
        prefix = absolute[:, :8]
        row = dict(snapshot=index, simulated_seconds=float(recorded["timestamp"][index]),
                   state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                   image_sha256={name: hashlib.sha256((folder/f"{name}.png").read_bytes()).hexdigest()
                                 for name in CAMERAS})
        if policy.action_space=='joint':
            row.update(joint_target_std_rad_per_tick=prefix[...,:6].std(0).tolist(),
                       absolute_action_mean_per_tick=prefix.mean(0).tolist(),
                       right_half_gap_std_m_per_tick=(prefix[...,6].std(0)*APERTURE_SCALE_M/2).tolist(),
                       base_forward_std_m_s_per_tick=prefix[...,7].std(0).tolist(),
                       base_yaw_std_rad_s_per_tick=prefix[...,8].std(0).tolist())
        else:
            positions = prefix[..., 10:13]
            position_std = np.sqrt(positions.var(0).sum(-1))
            angles = []
            for tick in range(prefix.shape[1]):
                rotations = np.stack([rotation_from_rows(value) for value in prefix[:,tick,13:19]])
                mean_rotation = rotation_from_rows(prefix[:,tick,13:19].mean(0))
                angles.append(np.sqrt(np.mean(Rotation.from_matrix(
                    rotations @ mean_rotation.T).magnitude() ** 2)))
            row.update(right_position_rms_std_m_per_tick=position_std.tolist(),
                       right_rotation_rms_std_rad_per_tick=angles,
                       right_half_gap_std_m_per_tick=(prefix[...,19].std(0)*.0975/2).tolist(),
                       base_forward_std_m_s_per_tick=prefix[...,20].std(0).tolist(),
                       base_yaw_std_rad_s_per_tick=prefix[...,22].std(0).tolist())
        reference=np.load(args.episode/'predicted_chunks.npy',mmap_mode='r')[index]
        row['first_draw_raw_max_abs_difference_from_recorded']=float(np.abs(chunks[0]-reference).max())
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(model=str(args.model.resolve()), config=str(args.config.resolve()),
                  episode=str(args.episode.resolve()), samples=args.samples, seed=args.seed,
                  action_space=policy.action_space,robot_state_dimensions=policy.state_width,
                  load_report=policy.load_report, observations=rows,
                  physics_advanced=False, task_success_measured=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note="Same observation, independent standard Gaussian flow samples; no teacher action or object state.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
