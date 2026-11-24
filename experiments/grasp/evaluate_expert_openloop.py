"""Compare deployed model actions to recorded expert commands, without physics.

Uses the exact inference observation adapter and model serializer from the
closed-loop experiment. Dataset truth supplies targets for scoring only.
Training-pool episode probes and unused internal episode probes are separate;
the independent external holdout is not opened by this script.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image


ANCHORS = [0, 21, 41, 42, 57, 71, 72, 80, 89, 90, 110, 140]
PROBES = [
    ("train_episode", "expert_pilot_v1", 0),
    ("train_episode", "expert_pilot_v1", 6),
    ("unused_internal_episode", "expert_train_extra_v1", 13),
    ("unused_internal_episode", "expert_train_extra_v1", 14),
    ("unused_internal_episode", "expert_train_extra_v1", 15),
]


def observation(vector):
    return {"follow1_pos": vector[:7].copy(), "follow2_pos": vector[7:14].copy(),
            "velocity_decomposed": vector[14:17].copy(),
            "velocity_decomposed_odom": vector[14:17].copy(),
            "lift": vector[17:18].copy(), "head_pos": vector[18:20].copy()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-key", default="local/pigment_grasp_train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=ANCHORS)
    p.add_argument("--probe-ids", type=int, nargs="+")
    p.add_argument("--cases",type=Path,help="JSON list of split/batch/episode cases")
    p.add_argument("--phase-anchors",action="store_true",help="Use start/middle/end of each recorded phase for variable-length episodes")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from walloss_runtime import build_policy
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    config = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
                         norm_key=args.norm_key, cam_names=["face_view", "right_wrist_view"],
                         model_device="cuda:0", robot_action_end_ratio=1.,
                         robot_action_interpolate_multiplier=1, smooth_action=False)
    torch.manual_seed(args.seed)
    policy = build_policy(config)
    records = []
    probes=([(c["split"],c["batch"],c["episode"]) for c in json.loads(args.cases.read_text())] if args.cases else PROBES)
    for split, batch, episode in [probes[i] for i in (args.probe_ids if args.probe_ids is not None else range(len(probes)))]:
        folder = args.runs/batch/f"episode_{episode:04d}"
        with np.load(folder/"trajectory.npz") as raw:
            states, actions, phases = raw["state"], raw["action"], raw["phase"]
            frames=args.frames
            if args.phase_anchors:
                frames=sorted({int(i) for phase in np.unique(phases)
                               for i in np.asarray(np.where(phases==phase)[0])[[0,len(np.where(phases==phase)[0])//2,-1]]})
            for frame in frames:
                if not 0<=frame<len(states):raise ValueError(f"Frame {frame} outside episode length {len(states)}")
                state = observation(states[frame])
                views = {camera: np.asarray(Image.open(folder/camera/f"{frame:06d}.jpg").convert("RGB"))[None]
                         for camera in ["face_view", "right_wrist_view"]}
                # Identical deterministic flow-noise seed for paired checkpoints.
                torch.manual_seed(args.seed + episode * 1000 + frame)
                policy.reset()
                started = time.monotonic()
                with torch.inference_mode():
                    result = policy.infer({"state": state, "views": views,
                        "instruction": "Use your right hand to pick up the blue bottle from the tray and lift it."})
                prediction = np.asarray(result["follow2_pos"], dtype=float)
                count = min(len(prediction), len(actions) - frame)
                target = actions[frame:frame+count, 7:14]
                xyz = np.linalg.norm(prediction[:count, :3] - target[:, :3], axis=1) * 100
                grip = np.abs(prediction[:count, 6] - target[:, 6])
                record = {"split": split, "batch": batch, "episode": episode,
                          "frame": frame, "phase": str(phases[frame]),
                          "first_xyz_cm": float(xyz[0]), "execute8_xyz_cm": float(xyz[:8].mean()),
                          "chunk_xyz_cm": float(xyz.mean()), "first_gripper_units": float(grip[0]),
                          "execute8_gripper_units": float(grip[:8].mean()),
                          "prediction": prediction.tolist(), "target": target.tolist(),
                          "inference_seconds": time.monotonic() - started}
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in ["prediction", "target"]}), flush=True)
    aggregates = {}
    for split in sorted({r["split"] for r in records}):
        selected = [r for r in records if r["split"] == split]
        aggregates[split] = {key: float(np.mean([r[key] for r in selected])) for key in
                            ["first_xyz_cm", "execute8_xyz_cm", "chunk_xyz_cm", "first_gripper_units", "execute8_gripper_units"]}
        aggregates[split]["samples"] = len(selected)
        aggregates[split]["phases"] = {phase: {
            "first_xyz_cm": float(np.mean([r["first_xyz_cm"] for r in selected if r["phase"] == phase])),
            "first_gripper_units": float(np.mean([r["first_gripper_units"] for r in selected if r["phase"] == phase]))}
            for phase in sorted({r["phase"] for r in selected})}
    report = {"model": "X-Square-Robot/wall-oss-0.5", "seed": args.seed,
              "mode": "teacher-observation open-loop, no physics execution",
              "checkpoint_name": args.checkpoint.parent.parent.name + "/" + args.checkpoint.name,
              "targets": "issued expert commands, including known normalizer clipping differences",
              "object_pose_given_to_policy": False, "external_holdout_used": False,
              "load_audit": policy.model_wrapper.load_audit, "aggregates": aggregates, "records": records}
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(aggregates), flush=True)


if __name__ == "__main__":
    main()
