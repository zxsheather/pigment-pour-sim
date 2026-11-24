"""Compare actual training and serving inputs for identical recorded frames.

Loads processors and normalizers on CPU, without loading model parameters.
Reports the effective action horizon rather than assuming YAML names map to
the same sequence length in the pinned dataset and inference implementations.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sys

os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image

from evaluate_expert_openloop import observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-folder",type=Path)
    p.add_argument("--norm-key",default="local/pigment_grasp_train")
    p.add_argument("--phase-frames",action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from wall_x.config.loader import load_config
    from wall_x.data.backends.lerobot.build import _build_flat_config
    from wall_x.data.backends.lerobot.loader import (
        PreprocessedDataset, DataCollator, get_data_configs, load_norm_stats)
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
    from wall_x._vendor.harrix.serving.policy import wall_x_policy

    class ProcessorOnlyWrapper(WallxModelWrapper):
        def _load_model(self):
            self.model = SimpleNamespace(processor=self.processor)

        def load_ckpt(self, checkpoint_path=None):
            pass

    wall_x_policy.WallxModelWrapper = ProcessorOnlyWrapper
    infer = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
                        norm_key=args.norm_key, cam_names=["face_view", "right_wrist_view"],
                        model_device="cpu", smooth_action=False)
    policy = wall_x_policy.WallXPolicy(infer, image_passing_mode="numpy")
    cfg = load_config(str(args.config))
    flat, lr_cfg = _build_flat_config(cfg)
    data_cfg = get_data_configs(flat["data"])
    stats = load_norm_stats(flat["norm_stats_path"], data_cfg["key_mappings"],
                           dof_config=flat["dof_config"], agent_pos_config=flat["agent_pos_config"])
    dataset = LeRobotDataset(lr_cfg["repo_id"], root=lr_cfg["root"], episodes=[0],
        delta_timestamps={"action":[i/25 for i in range(data_cfg["action_horizon"]-1)]}, video_backend="pyav")
    processed = PreprocessedDataset(dataset, flat, stats, data_cfg, lr_cfg, test_only=True)
    collator = DataCollator(flat, data_cfg, stats, lr_cfg)
    folder = args.episode_folder or args.runs/"expert_pilot_v1/episode_0000"
    with np.load(folder/"trajectory.npz") as raw:
        states = raw["state"].copy()
        actions = raw["action"].copy()
        phases=raw["phase"].copy()
    records = []
    cache = {}
    frame_ids=[0,21,41,42,57,71,72,89,90,110]
    pairs=[(21,41),(41,42),(71,72),(89,90)]
    if args.phase_frames:
        transitions=np.where(phases[1:]!=phases[:-1])[0]+1
        pairs=[(int(i-1),int(i)) for i in transitions]
        frame_ids=sorted(set(np.linspace(0,len(states)-1,min(10,len(states)),dtype=int).tolist()+[i for pair in pairs for i in pair]))
    for frame in frame_ids:
        train = collator([processed[frame]])
        views = {camera:np.asarray(Image.open(folder/camera/f"{frame:06d}.jpg").convert("RGB"))[None]
                 for camera in ["face_view", "right_wrist_view"]}
        obs, instruction = policy._preprocess_obs({"state":observation(states[frame]), "views":views,
            "instruction":"Use your right hand to pick up the blue bottle from the tray and lift it."})
        prefix, postfix = policy.model_wrapper.get_text_for_action(instruction)
        served = policy.model_wrapper.construct_model_input([obs], [prefix], [postfix])
        action_id = collator.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        train_ids = train["input_ids"][0][train["attention_mask"][0].bool()]
        serve_ids = served["input_ids"][0][served["attention_mask"][0].bool()]
        t_prefix = train_ids[:torch.where(train_ids == action_id)[0][0]]
        s_prefix = serve_ids[:torch.where(serve_ids == action_id)[0][0]]
        record = {"frame":frame,
            "proprioception_max_abs_error":float((train["proprioception"]-served["proprioception"]).abs().max()),
            "agent_mask_equal":bool(torch.equal(train["agent_pos_mask"],served["agent_pos_mask"])),
            "prefix_tokens_equal":bool(torch.equal(t_prefix,s_prefix)),
            "train_prefix_tokens":len(t_prefix),"serve_prefix_tokens":len(s_prefix),
            "train_action_tokens":int((train_ids==action_id).sum()),
            "serve_action_tokens":int((serve_ids==action_id).sum()),
            "training_action_chunk_shape":list(train["action_chunk"].shape),
            "image_grid_equal":bool(torch.equal(train["image_grid_thw"],served["image_grid_thw"])),
            "pixel_values_max_abs_error":float((train["pixel_values"]-served["pixel_values"]).abs().max()),
            "pixel_values_mean_abs_error":float((train["pixel_values"]-served["pixel_values"]).abs().mean())}
        records.append(record)
        cache[frame] = {"prefix":t_prefix.clone(),"views":list(views.values())}
    boundaries = []
    for before,after in pairs:
        a,b=cache[before],cache[after]
        boundaries.append({"frames":[before,after],
            "prefix_tokens_equal":bool(torch.equal(a["prefix"],b["prefix"])),
            "observed_xyz_change_mm":float(np.linalg.norm(states[before,7:10]-states[after,7:10])*1000),
            "observed_gripper_change_units":float(abs(states[before,13]-states[after,13])),
            "rgb_mean_absolute_change_0_255":float(np.mean([
                np.abs(x.astype(float)-y.astype(float)).mean() for x,y in zip(a["views"],b["views"])])),
            "expert_xyz_command_jump_mm":float(np.linalg.norm(actions[before,7:10]-actions[after,7:10])*1000),
            "expert_gripper_command_jump_units":float(abs(actions[before,13]-actions[after,13]))})
    report={"model_loaded":False,"frames":records,
            "proprioception_matches":all(r["proprioception_max_abs_error"]<1e-4 for r in records),
            "prefix_tokens_match":all(r["prefix_tokens_equal"] for r in records),
            "near_stationary_phase_boundaries":boundaries,
            "boundary_note":"Small observation changes with target jumps can indicate stage ambiguity; do not prove identical full model inputs or impossibility of learning."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
