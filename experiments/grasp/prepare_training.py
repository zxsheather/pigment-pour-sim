"""Generate a local action-expert-only smoke configuration, with explicit paths."""
from pathlib import Path
import argparse
import json

from safetensors import safe_open
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-id", default="local/pigment_grasp_pilot")
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    model_config = json.loads((checkpoint/"config.json").read_text())
    with safe_open(checkpoint/"model.safetensors", framework="pt", device="cpu") as f:
        model_config["vocab_size"] = f.get_slice("model.embed_tokens.weight").get_shape()[0]
    model_config["_attn_implementation"] = "sdpa"
    model_config["vision_config"]["_attn_implementation"] = "sdpa"
    architecture = args.output/"model_config.json"
    architecture.write_text(json.dumps(model_config, indent=2))
    base = yaml.safe_load((checkpoint/"config.yml").read_text())
    task = {k: base[k] for k in ["dof_config", "agent_pos_config"]}
    task.update(action_horizon=32, action_horizon_flow=32, use_state_string_representation=True)
    cfg = {
        "model_type": "qwen2_5", "task": task,
        "model": {"config_path": str(architecture.resolve()), "processor_path": str(checkpoint),
                  "pretrained_path": None, "attn_implementation": "sdpa", "use_ema": False,
                  "flow_loss_weight": 1., "ar_loss_weight": 0.},
        "hyperparams": {"num_epoch": 10, "batch_size_per_gpu": 1, "gradient_accumulation_steps": 1, "seed": 2026,
                        "optimizer": {"optimizer_type": "adamw", "learning_rate": 1e-5,
                                      "train_action_expert_only": True, "action_expert_learning_rate": 1e-5},
                        "scheduler": {"scheduler_type": "constant", "num_training_steps": args.steps}},
        "distributed": {"use_fsdp": False, "use_mixed_precision": True, "bf16": True,
                        "use_gradient_checkpointing": True, "find_unused_parameters": True},
        "data": {"dataset_type": "lerobot", "lerobot_config": {"repo_id": args.repo_id, "root": str(args.dataset.resolve())},
                 "norm_stats_path": str((args.output/"norm_stats.json").resolve()),
                 "key_mappings": {"camera": {"observation.images.face_view": "face_view", "observation.images.right_wrist_view": "right_wrist_view"},
                                  "state": "observation.state", "action": "action"},
                 "resolution": {"face_view": 224, "right_wrist_view": 224}, "max_length": 1024,
                 "state_bins": int(base.get("state_bins", base.get("data", {}).get("state_bins", 512))),
                 "use_state_string_representation": True,
                 "train_test_split": .875, "num_workers": 0},
        "checkpoint": {"resume_from": str(checkpoint/"model.safetensors"), "save_path": str((args.output/"checkpoints").resolve())},
        "logging": {"use_wandb": False, "log_interval": 1, "save_interval": args.steps,
                    "val_interval": 1000, "epoch_save_interval": 1000},
    }
    path = args.output/"train.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(path)


if __name__ == "__main__":
    main()
