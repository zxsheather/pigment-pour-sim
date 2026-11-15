"""Generate a bounded action-expert SFT config for a local carry dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps must be at least 2 for the official zero-based checkpoint schedule")
    args.output.mkdir(parents=True, exist_ok=False)
    dataset, model, output = (path.resolve() for path in (args.dataset, args.model, args.output))
    architecture = json.loads((model / "config.json").read_text())
    architecture["_attn_implementation"] = "sdpa"
    architecture["vision_config"]["_attn_implementation"] = "sdpa"
    architecture_path = output / "architecture.json"
    architecture_path.write_text(json.dumps(architecture, indent=2) + "\n")
    action = {"follow_left_ee_cartesian_pos_relative": 3,
              "follow_left_ee_rotation_6D_relative": 6, "follow_left_gripper": 1,
              "follow_right_ee_cartesian_pos_relative": 3,
              "follow_right_ee_rotation_6D_relative": 6, "follow_right_gripper": 1,
              "velocity_decomposed": 3, "action_padding": 3}
    state = {key.removesuffix("_relative"): value for key, value in action.items()}
    config = {
        "model_type": "qwen2_5",
        "task": {"dof_config": action, "ar_dof_config": dict(action), "agent_pos_config": state,
                 "action_horizon": 32, "action_horizon_flow": 32,
                 "use_state_string_representation": True},
        "model": {"backbone": "qwen2_5", "config_path": str(architecture_path),
                  "processor_path": str(model), "pretrained_path": None,
                  "attn_implementation": "sdpa", "attn_deterministic": True,
                  "use_ema": False, "flow_loss_weight": 1.0, "ar_loss_weight": 0.0},
        "hyperparams": {"num_epoch": 100, "batch_size_per_gpu": 1,
                        "gradient_accumulation_steps": 1, "seed": 42,
                        "optimizer": {"optimizer_type": "adamw", "learning_rate": 1e-5,
                                      "action_expert_learning_rate": 1e-5,
                                      "train_action_expert_only": True,
                                      "max_grad_norm": 1.0, "betas": [0.9, 0.95],
                                      "weight_decay": 1e-8},
                        "scheduler": {"scheduler_type": "constant", "num_training_steps": args.steps}},
        "distributed": {"use_fsdp": False, "use_mixed_precision": True, "bf16": True,
                        "find_unused_parameters": True, "use_gradient_checkpointing": True},
        "data": {"dataset_type": "lerobot", "lerobot_config": {"repo_id": str(dataset), "root": str(dataset)},
                 "key_mappings": {"camera": {f"observation.images.{cam}": cam for cam in
                                            ("face_view", "left_wrist_view", "right_wrist_view")},
                                  "state": "observation.state", "action": "action"},
                 "norm_stats_path": str(output / "norm_stats.json"),
                 "train_test_split": 0.75, "num_workers": 0, "max_length": 1024,
                 "use_state_string_representation": True, "action_horizon_flow": 32,
                 "action_horizon_ar": 32,
                 "resolution": {"face_view": 224, "left_wrist_view": 224, "right_wrist_view": 224},
                 "camera_name_mapping": {"face_view": "front view", "left_wrist_view": "left wrist view",
                                         "right_wrist_view": "right wrist view"}, "state_bins": 512},
        "logging": {"log_name": "carry_smoke", "use_wandb": False, "log_interval": 1,
                    # The official trainer checks saves before incrementing
                    # its zero-based step index. Save the final update.
                    "save_interval": args.steps - 1, "val_interval": 1000000, "epoch_save_interval": 1000},
        "checkpoint": {"save_path": str(output / "checkpoints"),
                       "resume_from": str(model / "model.safetensors")},
        "debug": {"profile": False, "nvtx": False},
    }
    path = output / "train.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)


if __name__ == "__main__":
    main()
