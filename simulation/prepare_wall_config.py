"""Generate a bounded action-expert SFT config for a local carry dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cylinder_joint_actions import JOINT_ACTION_LAYOUT, JOINT_STATE_LAYOUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--num-workers",type=int,default=0,choices=range(9))
    parser.add_argument("--save-every",type=int)
    parser.add_argument("--run-name",default="carry_smoke")
    parser.add_argument("--horizon",type=int,default=32,choices=range(1,33))
    parser.add_argument("--front-resolution",type=int,default=224,choices=(224,448))
    parser.add_argument("--train-episode-fraction",type=float,default=.75,
                        help="Official loader selects this leading fraction of episodes; use 1 for a train-only export.")
    parser.add_argument("--state-representation",choices=("string","continuous"),default="string")
    parser.add_argument("--action-space",choices=("cartesian","joint"),default="cartesian",
                        help="Joint mode uses six joint deltas, gripper, two base velocities and 18D robot state.")
    parser.add_argument("--base-odometry",action="store_true",
                        help="Use robot base x/y/yaw in the three previously masked state slots.")
    parser.add_argument("--state-projection-lr",type=float,default=3e-4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--frame-sampling-weights",type=Path,
                        help="Opt-in recovery sampling; requires patch_wall_recovery_sampling.py.")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps must be at least 2 for the official zero-based checkpoint schedule")
    if not 0<args.train_episode_fraction<=1:parser.error("Invalid training episode fraction")
    if args.action_space=="joint" and args.state_representation!="continuous":
        parser.error("Joint representation requires continuous robot state")
    args.output.mkdir(parents=True, exist_ok=False)
    dataset, model, output = (path.resolve() for path in (args.dataset, args.model, args.output))
    architecture = json.loads((model / "config.json").read_text())
    architecture["_attn_implementation"] = "sdpa"
    architecture["vision_config"]["_attn_implementation"] = "sdpa"
    state_string=args.state_representation=="string"
    architecture["use_state_string_representation"]=state_string
    architecture_path = output / "architecture.json"
    action = {"follow_left_ee_cartesian_pos_relative": 3,
              "follow_left_ee_rotation_6D_relative": 6, "follow_left_gripper": 1,
              "follow_right_ee_cartesian_pos_relative": 3,
              "follow_right_ee_rotation_6D_relative": 6, "follow_right_gripper": 1,
              "velocity_decomposed": 3, "action_padding": 3}
    state = {key.removesuffix("_relative"): value for key, value in action.items()}
    if args.base_odometry:
        state.pop("action_padding");state["base_odometry"]=3
    if args.action_space=="joint":
        action=dict(JOINT_ACTION_LAYOUT);state=dict(JOINT_STATE_LAYOUT)
        architecture.update(dof_config=action,ar_dof_config=dict(action),agent_pos_config=state)
    architecture_path.write_text(json.dumps(architecture, indent=2) + "\n")
    info=json.loads((dataset/"meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"]==[18 if args.action_space=="joint" else 26 if args.base_odometry else 23]
    if args.action_space=="joint":
        assert info["features"]["action"]["shape"]==[9]
        assert json.loads((dataset/"joint_export_validation.json").read_text())["passed"]
    config = {
        "model_type": "qwen2_5",
        "task": {"dof_config": action, "ar_dof_config": dict(action), "agent_pos_config": state,
                 "action_horizon": args.horizon, "action_horizon_flow": args.horizon,
                 "use_state_string_representation": state_string},
        "model": {"backbone": "qwen2_5", "config_path": str(architecture_path),
                  "processor_path": str(model), "pretrained_path": None,
                  "attn_implementation": "sdpa", "attn_deterministic": True,
                  "use_ema": False, "flow_loss_weight": 1.0, "ar_loss_weight": 0.0},
        "hyperparams": {"num_epoch": 1000, "batch_size_per_gpu": args.batch_size,
                        "gradient_accumulation_steps": 1, "seed": args.seed,
                        "optimizer": {"optimizer_type": "adamw", "learning_rate": args.learning_rate,
                                      "action_expert_learning_rate": args.learning_rate,
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
                 "train_test_split": args.train_episode_fraction, "num_workers": args.num_workers, "max_length": 1024,
                 "use_state_string_representation": state_string, "action_horizon_flow": args.horizon,
                 "action_horizon_ar": args.horizon,
                 "resolution": {"face_view": args.front_resolution, "left_wrist_view": 224, "right_wrist_view": 224},
                 "camera_name_mapping": {"face_view": "front view", "left_wrist_view": "left wrist view",
                                         "right_wrist_view": "right wrist view"}, "state_bins": 512},
        "logging": {"log_name": args.run_name, "use_wandb": False, "log_interval": 10,
                    # The official trainer checks saves before incrementing
                    # its zero-based step index. Save the final update.
                    "save_interval": args.save_every or args.steps - 1, "val_interval": 1000000, "epoch_save_interval": 1000},
        "checkpoint": {"save_path": str(output / "checkpoints"),
                       "resume_from": str(model / "model.safetensors")},
        "debug": {"profile": False, "nvtx": False},
    }
    if not state_string:
        optimizer=config["hyperparams"]["optimizer"]
        optimizer.pop("action_expert_learning_rate")
        optimizer["lr_groups"]=[
            dict(name="action",lr=args.learning_rate,include=["action_preprocessor.w",
                "action_preprocessor.time_","action_preprocessor.action_proj_back","moe.experts.1",
                "qkv_proj_experts.1","o_proj_experts.1","input_layernorms.1",
                "post_attention_layernorms.1","model.norms.1"],fail_on_empty=True),
            dict(name="state_projection",lr=args.state_projection_lr,
                 include=["action_preprocessor.propri_proj"],fail_on_empty=True)]
    if args.frame_sampling_weights:
        assert args.frame_sampling_weights.is_file()
        config["data"]["lerobot_config"]["frame_sampling_weights_path"]=str(args.frame_sampling_weights.resolve())
    if args.action_space=="joint":
        stats=dataset/"joint_norm_stats.json"
        assert stats.is_file()
        config["data"]["norm_stats_path"]=str(stats)
    path = output / "train.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)


if __name__ == "__main__":
    main()
