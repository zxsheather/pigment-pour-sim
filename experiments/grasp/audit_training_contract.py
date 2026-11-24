"""Check train/inference pose layout, control tokens and state bin contract."""
from pathlib import Path
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    from wall_x.config.loader import load_config
    from wall_x.trainer.trainer_utils import load_wallx_processors, load_wallx_processors_from_cfg

    infer = InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001", norm_key="ex_normal",
                        cam_names=["face_view", "right_wrist_view"], model_device="cpu")
    train = load_config(str(args.config))
    assert list(infer.train_config["dof_config"].items()) == list(train.task.dof_config.items())
    assert list(infer.train_config["agent_pos_config"].items()) == list(train.task.agent_pos_config.items())
    expected_bins = infer.train_config.get("state_bins", infer.train_config["data"].get("state_bins", 256))
    assert train._raw_data.get("state_bins", 256) == expected_bins, "State-string bin count differs"
    a = load_wallx_processors(infer.train_config, device="cpu")["processor"].tokenizer
    b = load_wallx_processors_from_cfg(train, device="cpu")["processor"].tokenizer
    tokens = {token: [a.convert_tokens_to_ids(token), b.convert_tokens_to_ids(token)] for token in ["<|propri|>", "<|action|>"]}
    assert all(x == y for x, y in tokens.values()), tokens
    report = {"passed": True, "state_bins": expected_bins, "control_tokens": tokens,
              "action_layout_matches": True, "state_layout_matches": True,
              "training_resolution": train.data.resolution, "normalization": "new dataset statistics, shared by evaluation baseline and adaptation"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
