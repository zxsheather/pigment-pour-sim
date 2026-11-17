"""Load one real batch through the official Wall-X preprocessing pipeline.

Launch with torchrun --nproc_per_node=1; model weights are not required.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from compute_cylinder_norm_stats import relative_chunks

from wall_x.config.loader import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.model.qact.qwen2_5.adapter import Qwen2_5Adapter
from wall_x.trainer.adapters.base_adapter import load_trainer_data_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    try:
        cfg = load_config(args.config)
        adapter = Qwen2_5Adapter(cfg=cfg, logger=logging.getLogger("carry_data"))
        action_norm, state_norm, *_ = load_lerobot_normalizers(cfg)
        adapter.normalizer_action, adapter.normalizer_propri = action_norm, state_norm
        processors = adapter.load_processor(None)
        model_config = adapter.build_model_config()
        assert model_config.use_state_string_representation == cfg.task.use_state_string_representation
        dataset, loader, iters = adapter.load_dataset(
            load_trainer_data_config(cfg), processors["processor"], rank=0, world_size=1,
            normalizer_action=action_norm, normalizer_propri=state_norm,
            model_config=model_config, tokenizer_mixin=processors["tokenizer_mixin"])
        batch = next(iter(loader))
        # Compare the first real sample before collation with its source.
        # This catches an accidental second Euler/6D or relative conversion.
        inner = dataset._inner
        subset_index = int(inner.train_dataset.indices[0])
        raw = inner.hf_dataset[subset_index]
        processed = inner[0]
        assert torch.allclose(processed["agent_pos"], raw["observation.state"])
        assert torch.allclose(processed["action"][..., 20:23], raw["action"][..., 20:23])
        for start in (0, 10):
            assert torch.allclose(processed["action"][..., start:start + 3],
                                  raw["action"][..., start:start + 3]
                                  - raw["observation.state"][start:start + 3], atol=1e-6)
        expected=relative_chunks(np.repeat(raw["observation.state"].numpy()[None],len(raw["action"]),axis=0),
                                 raw["action"].numpy(),len(raw["action"]))[0]
        assert np.allclose(processed["action"].numpy(),expected,atol=1e-6)
        # Query a real episode boundary through LeRobot, where future actions
        # must repeat that episode's last action rather than enter another one.
        raw_frame=inner.hf_dataset.hf_dataset
        episode_ids=np.asarray(raw_frame["episode_index"])
        boundary=int(np.flatnonzero(episode_ids==episode_ids[0])[-1])
        end=inner.hf_dataset[boundary]["action"]
        assert torch.allclose(end,end[0].expand_as(end))
        shapes = {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}
        for key, value in batch.items():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                assert torch.isfinite(value).all(), f"Nonfinite batch field: {key}"
        assert batch["dof_mask"].shape[-1] == 26
        assert batch["dof_mask"][..., 20:23].bool().all(), "Base actions were masked out"
        assert not batch["dof_mask"][..., 23:].bool().any(), "Padding contributes to loss"
        report = {"shapes": shapes, "training_iterations_per_epoch": iters,
                  "active_base_dimensions": [20, 21, 22], "padding_dimensions": [23, 24, 25],
                  "finite": True, "dataset_names": batch.get("dataset_names"),
                  "state_unchanged_and_base_labels_preserved": True,
                  "relative_rotations_match_loader": True,
                  "episode_end_actions_repeat_last": True,
                  "selected_training_episodes": inner.hf_dataset.episodes,
                  "note": "Data-path validation only; no model forward/backward."}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
