"""Load one real batch through the official Wall-X preprocessing pipeline.

Launch with torchrun --nproc_per_node=1; model weights are not required.
"""
import argparse
import hashlib
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
    parser.add_argument("--check-episode",type=int,action="append",default=[],
                        help="Also check first/middle/last usable training anchors from this episode.")
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
        assert (batch["action_chunk"][...,23:]==0).all()
        with_odometry="base_odometry" in cfg.task.agent_pos_config
        if with_odometry:
            assert batch["agent_pos_mask"].bool().all()
            assert raw["observation.state"].shape[-1]==26
        else:assert (batch["proprioception"][...,23:]==0).all()
        explicit_episode_checks=[]
        explicit_episode_training_coverage=[]
        train_indices=np.asarray(inner.train_dataset.indices,dtype=int)
        frame_numbers=np.asarray(raw_frame["frame_index"])
        sample_weights=(loader.sampler.weights.numpy() if hasattr(loader.sampler,"weights")
                        else np.ones(len(train_indices)))
        for episode in args.check_episode:
            selected=np.flatnonzero(episode_ids[train_indices]==episode)
            if not len(selected):raise ValueError(f"Episode {episode} has no usable training anchors")
            # random_split keeps a permuted list of source indices; sort by
            # source frame to actually check the start of the recovery.
            selected=selected[np.argsort(train_indices[selected])]
            source_frames=frame_numbers[train_indices[selected]]
            explicit_episode_training_coverage.append(dict(episode=episode,
                usable_training_anchors=len(selected),first_frame=int(source_frames[0]),
                last_frame=int(source_frames[-1]),source_frame_zero_in_training=bool((source_frames==0).any()),
                initial2s_training_anchors=int((source_frames<40).sum()),
                expected_sampling_fraction=float(sample_weights[selected].sum()/sample_weights.sum())))
            for subset_row in np.unique(selected[[0,len(selected)//2,-1]]):
                source_index=int(train_indices[subset_row])
                source_sample=inner.hf_dataset[source_index]
                processed_sample=inner[int(subset_row)]
                assert torch.equal(processed_sample["agent_pos"],source_sample["observation.state"])
                expected=relative_chunks(np.repeat(source_sample["observation.state"].numpy()[None],
                    len(source_sample["action"]),axis=0),source_sample["action"].numpy(),
                    len(source_sample["action"]))[0]
                assert np.allclose(processed_sample["action"].numpy(),expected,atol=1e-6)
                checked=loader.collate_fn([processed_sample])
                for value in checked.values():
                    if torch.is_tensor(value) and value.is_floating_point():assert torch.isfinite(value).all()
                assert tuple(checked["action_chunk"].shape[-2:])==(cfg.task.action_horizon_flow,26)
                assert not checked["dof_mask"][...,23:].bool().any()
                assert (checked["action_chunk"][...,23:]==0).all()
                if with_odometry:assert checked["agent_pos_mask"].bool().all()
                explicit_episode_checks.append(dict(episode=episode,source_index=source_index,
                    frame_index=int(raw_frame[source_index]["frame_index"]),
                    relative_actions_match=True,robot_state_preserved=True,
                    collated_action_shape=list(checked["action_chunk"].shape),
                    finite=True,padding_masked=True))
        sampling = {"sampler": type(loader.sampler).__name__}
        weights_path = cfg.data.lerobot_config.get("frame_sampling_weights_path")
        if weights_path:
            assert type(loader.sampler).__name__ == "RecoveryDistributedSampler"
            weights = np.load(weights_path, allow_pickle=False)
            selected = weights[np.asarray(inner.train_dataset.indices)]
            assert np.array_equal(loader.sampler.weights.numpy(), selected)
            manifest=json.loads(Path(weights_path).with_suffix(".json").read_text())
            weights_sha=hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
            assert weights_sha==manifest["weights_sha256"]
            recovery_ids=[row["episode"] for row in manifest["recovery_episodes"]]
            selected_recovery=np.isin(episode_ids[np.asarray(inner.train_dataset.indices)],recovery_ids)
            sampling.update(weights_sha256=weights_sha,
                validation_frames_excluded=True,
                expected_recovery_fraction=float(selected[selected_recovery].sum()/selected.sum()))
        report = {"shapes": shapes, "training_iterations_per_epoch": iters,
                  "sampling": sampling,
                  "base_odometry_observed":with_odometry,
                  "active_base_dimensions": [20, 21, 22], "padding_dimensions": [23, 24, 25],
                  "finite": True, "dataset_names": batch.get("dataset_names"),
                  "state_unchanged_and_base_labels_preserved": True,
                  "relative_rotations_match_loader": True,
                  "episode_end_actions_repeat_last": True,
                  "explicit_episode_checks":explicit_episode_checks,
                  "explicit_episode_training_coverage":explicit_episode_training_coverage,
                  "selected_training_episodes": inner.hf_dataset.episodes,
                  "virtual_action_and_state_tail":("Only action tail is padded; all 26 robot-state dimensions observed" if with_odometry else "zero after normalization"),
                  "note": "Data-path validation only; no model forward/backward."}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
