"""Check joint data through the pinned official Wall loader and normalizers.

Run with torchrun --nproc_per_node=1. Uses a CPU gloo group and processor only;
no neural weights are loaded and no policy forward/backward is performed.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.distributed as dist

from carry_env import CONTROL_DT
from cylinder_far_approach_prompt import (
    RANGE_CONDITIONED_PROMPT_PROTOCOLS, range_conditioned_prompt_indices, range_conditioned_prompts,
)
from cylinder_joint_actions import (
    ARM_SERVO_RATE_RAD_S, JOINT_SERVO_INCREMENT, absolute_joint_chunk,
    joint_action_representation, joint_state_width, require,
)
from wall_x.config.loader import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.model.qact.qwen2_5.adapter import Qwen2_5Adapter
from wall_x.trainer.adapters.base_adapter import load_trainer_data_config
from wall_x._vendor.harrix.adapters.qwen_vlact import _normalize_real_prefix
from wall_x._vendor.harrix.utils.normalizer import build_normalizers


def check_stratified_sampling(sampler, config, train_indices):
    block_size = config['frame_sampling_block_size']
    require(getattr(sampler, 'block_size', None) == block_size, 'Stratified sampler not activated')
    labels = np.load(config['frame_sampling_groups_path'], allow_pickle=False)[train_indices]
    groups, indices = np.unique(labels, return_inverse=True)
    require(len(sampler.group_members) == len(groups), 'Sampler has different training groups')
    for group_index, members in enumerate(sampler.group_members):
        require(np.array_equal(members.numpy(), np.flatnonzero(indices == group_index)),
                'Sampling groups misaligned with the training split')
    mass = np.bincount(indices, weights=sampler.weights.numpy())
    expected_per_block = mass / mass.sum() * block_size
    draws = np.fromiter(sampler, dtype=np.int64, count=len(sampler))
    blocks = len(draws) // block_size
    require(blocks > 0, 'No complete blocks in the training sampler')
    sampled_groups = indices[draws[:blocks * block_size]].reshape(blocks, block_size)
    group_summary = {}
    for group_index, tag in enumerate(groups):
        counts = (sampled_groups == group_index).sum(axis=1)
        expected = expected_per_block[group_index]
        require(bool((counts >= np.floor(expected + 1e-10)).all())
                and bool((counts <= np.ceil(expected - 1e-10)).all()),
                'Sampling blocks have incorrect group counts')
        group_summary[str(tag)] = dict(expected_per_block=float(expected),
                                       minimum_per_block=int(counts.min()),
                                       maximum_per_block=int(counts.max()))
    return dict(block_size=block_size, complete_blocks=blocks, groups=group_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-episode', type=int, action='append', default=[])
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists')
    logging.basicConfig(level=logging.INFO)
    dist.init_process_group('gloo')
    try:
        cfg = load_config(str(args.config))
        representation = joint_action_representation(cfg.task.dof_config)
        require(representation, 'Unexpected joint action configuration')
        increments = representation == JOINT_SERVO_INCREMENT
        state_width = joint_state_width(cfg.task.agent_pos_config)
        horizon = cfg.task.action_horizon_flow
        require(8 <= horizon <= 32, 'Joint flow horizon must cover the eight executed actions')
        torch.manual_seed(cfg.hyperparams.seed)
        adapter = Qwen2_5Adapter(cfg=cfg, logger=logging.getLogger('joint_data'))
        action_norm, state_norm, *_ = load_lerobot_normalizers(cfg)
        serving_view = args.output.parent/'serving-normalizer-view'
        serving_view.mkdir(exist_ok=True)
        stats = Path(cfg.data.norm_stats_path)
        target_stats = serving_view/'norm_stats.json'
        if target_stats.exists():
            require(target_stats.read_bytes() == stats.read_bytes(), 'Serving stats view changed')
        else:shutil.copyfile(stats, target_stats)
        serving_action, serving_state, serving_key = build_normalizers(
            str(serving_view), {'dof_config': dict(cfg.task.dof_config),
                                'agent_pos_config': dict(cfg.task.agent_pos_config)}, 'cylinder_joint_v1')
        adapter.normalizer_action, adapter.normalizer_propri = action_norm, state_norm
        processors = adapter.load_processor(None)
        model_config = adapter.build_model_config()
        wrist_loss = getattr(model_config, 'joint_servo_wrist_pose_loss', None)
        if increments and wrist_loss and 'tracking_affine_rad' in wrist_loss:
            state_bounds = json.loads(stats.read_text())['norm_stats']['observation.state']
            expected_tracking_affine = [state_bounds[key][19:21] for key in ('q01', 'q99')]
            require(np.array_equal(wrist_loss['tracking_affine_rad'], expected_tracking_affine),
                    'Wrist loss tracking-error bounds differ from the dataset normalizer')
        dataset, loader, iterations = adapter.load_dataset(
            load_trainer_data_config(cfg), processors['processor'], rank=0, world_size=1,
            normalizer_action=action_norm, normalizer_propri=state_norm,
            model_config=model_config, tokenizer_mixin=processors['tokenizer_mixin'])
        batch = next(iter(loader))
        require(tuple(batch['action_chunk'].shape[-2:]) == (horizon, 26), 'Wrong action batch shape')
        require(tuple(batch['proprioception'].shape[-2:]) == (1, 26), 'Wrong state batch shape')
        for tensor in batch.values():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                require(bool(torch.isfinite(tensor).all()), 'Nonfinite training batch')
        inner = dataset._inner
        raw_frame = inner.hf_dataset.hf_dataset
        ids = np.asarray(raw_frame['episode_index'])
        frames = np.asarray(raw_frame['frame_index'])
        train_indices = np.asarray(inner.train_dataset.indices, dtype=int)
        prompt_view = Path(cfg.data.lerobot_config['root']) / 'far_approach_prompt_view.json'
        far_protocol = getattr(model_config, 'joint_far_approach_prompt', None)
        require(prompt_view.is_file() == (far_protocol is not None),
                'Training language view and model prompt protocol must agree')
        prompt_checks = []
        if far_protocol is not None:
            require(far_protocol in RANGE_CONDITIONED_PROMPT_PROTOCOLS
                    and json.loads(prompt_view.read_text())['protocol'] == far_protocol,
                    'Invalid range-conditioned training language protocol')
            task_indices = np.asarray(raw_frame['task_index'])[train_indices]
            prompts = range_conditioned_prompts(far_protocol)
            for task_id, expected_prompt in enumerate(prompts):
                candidates = np.flatnonzero(task_indices == task_id)
                require(len(candidates) > 0, 'A prompt has no training anchor')
                subset_index = int(candidates[0])
                source = inner.hf_dataset[int(train_indices[subset_index])]
                measured = source['observation.state'].numpy()
                phase = 0 if task_id == len(prompts) - 1 else task_id
                require(int(range_conditioned_prompt_indices(measured, phase, far_protocol)) == task_id,
                        'Measured training input does not select its recorded prompt')
                sample = inner[subset_index]
                tokens = loader.collate_fn([sample])['input_ids'][0]
                decoded = processors['processor'].tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
                require(source['task'] == expected_prompt and expected_prompt in decoded,
                        'Recorded language prompt did not enter the training tokens')
                prompt_checks.append(dict(task_index=task_id, prompt=expected_prompt,
                                          source_label_matches=True, collated_tokens_match=True))
        episodes = args.check_episode or [0, int(ids.max())]
        checks = []
        for episode in episodes:
            selected = np.flatnonzero(ids[train_indices] == episode)
            selected = selected[np.argsort(frames[train_indices[selected]])]
            require(len(selected) > 0, 'No usable training anchors')
            for subset_index in np.unique(selected[[0, len(selected)//2, -1]]):
                source_index = int(train_indices[subset_index])
                source = inner.hf_dataset[source_index]
                sample = inner[int(subset_index)]
                raw_action = source['action'].numpy()
                raw_state = source['observation.state'].numpy()
                require(raw_action.shape == (horizon, 9), 'Wrong source action window shape')
                expected = raw_action.astype(np.float64)
                if increments:
                    require(np.abs(expected[:, :6]).max() <= ARM_SERVO_RATE_RAD_S * CONTROL_DT + 1e-8,
                            'Training increment is outside the actuator budget')
                    expected[source['action_is_pad'].numpy(), :6] = 0.
                else:
                    expected[:, :6] -= raw_state[:6]
                require(torch.equal(sample['agent_pos'], source['observation.state']),
                        'Robot state changed during processing')
                require(np.array_equal(sample['action'].numpy(), expected.astype(np.float32)),
                        'Prepared joint action labels differ from the expected source')
                require(np.array_equal(sample['action'].numpy()[:, 6:], raw_action[:, 6:]),
                        'Absolute gripper/base labels changed')
                if not increments:
                    require(np.allclose(absolute_joint_chunk(sample['action'].numpy(), raw_state),
                                        raw_action, rtol=0, atol=2e-7), 'Relative joint decoding mismatch')
                collated = loader.collate_fn([sample])
                require(bool(collated['dof_mask'][..., :9].bool().all())
                        and not bool(collated['dof_mask'][..., 9:].bool().any()), 'Wrong action mask')
                require(bool(collated['agent_pos_mask'][..., :state_width].bool().all())
                        and not bool(collated['agent_pos_mask'][..., state_width:].bool().any()), 'Wrong state mask')
                require(bool((collated['action_chunk'][..., 9:] == 0).all())
                        and bool((collated['proprioception'][..., state_width:] == 0).all()), 'Nonzero virtual padding')
                action_input = torch.zeros(1, len(raw_action), 26)
                state_input = torch.zeros(1, 1, 26)
                action_input[..., :9] = sample['action']
                state_input[..., :state_width] = sample['agent_pos']
                expected_action = _normalize_real_prefix(action_norm, action_input, collated['dataset_names'])
                expected_state = _normalize_real_prefix(state_norm, state_input, collated['dataset_names'])
                served_action = _normalize_real_prefix(serving_action, action_input, [serving_key])
                served_state = _normalize_real_prefix(serving_state, state_input, [serving_key])
                require(torch.allclose(collated['action_chunk'][..., :9], expected_action[..., :9], atol=1e-6),
                        'Inference/training action normalizers differ')
                require(torch.allclose(collated['proprioception'][..., :state_width],
                                       expected_state[..., :state_width], atol=1e-6),
                        'Inference/training robot state normalizers differ')
                require(torch.allclose(collated['action_chunk'][..., :9], served_action[..., :9], atol=1e-6)
                        and torch.allclose(collated['proprioception'][..., :state_width],
                                           served_state[..., :state_width], atol=1e-6),
                        'Checkpoint-serving normalizer factory differs from training')
                checks.append(dict(episode=episode, frame=int(frames[source_index]),
                                   action_labels_exact=True, robot_state_exact=True, gripper_and_base_absolute=True,
                                   normalization_matches_inference=True, masks_and_padding_correct=True))
                if increments:
                    checks[-1]['zero_arm_increments_at_padded_steps'] = True
                else:
                    checks[-1].update(relative_joint_labels_exact=True, decode_matches_source=True)
            boundary = int(np.flatnonzero(ids == episode)[-1])
            end = inner.hf_dataset[boundary]['action']
            require(torch.equal(end, end[0].expand_as(end)), 'Episode end enters another episode')
        weights = np.load(cfg.data.lerobot_config['frame_sampling_weights_path'])
        require(np.array_equal(loader.sampler.weights.numpy(), weights[train_indices]), 'Sampling weights misaligned')
        report = dict(kind='official_wall_joint_data_contract', passed=True, action_representation=representation,
            dataset=str(cfg.data.lerobot_config['root']),
            episodes=len(inner.hf_dataset.episodes), frames=len(raw_frame),
            flow_horizon=horizon,
            training_iterations_per_epoch=iterations,
            batch_shapes={k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
            checks=checks, episode_end_repeats_last=True,
            action_active_dimensions=list(range(9)), robot_state_active_dimensions=list(range(state_width)),
            sampling_weights_match_source_indices=True, processor_loaded=True,
            checkpoint_serving_normalizer_factory_matches_training=True,
            serving_normalizer_key=serving_key,
            distributed_backend='gloo', neural_weights_loaded=False,
            model_forward=False, backward=False, task_success_measured=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if far_protocol is not None:
            report['far_approach_prompt_protocol'] = far_protocol
            report['far_approach_prompt_checks'] = prompt_checks
        if increments and wrist_loss and 'tracking_affine_rad' in wrist_loss:
            report['wrist_loss_tracking_affine_matches_normalizer'] = True
        if cfg.data.lerobot_config.get('frame_sampling_groups_path') is not None:
            report['stratified_sampling'] = check_stratified_sampling(
                loader.sampler, cfg.data.lerobot_config, train_indices)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
