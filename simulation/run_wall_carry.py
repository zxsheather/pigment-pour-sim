"""Run Wall-X flow inference with carry observations and base-only execution.

Uses the public Wall-X source at ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f.
The upper body remains under the holding controller. Only vx and yaw are
executed; predicted arm motions and lateral base velocity are logged.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch
from accelerate import init_empty_weights

from carry_env import CarryEnv, carry_instruction
from wall_x._vendor.harrix.adapters.qwen_vlact import QwenVLActInferAdapter, _normalize_real_prefix
from wall_x._vendor.harrix.utils.ckpt_load import load_state_dict
from wall_x._vendor.harrix.utils.normalizer import build_normalizers
from wall_x._vendor.harrix.utils.train_config import (
    build_data_config, load_train_config_with_ckpt_overlay,
    normalize_train_config_for_inference, register_data_backend,
)
from wall_x.model.qact.qwen2_5.adapter import Qwen2_5Adapter
from wall_x.trainer.trainer_utils import load_wallx_processors


class CarryWallPolicy(QwenVLActInferAdapter):
    """Reuse official prompt/preprocessing while preserving the mobile DOFs."""

    @classmethod
    def _training_adapter(cls):
        return Qwen2_5Adapter

    def __init__(self, checkpoint: Path, train_config: Path, norm_key: str):
        checkpoint, train_config = checkpoint.resolve(), train_config.resolve()
        self._device = "cuda"
        self._action_mode = "flow"
        self._train_config = normalize_train_config_for_inference(
            load_train_config_with_ckpt_overlay(str(train_config), str(checkpoint)), str(train_config))
        config = self._train_config
        register_data_backend(config)
        self._normalizer_action, self._normalizer_propri, self._norm_key = build_normalizers(
            str(checkpoint), config, norm_key)
        self._data_config = build_data_config(str(train_config), config)
        self._cam_names = list(config.get("data", {}).get("key_mappings", {}).get("camera", {}).values())
        if not self._cam_names:
            self._cam_names = ["face_view", "right_wrist_view"]
        self._action_horizon = int(config.get("data", {}).get("action_horizon_flow", 32))
        self._action_dim = sum(config["dof_config"].values())
        self._robot_id = "10000"
        # Keep velocity in exactly the pretraining slots 20:23.
        for layout in (config["dof_config"], config["agent_pos_config"]):
            keys = list(layout)
            if (sum(list(layout.values())[:6]) != 20 or keys[6] != "velocity_decomposed"
                    or layout[keys[6]] != 3 or sum(layout.values()) != 26):
                raise ValueError(f"Unsupported carry layout: {layout}")
        model_config = Qwen2_5Adapter.config_class().from_json_file(str(checkpoint / "config.json"))
        model_config.update_model_config(config)
        model_config._attn_implementation = "sdpa"
        model_config.vision_config._attn_implementation = "sdpa"
        procs = load_wallx_processors(config, normalizer=self._normalizer_action, device="cuda")
        self._processor, self._tokenizer_mixin = procs["processor"], procs.get("tokenizer_mixin")
        model_class = Qwen2_5Adapter.model_class()
        print("Constructing Wall-X model", flush=True)
        with init_empty_weights():
            model = model_class(model_config, self._processor, self._tokenizer_mixin)
        print("Loading checkpoint weights", flush=True)
        state = load_state_dict(str(checkpoint), model_class)
        model.resize_token_embeddings(state["model.embed_tokens.weight"].shape[0], mean_resizing=False)
        result = model.load_state_dict(state, strict=False, assign=True)
        # Normalizers are supplied explicitly, and lm_head may share embeddings.
        unexplained = [key for key in result.missing_keys
                       if "normalizer" not in key and key != "lm_head.weight"]
        unexpected = [key for key in result.unexpected_keys if "normalizer" not in key]
        if unexplained or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={unexplained}, unexpected={unexpected}")
        model.tie_weights()
        del state
        model.set_normalizer(copy.deepcopy(self._normalizer_action), copy.deepcopy(self._normalizer_propri))
        model.eval().to("cuda")
        model.to_bfloat16_for_selected_params()
        self._model = model
        print("Wall-X model ready on CUDA", flush=True)
        self.load_report = {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
                            "norm_key": self._norm_key, "cameras": self._cam_names,
                            "attention": "sdpa", "torch": torch.__version__}

    @torch.inference_mode()
    def predict(self, observation: dict, instruction: str) -> np.ndarray:
        state = np.zeros((1, 1, 26), dtype=np.float32)
        state[..., :23] = observation["state"]
        state_mask = np.zeros_like(state)
        state_mask[..., :23] = 1
        # Height/head are unobserved by this policy and kept by the controller.
        dof_mask = np.zeros((1, self._action_horizon, 26), dtype=np.float32)
        dof_mask[..., :23] = 1
        encoded = dict(proprioception=state, agent_pos_mask=state_mask, dof_mask=dof_mask,
                       **observation["images"])
        prefix, postfix = self._get_flow_prompt(instruction)
        inputs = self._construct_model_input([encoded], [prefix], [postfix])
        padding = _normalize_real_prefix(self._normalizer_action, torch.zeros(1, 1, 26),
                                         inputs["dataset_names"]).to("cuda")
        result = self._model.generate_flow_action(action_horizon=self._action_horizon,
                                                  action_dim=26, num_inference_timesteps=10,
                                                  padding_action=padding, **inputs)
        actions = result["predict_action"].float().cpu().numpy()[0]
        if actions.shape != (self._action_horizon, 26) or not np.isfinite(actions).all():
            raise ValueError("Model returned invalid actions")
        return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint directory, not weight file.")
    parser.add_argument("--config", type=Path, help="Defaults to checkpoint/config.yml.")
    parser.add_argument("--norm-key", default="ex_normal")
    parser.add_argument("--color", default="blue", choices=("blue", "red", "green", "yellow"))
    parser.add_argument("--goal", default="B", choices=("A", "B"))
    parser.add_argument("--instruction", help="Override the visual goal description.")
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--start-jitter", default=0.03, type=float)
    parser.add_argument("--heading-jitter-deg", default=0.0, type=float)
    parser.add_argument("--execute-steps", type=int, default=8, help="Replan every N frames (1..32).")
    parser.add_argument("--max-seconds", type=float, default=75)
    parser.add_argument("--probe-only", action="store_true", help="Generate one chunk without moving.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.execute_steps <= 32:
        parser.error("execute-steps must be in [1,32]")
    if (not np.isfinite([args.start_jitter, args.heading_jitter_deg, args.max_seconds]).all()
            or min(args.start_jitter, args.heading_jitter_deg) < 0 or args.max_seconds <= 0):
        parser.error("jitter ranges must be nonnegative and max-seconds must be positive; all must be finite")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    policy = CarryWallPolicy(args.model, args.config or args.model / "config.yml", args.norm_key)
    (args.output / "model_load.json").write_text(json.dumps(policy.load_report, indent=2) + "\n")
    env = CarryEnv(render=True, max_seconds=args.max_seconds)
    chunks, latencies = [], []
    states, applied = [], []
    instruction = args.instruction or carry_instruction(args.color, args.goal)
    observation = env.reset(color=args.color, goal=args.goal, seed=args.seed,
                            start_jitter=args.start_jitter,
                            heading_jitter=float(np.deg2rad(args.heading_jitter_deg)))
    writer = imageio.get_writer(args.output / "overview.mp4", fps=20, codec="libx264", macro_block_size=1)
    info = None
    try:
        while not env.done:
            start = time.monotonic()
            chunk = policy.predict(observation, instruction)
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - start)
            chunks.append(chunk)
            np.save(args.output / "predicted_chunks.npy", np.asarray(chunks))
            print(json.dumps({"chunk": len(chunks), "inference_seconds": latencies[-1],
                              "first_base_velocity": chunk[0, 20:23].tolist()}), flush=True)
            if args.probe_only:
                imageio.imwrite(args.output / "initial_view.png", env.render("carry_overview"))
                break
            for action in chunk[:args.execute_steps]:
                states.append(observation["state"])
                writer.append_data(env.render("carry_overview"))
                observation, info = env.step(action[[20, 22]])
                applied.append(env.action_vector())
                if env.done:
                    break
    finally:
        writer.close()
        env.close()
    np.savez_compressed(args.output / "trajectory.npz", state=np.asarray(states), action=np.asarray(applied))
    summary = {"policy": "wall_x_base_only", "instruction": instruction, "probe_only": args.probe_only,
               "result": info, "inference_seconds": latencies, "model": str(args.model.resolve()),
               "norm_key": policy.load_report["norm_key"], "seed": args.seed,
               "start_jitter_m": args.start_jitter, "heading_jitter_deg": args.heading_jitter_deg,
               "execution": "Only vx and yaw; arms/grippers held by servo controller; vy ignored."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
