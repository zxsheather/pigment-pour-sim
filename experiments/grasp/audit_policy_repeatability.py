"""Repeat identical input/seed before accepting fresh paired policy comparisons."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from diagnostic_environment import DiagnosticEnvironment
from reactive_expert import apply_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/guarded_grasp_v9/repeatability"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from walloss_runtime import build_policy
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    policy = build_policy(InferConfig(checkpoint_path=str(args.checkpoint.resolve()), robot_type="ex001",
        norm_key="local/pigment_grasp_reactive_v4", cam_names=["face_view", "right_wrist_view"],
        model_device="cuda:0", robot_action_end_ratio=1., robot_action_interpolate_multiplier=1, smooth_action=False))
    source = Path("experiments/grasp/runs/reactive_train_v4/episode_0000")
    env = DiagnosticEnvironment(source / "input_scene.xml", Path("experiments/grasp/generated/blue_only/scene_config.json"),
        args.output / "scene", "blue", tcp="grasp", controller_label="REPEATABILITY ONLY; NO EXECUTED ACTIONS")
    apply_setup(env, json.loads((source / "episode.json").read_text())["setup"])
    state, views = env.observation()
    obs = {"state": state, "views": views, "instruction": "Use your right hand to pick up the blue bottle from the tray and lift it."}
    constructor = policy.model_wrapper.construct_model_input
    inputs = []
    def traced(*args, **kwargs):
        result = constructor(*args, **kwargs)
        hashes = {}
        for key, value in result.items():
            if torch.is_tensor(value):
                data = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                hashes[key] = hashlib.sha256(data).hexdigest()
            else:
                hashes[key] = repr(value)
        inputs.append(hashes)
        return result
    policy.model_wrapper.construct_model_input = traced
    actions = []
    try:
        for index in range(5):
            policy.reset(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
            with torch.inference_mode():
                policy.infer(copy.deepcopy(obs))  # One discarded warmup per repeat.
            policy.reset(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
            with torch.inference_mode():
                result = policy.infer(copy.deepcopy(obs))
            action = np.asarray(result["follow2_pos"])
            actions.append(action.tolist())
            print(json.dumps({"repeat": index, "max_abs_difference_from_first": float(np.max(np.abs(action - actions[0])))}), flush=True)
    finally:
        env.close()
    differences = [float(np.max(np.abs(np.asarray(action) - actions[0]))) for action in actions]
    checked_inputs = inputs[1::2]
    report = {"completed": True, "repeats": len(actions), "model_inputs_identical": all(v == checked_inputs[0] for v in checked_inputs),
        "max_abs_differences": differences, "exactly_repeatable": max(differences) == 0.,
        "actions": actions, "model_inputs": inputs, "torch_deterministic_algorithms": True,
        "python_numpy_torch_seed": 0, "warmup_per_repeat": 1}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
