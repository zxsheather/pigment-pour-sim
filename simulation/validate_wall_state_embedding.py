"""Compare real inference state embeddings with the actual training method.

Uses fixed recorded RGB/state and real model weights. No physics is advanced,
and no task score is produced. Native inference is compared with a version
that rounds state/mask to the embedding dtype exactly as training does.
"""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from carry_env import CAMERAS
from cylinder_actions import relative_to_absolute
from run_wall_cylinder import CylinderWallPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 70])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-key", default="cylinder_official_v2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policy = CylinderWallPolicy(args.model, args.config, args.norm_key, state_embedding_precision="native")
    model = policy._model
    if model.config.use_state_string_representation:
        raise ValueError("This check requires the continuous-state model")
    prepare = model._prepare_flow_action_inputs
    mode = "native"
    comparisons = []
    baseline_reference = None

    def capture(*positional, **kwargs):
        nonlocal baseline_reference
        original_state = kwargs["proprioception"]
        original_mask = kwargs["agent_pos_mask"]
        context = prepare(*positional, **kwargs)
        actual = context["inputs_embeds"]
        ids = kwargs["input_ids"]
        selected = ids == model.action_token_id_set["propri_token_id"]
        if int(selected.sum()) != len(ids):
            raise RuntimeError("Expected one actual state token per observation")
        reference = model.scatter_proprioception_embeddings(
            ids, torch.zeros_like(actual), original_state, kwargs["dataset_names"], original_mask)
        training_state = reference[selected].float()
        if baseline_reference is None:
            baseline_reference = training_state.detach().clone()
        x, y = actual[selected].float(), baseline_reference
        difference = (x-y).abs()
        comparisons.append(dict(mode=mode, state_token_count=int(selected.sum()),
            state_input_dtype=str(original_state.dtype), embedding_dtype=str(actual.dtype),
            projection_weight_dtype=str(model.action_preprocessor.propri_proj.weight.dtype),
            actual_training_method="scatter_proprioception_embeddings",
            max_embedding_difference=float(difference.max()),
            unequal_embedding_elements=int(torch.count_nonzero(difference)),
            training_reference_same_as_native_call=bool(torch.equal(training_state, baseline_reference)),
            training_embedding_exactly_equal=bool(torch.equal(x, y))))
        return context

    model._prepare_flow_action_inputs = capture
    saved = np.load(args.episode / "model_observations.npz")
    rows = []
    try:
        for index in args.snapshots:
            baseline_reference = None
            folder = args.episode / "model_observations" / f"chunk_{index:04d}"
            state = saved["state"][index].copy()
            images = {camera: imageio.imread(folder/f"{camera}.png") for camera in CAMERAS}
            observation = dict(state=state, images=images)
            predictions = []
            captures = []
            for selected_mode in ["native", "native_repeat", "training_dtype"]:
                mode = "native" if selected_mode == "native_repeat" else selected_mode
                policy.state_embedding_precision = "training" if mode == "training_dtype" else "native"
                torch.manual_seed(47)
                prediction = policy.predict_many([observation])[0]
                predictions.append(prediction)
                captures.append(dict(comparisons[-1], call=selected_mode))
            native, repeated, aligned = predictions
            native_abs = relative_to_absolute(native, state[:23])
            aligned_abs = relative_to_absolute(aligned, state[:23])
            position_delta = np.linalg.norm(native_abs[:8,10:13]-aligned_abs[:8,10:13], axis=-1)
            np.savez_compressed(args.output/f"snapshot_{index:03d}.npz", state=state,
                                native=native, native_repeat=repeated, training_dtype=aligned)
            row = dict(snapshot=index, seconds=float(saved["timestamp"][index]), embeddings=captures,
                fixed_input_seed_repeat_exact=bool(np.array_equal(native, repeated)),
                fixed_input_seed_repeat_max_difference=float(abs(native-repeated).max()),
                aligned_action_max_difference=float(abs(native-aligned).max()),
                first8_right_position_delta_m=position_delta.tolist(),
                first8_right_half_gap_max_delta_m=float(abs(native[:8,19]-aligned[:8,19]).max()*.0975/2),
                image_sha256={camera:hashlib.sha256((folder/f"{camera}.png").read_bytes()).hexdigest()
                              for camera in CAMERAS})
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        model._prepare_flow_action_inputs = prepare
    report = dict(kind="real_model_internal_state_embedding_contract", model=str(args.model.resolve()),
        config=str(args.config.resolve()), episode=str(args.episode.resolve()),
        state_string_representation=bool(model.config.use_state_string_representation),
        observations=rows,
        training_dtype_embedding_matches=all(r["embeddings"][-1]["training_embedding_exactly_equal"] for r in rows),
        native_embedding_matches=all(r["embeddings"][0]["training_embedding_exactly_equal"] for r in rows),
        training_reference_stable_across_modes=all(e["training_reference_same_as_native_call"]
                                                  for r in rows for e in r["embeddings"]),
        physics_advanced=False, task_success_measured=False,
        production_policy_cast_exercised=True,
        source_sha256={str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        note="The reference calls the actual model training scatter method with identical IDs, normalized state and mask; no parameter updates.")
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["training_dtype_embedding_matches"] or not report["training_reference_stable_across_modes"]:
        raise SystemExit("Training-dtype variant did not match the actual training method")


if __name__ == "__main__":
    main()
