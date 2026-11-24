"""Low-memory loader for the public Wall-OSS runtime, without changing weights.

Use native SDPA, retain the checkpoint's original vision MLP width, and load
all parameters with assign=True. The official prompt, normalizers, action
generation and robot serialization are reused unchanged.
"""
from pathlib import Path
import copy
import gc

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file


def build_policy(config):
    from runtime_compat import configure_runtime
    configure_runtime()
    from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
    from wall_x._vendor.harrix.serving.policy import wall_x_policy as policy_module

    class LowMemoryWrapper(WallxModelWrapper):
        def _load_model(self):
            from wall_x.trainer.adapters import resolve_adapter
            adapter = resolve_adapter(self.config.train_config["model_type"])
            self.ModelClass = adapter.inference_model_class()
            weight_path = str(Path(self.config.checkpoint_path)/"model.safetensors")
            with safe_open(weight_path, framework="pt", device="cpu") as weights:
                vocab = weights.get_slice("model.embed_tokens.weight").get_shape()[0]
            self.config.model_config.vocab_size = vocab
            with init_empty_weights():
                self.model = self.ModelClass(self.config.model_config, self.processor, self.tokenizer_mixin)
            self.model_type = self.config.train_config["model_type"]

        def load_ckpt(self, checkpoint_path=None):
            checkpoint_path = checkpoint_path or self.config.checkpoint_path
            state = load_file(str(Path(checkpoint_path)/"model.safetensors"), device="cpu")
            if not self.ModelClass.is_fused(state):
                state = self.ModelClass.convert_to_fused(state)
            # Current-run files own normalization, including for an untrained
            # baseline using the same demonstration statistics as adaptation.
            def normalization_key(key):
                return "normalizer_action." in key or "normalizer_propri." in key
            state = {k: v for k, v in state.items() if not normalization_key(k)}
            overlay = Path(checkpoint_path)/"action_expert.safetensors"
            overlay_count = 0
            if overlay.exists():
                adapted = load_file(str(overlay), device="cpu")
                for key, value in adapted.items():
                    if key not in state or value.shape != state[key].shape:
                        raise RuntimeError(f"Incompatible action expert weight: {key}")
                state.update(adapted)
                overlay_count = len(adapted)
            self.model.set_normalizer(copy.deepcopy(self.normalizer_action), copy.deepcopy(self.normalizer_propri))
            msg = self.model.load_state_dict(state, strict=False, assign=True)
            missing = [k for k in msg.missing_keys if k != "lm_head.weight" and not normalization_key(k)]
            if missing or msg.unexpected_keys:
                raise RuntimeError(f"Incomplete checkpoint: missing={missing[:12]}, unexpected={msg.unexpected_keys[:12]}")
            self.model.tie_weights()
            meta = [k for k, p in self.model.named_parameters() if p.is_meta]
            if meta:
                raise RuntimeError(f"Unloaded parameters: {meta}")
            self.model.set_normalizer(copy.deepcopy(self.normalizer_action), copy.deepcopy(self.normalizer_propri))
            self.model.eval()
            self.model.to(self.config.model_device)
            self.model.to_bfloat16_for_selected_params()
            self.load_audit = {"missing_non_tied_keys": missing, "unexpected_keys": list(msg.unexpected_keys), "unloaded_meta_parameters": meta, "vision_intermediate_size": self.config.model_config.vision_config.intermediate_size, "vision_alignment": 1, "attention": "native PyTorch SDPA", "checkpoint_weights_changed": overlay_count > 0, "action_expert_overlay_tensors": overlay_count, "normalization_from_current_run_files": True}
            del state
            gc.collect()
            print("Checkpoint loaded and audited:", self.load_audit, flush=True)

    policy_module.WallxModelWrapper = LowMemoryWrapper
    return policy_module.WallXPolicy(config, image_passing_mode="numpy", serialize_actions=True)
