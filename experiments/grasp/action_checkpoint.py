"""Export action-expert weights beside an immutable base checkpoint.

These are evaluation snapshots, not full optimizer-resume checkpoints.
Baseline and adapted folders share the same preprocessing and normalizers.
"""
from pathlib import Path
import json
import os
import shutil

import torch
from safetensors.torch import save_file
import yaml


def _link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def export_action_checkpoint(trainer, base, output):
    base, output = Path(base), Path(output)
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    weights = {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}
    if not weights:
        raise RuntimeError("No trainable action-expert weights to export")
    description = {"method": "action expert fine-tuning, not LoRA", "global_step": trainer.global_step,
                   "parameter_tensors": len(weights), "trainable_parameters": sum(t.numel() for t in weights.values()),
                   "base_revision": "b7dc23a82dbf70a65859c9fdd596703234597028", "optimizer_resume_supported": False}
    initialization = getattr(trainer, "action_expert_initialization", None)
    if initialization:
        description["action_expert_initialization"] = initialization
        description["additional_updates_with_fresh_optimizer"] = trainer.global_step
        description["logical_total_weight_updates"] = initialization["source_completed_updates"] + trainer.global_step
    for name in ["baseline", "adapted"]:
        folder = output/name
        folder.mkdir(parents=True, exist_ok=False)
        _link(base/"model.safetensors", folder/"model.safetensors")
        for asset in base.iterdir():
            if asset.is_file() and asset.name != "config.json" and asset.suffix in [".json", ".txt", ".model"]:
                _link(asset, folder/asset.name)
        shutil.copyfile(trainer.cfg.model.config_path, folder/"config.json")
        runtime = yaml.safe_load((base/"config.yml").read_text())
        runtime["processor_path"] = str(folder.resolve())
        runtime["dof_config"] = dict(trainer.cfg.task.dof_config)
        runtime["agent_pos_config"] = dict(trainer.cfg.task.agent_pos_config)
        runtime["state_bins"] = trainer.cfg._raw_data["state_bins"]
        runtime.setdefault("data", {}).update(resolution=trainer.cfg.data.resolution,
                                               state_bins=runtime["state_bins"], use_state_string_representation=True,
                                               max_length=trainer.cfg._raw_data["max_length"])
        (folder/"config.yml").write_text(yaml.safe_dump(runtime, sort_keys=False))
        torch.save(trainer.normalizer_action.state_dict(), folder/"normalizer_action.pth")
        torch.save(trainer.normalizer_propri.state_dict(), folder/"normalizer_propri.pth")
        if name == "adapted":
            save_file(weights, str(folder/"action_expert.safetensors"))
        folder_description = {**description, "adapted": name == "adapted"}
        if initialization and name == "baseline":
            folder_description.update(logical_total_weight_updates=0,
                baseline_is_public_base_not_warm_start=True)
        (folder/"adaptation.json").write_text(json.dumps(folder_description, indent=2))
    return description
