"""Verify action-expert updates and unchanged frozen weights after SFT.

Compares each checkpoint tensor to the base checkpoint after the runtime dtype
cast. Task-specific normalizer buffers are excluded. No GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch

ACTION_KEYS=("action_preprocessor","moe.experts.1","qkv_proj_experts.1","o_proj_experts.1",
             "input_layernorms.1","post_attention_layernorms.1","model.norms.1")


def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    base=a.base/"model.safetensors";checkpoint=a.checkpoint/"model.safetensors"
    changed=[];frozen_changed=[];missing=[];frozen_count=0;action_count=0
    with safe_open(str(base),framework="pt",device="cpu") as source, \
            safe_open(str(checkpoint),framework="pt",device="cpu") as trained:
        for key in trained.keys():
            if "normalizer" in key:continue
            if key not in source.keys():missing.append(key);continue
            target=trained.get_tensor(key);original=source.get_tensor(key).to(target.dtype)
            assert target.shape==original.shape and torch.isfinite(target).all(),key
            action=any(part in key for part in ACTION_KEYS)
            if action:action_count+=1
            else:frozen_count+=1
            if not torch.equal(original,target):
                row=dict(name=key,max_abs_change=float((original.float()-target.float()).abs().max()))
                (changed if action else frozen_changed).append(row)
    report=dict(base_sha256=file_hash(base),checkpoint_sha256=file_hash(checkpoint),
                checkpoint=str(a.checkpoint.resolve()),action_tensors=action_count,
                frozen_tensors=frozen_count,changed_action_tensors=changed,
                changed_frozen_tensors=frozen_changed,missing_base_keys=missing,
                comparison="Exact equality after casting the original tensor to the checkpoint dtype",
                passed=bool(changed) and not frozen_changed and not missing)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(passed=report["passed"],changed_action_tensors=len(changed),
                         changed_frozen_tensors=len(frozen_changed),frozen_tensors=frozen_count)))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
