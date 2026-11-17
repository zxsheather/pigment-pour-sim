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
import torch.nn.functional as functional

ACTION_KEYS=("action_preprocessor","moe.experts.1","qkv_proj_experts.1","o_proj_experts.1",
             "input_layernorms.1","post_attention_layernorms.1","model.norms.1")


def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def align_runtime_padding(key,original,target):
    """Reproduce the pinned vision MLP's documented zero padding to 128.

    Compare the complete padded tensor, including zeros; never discard a
    potentially changed row/column. Other shape changes remain errors.
    """
    if original.shape==target.shape:return original,False
    assert key.startswith("visual.blocks.") and ".mlp." in key,key
    if key.endswith("down_proj.weight"):
        pad=target.shape[1]-original.shape[1]
        assert target.shape[0]==original.shape[0] and 0<pad<128 and target.shape[1]%128==0,key
        return functional.pad(original,(0,pad)),True
    assert key.endswith(("gate_up_proj.weight","gate_up_proj.bias")),key
    orig=original.shape[0]//2;padded=target.shape[0]//2;pad=padded-orig
    assert 0<pad<128 and padded%128==0 and original.shape[1:]==target.shape[1:],key
    gate,up=original.split(orig,dim=0)
    padding=(0,0,0,pad) if original.ndim==2 else (0,pad)
    return torch.cat((functional.pad(gate,padding),functional.pad(up,padding)),dim=0),True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    base=a.base/"model.safetensors";checkpoint=a.checkpoint/"model.safetensors"
    changed=[];frozen_changed=[];missing=[];frozen_count=0;action_count=0;padding=[];aliases={}
    tied=json.loads((a.base/"config.json").read_text()).get("tie_word_embeddings",False)
    with safe_open(str(base),framework="pt",device="cpu") as source, \
            safe_open(str(checkpoint),framework="pt",device="cpu") as trained:
        for key in trained.keys():
            if "normalizer" in key:continue
            source_key=key
            if key not in source.keys():
                if key=="lm_head.weight" and tied:
                    source_key="model.embed_tokens.weight";aliases[key]=source_key
                else:missing.append(key);continue
            target=trained.get_tensor(key);original=source.get_tensor(source_key).to(target.dtype)
            original,aligned=align_runtime_padding(key,original,target)
            if aligned:padding.append(key)
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
                runtime_zero_padded_tensors=padding,
                tied_weight_aliases=aliases,
                comparison="Exact equality after runtime dtype cast and documented vision-MLP zero padding (all zeros included)",
                passed=bool(changed) and not frozen_changed and not missing)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(passed=report["passed"],changed_action_tensors=len(changed),
                         changed_frozen_tensors=len(frozen_changed),frozen_tensors=frozen_count)))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
