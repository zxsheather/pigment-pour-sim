"""Initialize the official Wall-X continuous proprioception projection option.

Preserves every parent checkpoint tensor and adds one seeded trainable linear
projection. The original model directory is never modified. This is a declared
input-representation experiment, not the unmodified pretrained architecture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=904)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    config=json.loads((a.model/"config.json").read_text())
    config["use_state_string_representation"]=False
    (a.output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    for name in ("preprocessor_config.json","tokenizer_config.json","tokenizer.json","tokenizer.model",
                 "special_tokens_map.json","vocab.json","merges.txt"):
        if (a.model/name).exists():shutil.copyfile(a.model/name,a.output/name)
    dim=int(config.get("state_hidden_size",2048));width=26*(2 if config.get("proj_with_mask",True) else 1)
    key="action_preprocessor.propri_proj.weight"
    generator=torch.Generator().manual_seed(a.seed)
    weight=torch.randn(dim,width,generator=generator)*float(config.get("initializer_range",.02))
    with safe_open(str(a.model/"model.safetensors"),framework="pt",device="cpu") as source:
        assert key not in source.keys(),"Expected a parent using state strings"
        tensors={name:source.get_tensor(name) for name in source.keys()};tensors[key]=weight
        save_file(tensors,str(a.output/"model.safetensors"))
    report=dict(parent=str(a.model.resolve()),parent_sha256=digest(a.model/"model.safetensors"),
                initialized_sha256=digest(a.output/"model.safetensors"),new_parameter=key,
                shape=list(weight.shape),seed=a.seed,
                projection_tensor_sha256=hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                initialization="Seeded Gaussian with config.initializer_range; every parent tensor preserved",
                observation="Same 23 robot-state values and three masked zero padding slots; no object truth or phase")
    (a.output/"state_projection_initialization.json").write_text(json.dumps(report,indent=2)+"\n")
    torch.save(weight,a.output/"initial_state_projection.pt")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
