"""Use base weights with task normalization for a comparable zero-shot test.

The original checkpoint remains untouched. Symlinks keep its weights/tokenizer;
task statistics and config are copied explicitly, excluding old normalizer PKLs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for source in a.model.resolve().iterdir():
        if source.is_file() and (source.suffix in (".json",".safetensors",".model")
                                 or source.name in ("merges.txt",)):
            if source.name!="norm_stats.json":(a.output/source.name).symlink_to(source)
    for source,name in ((a.run/"norm_stats.json","norm_stats.json"),(a.run/"train.yml","config.yml")):
        shutil.copyfile(source,a.output/name)
    manifest=dict(model=str(a.model.resolve()),run=str(a.run.resolve()),
                  weight_sha256=hashlib.sha256((a.model/"model.safetensors").read_bytes()).hexdigest(),
                  normalization_sha256=hashlib.sha256((a.output/"norm_stats.json").read_bytes()).hexdigest(),
                  note="Unmodified base weights, task-trained normalization/layout. Not pretrained default statistics.")
    (a.output/"evaluation_view.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))


if __name__=="__main__":main()
