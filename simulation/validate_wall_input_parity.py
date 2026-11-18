"""Compare actual training and inference preprocessing on identical RGB/state.

No model weights or robot actions are evaluated. The uninitialized policy
object below supplies only the real inference preprocessing methods.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from run_wall_cylinder import CylinderWallPolicy
from run_wall_carry import build_data_config,normalize_train_config_for_inference,register_data_backend
from cylinder_joint_actions import JOINT_ACTION_LAYOUT, JOINT_ACTION_WIDTH, JOINT_STATE_WIDTH, require
from wall_x._vendor.harrix.utils.normalizer import build_normalizers
from wall_x.config.loader import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.model.qact.qwen2_5.adapter import Qwen2_5Adapter
from wall_x.trainer.adapters.base_adapter import load_trainer_data_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--normalizer-dir",type=Path,help="Directory containing the serving norm_stats.json")
    p.add_argument("--norm-key",default="parity_probe")
    p.add_argument("--episode",type=int,action="append",default=[],
                   help="Check the episode's first, middle and last training-subset frames.")
    p.add_argument("--samples",type=int,default=3);a=p.parse_args()
    if a.samples<1:p.error("--samples must be positive")
    if a.output.exists():raise FileExistsError(a.output)
    dist.init_process_group("gloo")
    try:
        cfg=load_config(str(a.config));adapter=Qwen2_5Adapter(cfg=cfg,logger=logging.getLogger("input_parity"))
        dof_config=dict(cfg.task.dof_config)
        joint=dof_config==JOINT_ACTION_LAYOUT
        action_width=JOINT_ACTION_WIDTH if joint else 23
        act,state,*_=load_lerobot_normalizers(cfg);adapter.normalizer_action,adapter.normalizer_propri=act,state
        processors=adapter.load_processor(None);model_config=adapter.build_model_config()
        dataset,loader,_=adapter.load_dataset(load_trainer_data_config(cfg),processors["processor"],rank=0,world_size=1,
            normalizer_action=act,normalizer_propri=state,model_config=model_config,tokenizer_mixin=processors["tokenizer_mixin"])
        policy=object.__new__(CylinderWallPolicy)
        policy._train_config=normalize_train_config_for_inference(yaml.safe_load(a.config.read_text()),str(a.config.resolve()))
        register_data_backend(policy._train_config)
        policy._normalizer_action,policy._normalizer_propri,policy._norm_key=build_normalizers(
            str(a.normalizer_dir or a.config.parent),policy._train_config,a.norm_key)
        policy._data_config=build_data_config(str(a.config),policy._train_config)
        policy._processor=processors["processor"];policy._device="cpu";policy._robot_id="10000"
        mapping=cfg.data.key_mappings["camera"];policy._cam_names=list(mapping.values())
        policy._action_horizon=cfg.task.action_horizon_flow
        fields=["input_ids","attention_mask","pixel_values","image_grid_thw","proprioception","agent_pos_mask","dof_mask","moe_token_types"]
        inner=dataset._inner
        indices=np.asarray(inner.train_dataset.indices)
        raw_dataset=inner.hf_dataset.hf_dataset
        if a.episode:
            episodes=np.asarray(raw_dataset["episode_index"])[indices]
            frames=np.asarray(raw_dataset["frame_index"])[indices]
            selected=[]
            for episode in a.episode:
                subset=np.flatnonzero(episodes==episode)
                require(len(subset)>0,f"Episode {episode} has no selected training frames")
                subset=subset[np.argsort(frames[subset])]
                selected.extend(np.unique(subset[[0,len(subset)//2,-1]]).tolist())
        else:
            selected=list(range(min(a.samples,len(indices))))
        require(bool(selected),"Expected at least one training-subset record")
        rows=[]
        for index in selected:
            raw=inner.hf_dataset[int(indices[index])]
            processed=inner[index];train=loader.collate_fn([processed])
            values=raw["observation.state"].numpy();width=len(values)
            require(not joint or width==JOINT_STATE_WIDTH,"Wrong joint observation width")
            proprio=np.zeros((1,1,26),np.float32);proprio[...,:width]=values
            mask=np.zeros_like(proprio);mask[...,:width]=1
            dof=np.zeros((1,policy._action_horizon,26),np.float32);dof[...,:action_width]=1
            images={name:(raw[key].permute(1,2,0)*255).to(torch.uint8).numpy() for key,name in mapping.items()}
            encoded=dict(proprioception=proprio,agent_pos_mask=mask,dof_mask=dof,**images)
            prefix,postfix=policy._get_flow_prompt(raw["task"])
            infer=policy._construct_model_input([encoded],[prefix],[postfix])
            infer["proprioception"][...,width:]=0
            comparison={}
            for key in fields:
                x,y=train[key].cpu(),infer[key].cpu()
                same_shape=x.shape==y.shape
                difference=float((x.float()-y.float()).abs().max()) if same_shape and x.numel() else None
                comparison[key]=dict(train_shape=list(x.shape),inference_shape=list(y.shape),max_absolute_difference=difference,
                                     equal=bool(same_shape and torch.allclose(x.float(),y.float(),rtol=0,atol=1e-6)))
            rows.append(dict(raw_frame=int(raw["index"]),episode=int(raw["episode_index"]),
                             episode_frame=int(raw["frame_index"]),state_width=width,fields=comparison))
        report=dict(kind="training_inference_input_contract_no_model_execution",samples=rows,
                    passed=all(field["equal"] for row in rows for field in row["fields"].values()),
                    action_space="joint" if joint else "cartesian",action_width=action_width,
                    actual_loader_training_subset=True,distributed_backend="gloo",
                    note="Identical decoded RGB and measured robot state; no model weights, actions or task score")
        a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
        if not report["passed"]:raise SystemExit(1)
    finally:dist.destroy_process_group()


if __name__=="__main__":main()
