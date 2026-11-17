"""Wall normalization with episode-local, end-clamped relative action chunks.

The upstream statistics script concatenates all episodes before forming chunks.
Here every anchor has exactly the same end padding and relative convention as
the LeRobot/Wall loader, avoiding artificial jumps to the next demonstration.
Only the explicitly exported training dataset is read; no evaluation cases.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow
import yaml


def rows_to_matrix(rows):
    rows=np.asarray(rows,dtype=np.float64).reshape(*rows.shape[:-1],2,3)
    x=rows[...,0,:];x=x/np.linalg.norm(x,axis=-1,keepdims=True)
    y=rows[...,1,:]-(rows[...,1,:]*x).sum(-1,keepdims=True)*x
    y=y/np.linalg.norm(y,axis=-1,keepdims=True)
    return np.stack((x,y,np.cross(x,y)),axis=-2)


def relative_chunks(states,actions,horizon):
    indices=np.minimum(np.arange(len(actions))[:,None]+np.arange(horizon),len(actions)-1)
    chunks=actions[indices].copy()
    for start in (0,10):
        chunks[...,start:start+3]-=states[:,None,start:start+3]
        target=rows_to_matrix(chunks[...,start+3:start+9])
        anchor=rows_to_matrix(states[:,start+3:start+9])
        chunks[...,start+3:start+9]=(target@anchor[:,None].swapaxes(-1,-2))[...,:2,:].reshape(len(actions),horizon,6)
    return chunks


def stats(values):
    return dict(mean=values.mean(0).tolist(),std=values.std(0).tolist(),
                q01=np.quantile(values,.01,axis=0).tolist(),q99=np.quantile(values,.99,axis=0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
    assert cfg["data"]["train_test_split"]==1.,"Export must contain training episodes only"
    root=Path(cfg["data"]["lerobot_config"]["root"])
    table=arrow.dataset(sorted((root/"data").rglob("*.parquet")),format="parquet").to_table(
        columns=["episode_index","frame_index","observation.state","action"])
    episodes=np.asarray(table["episode_index"]);frames=np.asarray(table["frame_index"])
    state=np.asarray(table["observation.state"].to_pylist(),dtype=np.float64)
    action=np.asarray(table["action"].to_pylist(),dtype=np.float64)
    assert state.shape==action.shape and state.shape[1]==23
    horizon=cfg["task"]["action_horizon_flow"];chunks=[];counts={}
    for episode in sorted(set(episodes)):
        selected=np.flatnonzero(episodes==episode);selected=selected[np.argsort(frames[selected])]
        assert np.array_equal(frames[selected],np.arange(len(selected)))
        chunks.append(relative_chunks(state[selected],action[selected],horizon).reshape(-1,23))
        counts[int(episode)]=len(selected)
    relative=np.concatenate(chunks)
    assert np.isfinite(relative).all()
    report=dict(norm_stats={"observation.state":stats(state),"action":stats(relative)},
                provenance=dict(dataset=str(root.resolve()),horizon=horizon,episode_frames=counts,
                    frames=len(state),cross_episode_pairs=0,end_padding="repeat last action",
                    rotation="first two rows; R_rel = R_abs @ R_state.T",
                    data_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((root/"data").rglob("*.parquet"))}))
    output=Path(cfg["data"]["norm_stats_path"])
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(output=str(output),episodes=len(counts),frames=len(state),finite=True,cross_episode_pairs=0)))


if __name__=="__main__":main()
