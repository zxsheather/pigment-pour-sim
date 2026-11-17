"""Check explicit opening confirmation using commands, without task feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_action_filter import GripperOpeningConfirmation


def forecast(aperture):
    command=np.arange(32*23,dtype=np.float32).reshape(32,23)/1000
    command[:,19]=aperture
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    closed,opened=forecast(.06),forecast(.70)
    others=np.arange(23)!=19
    disabled=GripperOpeningConfirmation(.7,0)
    for x in [opened,closed,opened]:
        y=disabled(x,8)
        assert y.dtype==x.dtype and np.array_equal(y,x)
    guard=GripperOpeningConfirmation(.7,2)
    assert np.array_equal(guard(opened,8),opened)
    assert np.array_equal(guard(closed,8),closed)
    blocked=guard(opened,8)
    assert np.all(blocked[:,19]==closed[-1,19])
    assert np.array_equal(blocked[:,others],opened[:,others])
    assert np.array_equal(guard(opened,8),opened),"Sustained release intent must pass"
    assert np.array_equal(guard(closed,8),closed),"Closure must pass immediately"
    guard(opened,8)
    guard(closed,8)
    guard(opened,8)
    assert guard.decisions[-1]['blocked'] and guard.pending==1,"Closure must reset pending opening"
    gradual=GripperOpeningConfirmation(.06,2)
    for value in [.10,.14,.18]:
        command=forecast(value)
        assert np.array_equal(gradual(command,8),command),"Gradual opening must remain available"
    rows=[]
    if args.episode:
        chunks=np.load(args.episode/'predicted_chunks.npy')
        states=np.load(args.episode/'model_observations.npz')['state']
        assert len(states)==len(chunks)
        off=GripperOpeningConfirmation(states[0,19],0)
        on=GripperOpeningConfirmation(states[0,19],2)
        for index,chunk in enumerate(chunks):
            command=chunk[:,:23]
            assert np.array_equal(off(command,8),command)
            filtered=on(command,8)
            assert np.array_equal(filtered[:,others],command[:,others])
            rows.append(dict(forecast_index=index,blocked=on.decisions[-1]['blocked'],
                             raw_last_prefix_target=float(command[7,19]),
                             accepted_last_prefix_target=float(filtered[7,19])))
    report=dict(passed=True,default_exact_passthrough=True,closure_immediate=True,
                two_distinct_forecasts_required=True,sustained_opening_allowed=True,
                closure_resets_pending=True,gradual_opening_allowed=True,
                non_gripper_channels_unchanged=True,physics_advanced=False,
                task_success_measured=False,recorded_forecast_check=rows,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__),Path(__file__).with_name('robot_action_filter.py')]})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['recorded_forecast_check','source_sha256']}))


if __name__=='__main__':main()
