"""Optional task-independent low-pass filter for predicted robot commands.

Uses only robot-frame Cartesian commands and an initial robot state. Gripper
commands pass through so closure preload is preserved. No object/task feedback.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from cylinder_actions import rotation_from_rows


def average_model_action_samples(samples):
    """Average independent model draws anchored at the same observation.

    Unlike a temporal ensemble, these draws all use current feedback. Keep the
    one-sample path exact; project averaged 6D rotations back onto SO(3).
    """
    samples=np.asarray(samples)
    if samples.ndim!=3 or samples.shape[0]<1 or samples.shape[-1]<23 or not np.isfinite(samples).all():
        raise ValueError("Invalid model action samples")
    if len(samples)==1:return samples[0].copy()
    result=samples.mean(0,dtype=np.float64)
    for start in (3,13):
        for action in result:
            action[start:start+6]=rotation_from_rows(action[start:start+6])[:2].reshape(6)
    return result


class TemporalActionEnsemble:
    """Average overlapping model forecasts for the same future control tick.

    These are robot-frame absolute targets, already decoded at each observation
    anchor. The newest gripper command is retained to preserve closure preload.
    """
    def __init__(self, max_predictions=0):
        if not 0<=max_predictions<=8:raise ValueError("Invalid temporal ensemble size")
        self.max_predictions=max_predictions;self.forecasts=[]

    def add(self, start_tick, actions):
        if not self.max_predictions:return
        self.forecasts=[(start,chunk) for start,chunk in self.forecasts if start+len(chunk)>start_tick]
        self.forecasts.append((start_tick,np.asarray(actions,dtype=float).copy()))
        self.forecasts=self.forecasts[-self.max_predictions:]

    def command(self, tick, latest):
        if not self.max_predictions:return np.asarray(latest).copy()
        candidates=[chunk[tick-start] for start,chunk in self.forecasts if 0<=tick-start<len(chunk)]
        if not candidates:raise ValueError("No model forecast covers the current timestep")
        average=np.mean(candidates,axis=0);output=np.asarray(latest,dtype=float).copy()
        output[10:13]=average[10:13]
        output[13:19]=rotation_from_rows(average[13:19])[:2].reshape(6)
        output[20:23]=average[20:23]
        return output


class GripperOpeningConfirmation:
    """Optionally confirm abrupt opening across distinct policy forecasts.

    Closure passes immediately. A >5mm per-finger increase over the last
    accepted command needs repeated forecasts; gradual opening still passes.
    Only predicted commands and the initial robot aperture enter this helper.
    This is an explicit execution ablation, not a learned-policy improvement.
    """
    def __init__(self, initial_aperture, confirmations=0, mode="prefix-median"):
        if (confirmations not in (0, 2, 3, 4) or not np.isfinite(initial_aperture)
                or mode not in ("prefix-median", "aligned-forecasts")):
            raise ValueError("Invalid gripper opening confirmation settings")
        self.confirmations=confirmations
        self.mode=mode
        self.last=float(initial_aperture)
        self.pending=0
        self.decisions=[]
        self.normalized_margin=.005/(.0975/2)
        self.forecast_tick=0
        self.forecasts=[]

    def __call__(self, actions, execute_steps):
        actions=np.asarray(actions)
        if (actions.ndim!=2 or actions.shape[1]!=23 or
                not 1<=execute_steps<=len(actions) or not np.isfinite(actions).all()):
            raise ValueError("Invalid action forecast for gripper confirmation")
        output=actions.copy()
        if not self.confirmations:
            return output
        if self.mode=="aligned-forecasts":
            return self._aligned(actions,execute_steps)
        opening=bool(np.median(actions[:execute_steps,19])>self.last+self.normalized_margin)
        self.pending=self.pending+1 if opening else 0
        blocked=opening and self.pending<self.confirmations
        previous=self.last
        if blocked:
            output[:,19]=np.minimum(output[:,19],self.last)
        self.last=float(output[execute_steps-1,19])
        self.decisions.append(dict(forecast_index=len(self.decisions),
            previous_normalized_target=previous,
            proposed_prefix_median=float(np.median(actions[:execute_steps,19])),
            pending_forecasts=self.pending,blocked=bool(blocked),
            accepted_last_prefix_target=self.last))
        return output

    def _aligned(self, actions, execute_steps):
        """Confirm each abrupt opening at the SAME future control tick.

        Forecasts overlap in time. A late-prefix spike cannot gain approval
        from a different tick, a prefix median, or repeated frames in one call.
        Closure and gradual opening remain immediately available.
        """
        if execute_steps*self.confirmations>len(actions):
            raise ValueError("Aligned gripper confirmation requires overlapping forecasts")
        start=self.forecast_tick
        forecasts=self.forecasts+[(start,actions[:,19].copy())]
        output=actions.copy();blocked=[];agreement=[];previous=self.last
        for offset in range(execute_steps):
            proposed=float(actions[offset,19]);tick=start+offset;count=0
            if proposed>self.last+self.normalized_margin:
                for begin,apertures in reversed(forecasts):
                    position=tick-begin
                    if (not 0<=position<len(apertures) or
                            apertures[position]<=self.last+self.normalized_margin):
                        break
                    count+=1
                if count<self.confirmations:
                    output[offset,19]=self.last;blocked.append(offset)
            self.last=float(output[offset,19]);agreement.append(count)
        self.forecasts=forecasts[-(self.confirmations-1):]
        self.forecast_tick+=execute_steps
        self.decisions.append(dict(forecast_index=len(self.decisions),
            mode=self.mode,start_tick=start,previous_normalized_target=previous,
            blocked=bool(blocked),blocked_prefix_steps=blocked,
            same_tick_opening_agreement=agreement,
            accepted_last_prefix_target=self.last))
        return output


def limit_pose_step(command, robot_state, translation=0., rotation=0.):
    """Bound a Cartesian command relative to measured robot pose before IK.

    A zero bound disables that limit. Base/gripper outputs pass through.
    This is a generic robot servo envelope, independent of task/object state.
    """
    output=np.asarray(command,dtype=float).copy()
    state=np.asarray(robot_state,dtype=float)
    if output.shape!=(23,) or state.shape!=(23,) or translation<0 or rotation<0:
        raise ValueError("Invalid Cartesian servo envelope")
    if translation:
        delta=output[10:13]-state[10:13];size=np.linalg.norm(delta)
        if size>translation:output[10:13]=state[10:13]+delta*(translation/size)
    if rotation:
        before=rotation_from_rows(state[13:19]);target=rotation_from_rows(output[13:19])
        delta=Rotation.from_matrix(target@before.T).as_rotvec();size=np.linalg.norm(delta)
        if size>rotation:
            target=Rotation.from_rotvec(delta*(rotation/size)).as_matrix()@before
            output[13:19]=target[:2].reshape(6)
    return output


class RobotActionFilter:
    def __init__(self, initial_state, seconds=0., dt=.05):
        if not np.isfinite(seconds) or seconds < 0 or dt <= 0:
            raise ValueError("Invalid robot filter time constant")
        self.seconds = float(seconds)
        self.alpha = 1. if seconds == 0 else float(-np.expm1(-dt/seconds))
        self.last = np.asarray(initial_state, dtype=float).copy()
        assert self.last.shape == (23,)

    def __call__(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (23,) or not np.isfinite(command).all():
            raise ValueError("Invalid robot filter command")
        if self.seconds == 0:
            return command.copy()
        output = command.copy()
        output[10:13] = self.last[10:13] + self.alpha*(command[10:13]-self.last[10:13])
        before = rotation_from_rows(self.last[13:19])
        target = rotation_from_rows(command[13:19])
        delta = Rotation.from_matrix(target @ before.T).as_rotvec()
        rotation = Rotation.from_rotvec(self.alpha*delta).as_matrix() @ before
        output[13:19] = rotation[:2].reshape(6)
        output[20:23] = self.last[20:23] + self.alpha*(command[20:23]-self.last[20:23])
        self.last = output.copy()
        return output
