"""Optional language cue for approaching the RGB-localized cylinder."""

import numpy as np

from cylinder_joint_actions import JOINT_SERVO_STATE_WIDTH, require
from cylinder_progress_prompts import PROGRESS_PROMPTS
from cylinder_tray_vision import WRIST_TARGET_CUE_INDICES, WRIST_TARGET_NEAR_SCALE_M


FAR_APPROACH_PROMPT_PROTOCOL = 'observed_rgb_wrist_range_18cm_v1'
FAR_APPROACH_DISTANCE_M = .18
FAR_APPROACH_PROMPT = (
    'Move the right wrist toward the measuring cylinder on the tray with the '
    'parallel fingers open. Align the wrist for grasping.'
)
RANGE_CONDITIONED_PROMPTS = (*PROGRESS_PROMPTS, FAR_APPROACH_PROMPT)


def range_conditioned_prompt_indices(state, phase):
    """Preserve the phase index unless the approach has a valid far RGB cue."""
    phase = np.asarray(phase)
    state = np.asarray(state, dtype=np.float64)
    require(state.shape == (*phase.shape, JOINT_SERVO_STATE_WIDTH) and np.isfinite(state).all(),
            'Range-conditioned prompts require the measured joint servo state')
    require(np.issubdtype(phase.dtype, np.integer)
            and np.all((phase >= 0) & (phase < len(PROGRESS_PROMPTS))), 'Invalid progress phase')
    offset = WRIST_TARGET_NEAR_SCALE_M * np.sinh(state[..., WRIST_TARGET_CUE_INDICES[:-1]])
    observed = state[..., WRIST_TARGET_CUE_INDICES[-1]] > .5
    farther = np.linalg.norm(offset, axis=-1) > FAR_APPROACH_DISTANCE_M
    return np.where((phase == 0) & observed & farther, len(PROGRESS_PROMPTS), phase)
