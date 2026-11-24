"""Optional language cues for approaching the RGB-localized cylinder."""

import numpy as np

from cylinder_joint_actions import APERTURE_SCALE_M, JOINT_SERVO_STATE_WIDTH, require
from cylinder_progress_prompts import PROGRESS_PROMPTS
from cylinder_tray_vision import WRIST_TARGET_CUE_INDICES, WRIST_TARGET_NEAR_SCALE_M


FAR_APPROACH_PROMPT_PROTOCOL = 'observed_rgb_wrist_range_18cm_v1'
FAR_APPROACH_DISTANCE_M = .18
FAR_APPROACH_PROMPT = (
    'Move the right wrist toward the measuring cylinder on the tray with the '
    'parallel fingers open. Align the wrist for grasping.'
)
RANGE_CONDITIONED_PROMPTS = (*PROGRESS_PROMPTS, FAR_APPROACH_PROMPT)
CLEARANCE_APPROACH_PROMPT_PROTOCOL = 'observed_rgb_wrist_clearance_5_to_12cm_v1'
CLEARANCE_APPROACH_RANGE_M = (.05, .12)
CLEARANCE_MAX_APERTURE_M = .025
CLEARANCE_APPROACH_PROMPT = (
    'Back the right wrist away slightly to clear the measuring cylinder, '
    'then open the parallel fingers and align the wrist before closing.'
)
RANGE_CONDITIONED_PROMPT_PROTOCOLS = (FAR_APPROACH_PROMPT_PROTOCOL, CLEARANCE_APPROACH_PROMPT_PROTOCOL)


def range_conditioned_prompts(protocol):
    require(protocol in RANGE_CONDITIONED_PROMPT_PROTOCOLS, 'Invalid range-conditioned prompt protocol')
    if protocol == FAR_APPROACH_PROMPT_PROTOCOL:
        return RANGE_CONDITIONED_PROMPTS
    return (*PROGRESS_PROMPTS, CLEARANCE_APPROACH_PROMPT)


def range_conditioned_prompt_indices(state, phase, protocol=FAR_APPROACH_PROMPT_PROTOCOL):
    """Select the approach prompt from measured wrist range and finger opening."""
    require(protocol in RANGE_CONDITIONED_PROMPT_PROTOCOLS, 'Invalid range-conditioned prompt protocol')
    phase = np.asarray(phase)
    state = np.asarray(state, dtype=np.float64)
    require(state.shape == (*phase.shape, JOINT_SERVO_STATE_WIDTH) and np.isfinite(state).all(),
            'Range-conditioned prompts require the measured joint servo state')
    require(np.issubdtype(phase.dtype, np.integer)
            and np.all((phase >= 0) & (phase < len(PROGRESS_PROMPTS))), 'Invalid progress phase')
    offset = WRIST_TARGET_NEAR_SCALE_M * np.sinh(state[..., WRIST_TARGET_CUE_INDICES[:-1]])
    observed = state[..., WRIST_TARGET_CUE_INDICES[-1]] > .5
    distance = np.linalg.norm(offset, axis=-1)
    if protocol == FAR_APPROACH_PROMPT_PROTOCOL:
        selected = distance > FAR_APPROACH_DISTANCE_M
    else:
        lower, upper = CLEARANCE_APPROACH_RANGE_M
        opening = state[..., 6] * APERTURE_SCALE_M
        selected = (distance >= lower) & (distance <= upper) & (opening <= CLEARANCE_MAX_APERTURE_M)
    return np.where((phase == 0) & observed & selected, len(PROGRESS_PROMPTS), phase)
