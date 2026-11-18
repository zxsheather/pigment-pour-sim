"""Select Wall's language prompt from measured gripper and base progress."""
import numpy as np

from cylinder_joint_actions import APERTURE_SCALE_M, JOINT_SERVO_STATE_WIDTH, require


PROGRESS_PROTOCOL = 'gripper_aperture_and_base_retreat_v1'
PROGRESS_PROMPTS = (
    'Approach the measuring cylinder on the tray and grip it securely between the parallel fingers. Keep it upright.',
    'Keep the measuring cylinder securely held and upright. Lift it clear of the tray, retract the arm and move back for the camera reading.',
    'Keep holding the measuring cylinder. Present its scale close to the head camera, align the liquid surface with the image center and hold it steady for reading.',
)


class CylinderProgressPrompts:
    def __init__(self, initial_base):
        base = np.asarray(initial_base, dtype=np.float64)
        require(base.shape == (3,) and np.isfinite(base).all(), 'Expected measured initial x, y and yaw')
        self.initial_xy = base[:2].copy()
        self.initial_forward = np.array([np.cos(base[2]), np.sin(base[2])])
        self.index = 0
        self.previous_gap = None
        self.consecutive_stalled_ticks = 0

    @property
    def instruction(self):
        return PROGRESS_PROMPTS[self.index]

    def observe(self, state):
        """Update once per control tick from state available before the next action."""
        observed = np.asarray(state)
        require(observed.shape == (JOINT_SERVO_STATE_WIDTH,) and np.isfinite(observed).all(),
                'Progress prompts require the measured 26D joint-servo state')
        gap = float(observed[6]) * APERTURE_SCALE_M
        tracking_error = float(observed[24]) * APERTURE_SCALE_M
        stable = self.previous_gap is None or abs(gap - self.previous_gap) <= .002
        self.previous_gap = gap
        if self.index == 0:
            stalled = stable and .012 <= gap <= .033 and gap + tracking_error <= .025 and tracking_error <= -.0009
            self.consecutive_stalled_ticks = self.consecutive_stalled_ticks + 1 if stalled else 0
            if self.consecutive_stalled_ticks >= 5:
                self.index = 1
        if self.index == 1:
            retreat = (self.initial_xy - observed[15:17]) @ self.initial_forward
            if retreat >= .205 and abs(observed[7]) <= .03:
                self.index = 2
        return self.index
