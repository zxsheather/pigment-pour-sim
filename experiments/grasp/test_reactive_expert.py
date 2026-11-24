"""Regression checks for observable pregrasp progression and reopening."""
import unittest

import numpy as np

from reactive_expert import alignment_target


class ReactiveApproachTest(unittest.TestCase):
    def test_small_tracking_error_does_not_create_a_standoff_equilibrium(self):
        bottle=np.array([.15447,.49551,.84450])
        tool=np.array([.14088,.49100,.85209])
        target=alignment_target(bottle,tool,4.46,.001)
        self.assertGreater(target[0],tool[0]+.01)
        np.testing.assert_allclose(target,bottle+[0.,0.,.005])

    def test_closed_jaws_retreat_before_advancing_into_the_bottle(self):
        bottle=np.array([.14,.49,.843])
        target=alignment_target(bottle,bottle+[0.,0.,.005],2.2,0.)
        self.assertLess(target[0],bottle[0]-.09)
        self.assertGreater(target[2],bottle[2]+.02)


if __name__=="__main__":unittest.main()
