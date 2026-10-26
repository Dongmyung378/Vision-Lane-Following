"""Selection is explicit; the existing baseline remains the default."""
import math
import unittest

from carla_tools.traffic_experimental import HypothesisTrafficBridge, make_evidence_bridge
from carla_tools.traffic_policy import TrafficEvidenceBridge


def calibration():
    c = math.cos(math.radians(12.))
    s = math.sin(math.radians(12.))
    return {'width':640, 'height':360,
        'intrinsics':{'fx':320., 'fy':320., 'cx':320., 'cy':180.},
        'camera_to_actor_matrix':[[c,0.,s,1.5], [0.,1.,0.,0.], [-s,0.,c,2.4], [0.,0.,0.,1.]]}


class ExplicitSelectionTests(unittest.TestCase):
    def test_default_remains_existing_baseline(self):
        self.assertIs(type(make_evidence_bridge(nominal_calibration=calibration())), TrafficEvidenceBridge)

    def test_hypothesis_requires_explicit_selection(self):
        self.assertIs(type(make_evidence_bridge(nominal_calibration=calibration(), strategy='hypothesis')),
            HypothesisTrafficBridge)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            make_evidence_bridge(nominal_calibration=calibration(), strategy='automatic')


if __name__ == '__main__': unittest.main()
