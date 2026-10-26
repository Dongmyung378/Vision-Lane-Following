"""Explicitly selected RGB hypothesis/confirmation bridge for simulation."""
from .traffic_hypothesis import HypothesisMemoryBridge
from .traffic_policy import TrafficEvidenceBridge
from .traffic_signal_confirmation import SignalConfirmationAdapter


class HypothesisTrafficBridge:
    """Compose optional hypothesis memory and signal confirmation.

    The caller owns compatibility with the fixed nominal camera, 20 Hz update,
    3 m/s control envelope and unchanged detector/supervisor contracts. This
    simulation experiment does not supply an occlusion or collision guarantee.
    """
    def __init__(self, *, nominal_calibration, front_bumper_x_m=2.4251085, ego_half_width_m=1.):
        self.range = HypothesisMemoryBridge(nominal_calibration=nominal_calibration,
            front_bumper_x_m=front_bumper_x_m, ego_half_width_m=ego_half_width_m)
        self.signal = SignalConfirmationAdapter(nominal_calibration=nominal_calibration)
        self.reset()

    def reset(self):
        self.range.reset()
        self.signal.reset()

    def update(self, raw, prediction, observation):
        ranged, range_notes = self.range.update(raw, prediction, observation)
        result, signal_notes = self.signal.update(ranged, observation, prediction=prediction)
        return result, {'range':range_notes, 'signal':signal_notes, 'GT_inputs':False,
            'composition':'source-bound-RGB-memory-then-temporal-color-evidence'}

    def acknowledge(self, command):
        self.range.acknowledge(command)


def make_evidence_bridge(*, nominal_calibration, strategy='baseline',
        front_bumper_x_m=2.4251085, ego_half_width_m=1.):
    """Use the baseline unless the simulation hypothesis is explicitly selected."""
    if strategy == 'baseline':
        bridge_class = TrafficEvidenceBridge
    elif strategy == 'hypothesis':
        bridge_class = HypothesisTrafficBridge
    else:
        raise ValueError('unknown-RGB-evidence-strategy')
    return bridge_class(nominal_calibration=nominal_calibration,
        front_bumper_x_m=front_bumper_x_m, ego_half_width_m=ego_half_width_m)
