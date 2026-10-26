"""Compose injected RGB perception, waypoint control and traffic supervision."""
from dataclasses import dataclass
import copy
import math
import time

from carla_tools.controller import ControllerError
from carla_tools.traffic_behavior import TrafficObservation, TrafficSupervisor, TrafficConfig
from carla_tools.traffic_perception import RGBObservation, to_traffic_observation, TrafficPerceptionConfig
from .traffic_memory import DirectedRangeBridge
from .traffic_signal import SignalEvidenceAdapter



class TrafficEvidenceBridge:
    """Combine distance/memory and signal evidence without simulator inputs."""
    def __init__(self, *, nominal_calibration, front_bumper_x_m=2.4251085, ego_half_width_m=1.):
        self.range=DirectedRangeBridge(nominal_calibration=nominal_calibration,
            front_bumper_x_m=front_bumper_x_m,ego_half_width_m=ego_half_width_m)
        self.signal=SignalEvidenceAdapter()
        self.reset()

    def reset(self):self.range.reset();self.signal.reset()

    def update(self,raw,prediction,observation):
        ranged,range_notes=self.range.update(raw,prediction,observation)
        result,signal_notes=self.signal.update(ranged,observation)
        return result,{'range':range_notes,'signal':signal_notes,'GT_inputs':False,
            'composition':'RGB-directed-nearest-interval-then-signal-evidence;no-GT-inputs'}

    def acknowledge(self,command):self.range.acknowledge(command)


@dataclass(frozen=True)
class TrafficPolicyConfig:
    maximum_speed_mps: float = 3.

    def __post_init__(self):
        if type(self.maximum_speed_mps) not in (int,float) or self.maximum_speed_mps!=3.:
            raise ValueError('traffic-policy-3mps-envelope-required')


def _validate_default_bridge_contract(perception_config, supervisor_config):
    """Keep the fixed evidence bounds consistent with their control contract.

    Camera/vehicle geometry is explicitly injected. Independent detector score
    thresholds and supervisor settings remain configurable. Callers providing
    their own bridge are responsible for matching its estimation/control model.
    """
    if not isinstance(perception_config, TrafficPerceptionConfig):
        raise ValueError('traffic-policy-default-bridge-typed-perception-config-required')
    if not isinstance(supervisor_config, TrafficConfig):
        raise ValueError('traffic-policy-default-bridge-typed-supervisor-config-required')
    perception_fixed = {'bbox_footpoint_uncertainty_px':3., 'camera_height_uncertainty_m':.15,
        'camera_pitch_uncertainty_deg':2., 'path_margin_m':.35,
        'minimum_signal_confidence':.55, 'track_max_age_s':.5}
    supervisor_fixed = {'fixed_delta_seconds':.05, 'comfortable_deceleration_mps2':2.,
        'object_standstill_gap_m':2., 'reaction_time_seconds':.2,
        'minimum_confidence':.5, 'served_stop_reset_distance_m':5.}
    for group,config,expected in (('perception',perception_config,perception_fixed),
                                  ('supervisor',supervisor_config,supervisor_fixed)):
        for name,value in expected.items():
            if getattr(config,name)!=value:
                raise ValueError('traffic-policy-default-bridge-'+group+'-config-mismatch:'+name)


class TrafficPolicy:
    def __init__(self,path_predictor,traffic_perception,lane_controller,*,config=None,supervisor=None,bridge=None):
        self.path=path_predictor;self.traffic=traffic_perception;self.lane=lane_controller
        self.config=config or TrafficPolicyConfig();self.supervisor=supervisor or TrafficSupervisor()
        if bridge is None:
            _validate_default_bridge_contract(traffic_perception.config,self.supervisor.config)
            self.bridge=TrafficEvidenceBridge(nominal_calibration=traffic_perception.calibration,
                front_bumper_x_m=traffic_perception.config.front_bumper_x_m,ego_half_width_m=traffic_perception.config.ego_half_width_m)
        else:
            # Explicit custom evidence adapters own compatibility with the
            # caller's perception and supervisor configuration.
            self.bridge=bridge
        self.reset()

    def reset(self):
        self.lane.reset();self.traffic.reset();self.supervisor.reset();self.bridge.reset();self.previous=0.

    def command(self,observation:RGBObservation):
        if type(observation) is not RGBObservation:raise ValueError('traffic-policy-RGB-only-input-required')
        began=time.perf_counter();timer=self.path.timer()
        prediction=self.path.predict_profiled(observation.rgb,timer,width=observation.width,height=observation.height)
        raw_traffic=self.traffic.process(observation)
        if raw_traffic['frame']!=observation.frame or abs(raw_traffic['timestamp']-observation.timestamp)>1e-7:
            raise ValueError('traffic-policy-observation-age-mismatch')
        traffic,bridge_notes=self.bridge.update(raw_traffic,prediction,observation)
        original=None;failure=None
        if prediction['status']=='PASS':
            try:original=self.lane.step(prediction['waypoints_ego_m'],observation.ego_speed_mps,dt_seconds=.05)
            except ControllerError as error:failure={'type':'ControllerError','code':str(error)}
        if original is None:
            source={'steer':self.previous,'throttle':0.,'brake':1.,'hand_brake':False,'reverse':False,
                    'diagnostics':{'effective_target_speed_mps':0.,'lane_policy_invalid':True}}
        else:source=copy.deepcopy(original)
        lane_target=source['diagnostics']['effective_target_speed_mps']
        source['diagnostics']['effective_target_speed_mps']=min(lane_target,self.config.maximum_speed_mps)
        typed=to_traffic_observation(traffic)
        if original is None:
            typed=TrafficObservation(frame=typed.frame,timestamp=typed.timestamp,perception_valid=False)
        command=self.supervisor.step(typed,observation.ego_speed_mps,source,timestamp=observation.timestamp)
        self.bridge.acknowledge(command)
        self.previous=command['steer']
        if hasattr(self.lane,'_previous_steer'):self.lane._previous_steer=self.previous
        return command,{'prediction':prediction,'traffic_perception_raw':raw_traffic,'traffic_perception':traffic,
            'RGB_temporal_adapter':bridge_notes,'lane_command_before_speed_cap':original,
            'lane_controller_failure':failure,'maximum_speed_mps':self.config.maximum_speed_mps,
            'lane_target_before_cap_mps':lane_target,'command':copy.deepcopy(command),
            'GT_inputs':False,'world_inputs_received':False,'source_frame':observation.frame,
            'source_timestamp':observation.timestamp,'path_timing':timer.finish(),
            'policy_wall_ms':(time.perf_counter()-began)*1000.,
            'fixed_delta_seconds':.05}
