"""Camera-observation traffic supervision, independent of simulator objects.

Distances are uncertain RGB estimates from the ego vehicle's front bumper.
World poses, actor handles, simulator signal state and map stop lines are not
policy inputs. The supervisor leaves the lane controller's steering untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

from .controller import ControllerConfig


class TrafficError(ValueError):
    """Invalid observation, control contract or supervisor configuration."""


def _number(value, name, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or minimum is not None and value < minimum:
        raise TrafficError('invalid-traffic-'+name)
    return float(value)


def _probability(value, name):
    value = _number(value, name, 0.)
    if value > 1.:
        raise TrafficError('invalid-traffic-'+name)
    return value


def _clip(value, low, high):
    return max(low, min(high, value))


@dataclass(frozen=True)
class TrackedObject:
    track_id: str
    kind: str
    distance_m: float | None
    lateral_y_m: float | None
    uncertainty_m: float
    closing_speed_mps: float | None
    in_path: bool
    confidence: float
    lateral_uncertainty_m: float = 0.
    # Optional one-sided bound. The nominal estimate and its symmetric interval
    # remain separate; a finite lower bound can exist with unbounded far range.
    conservative_nearest_distance_m: float | None = None

    def __post_init__(self):
        if type(self.track_id) is not str or not self.track_id or self.kind not in ('pedestrian', 'vehicle'):
            raise TrafficError('invalid-traffic-track-identity')
        if type(self.in_path) is not bool:
            raise TrafficError('invalid-traffic-path-membership')
        for name in ('distance_m','lateral_y_m','closing_speed_mps'):
            value=getattr(self,name)
            if value is not None: _number(value,name)
        if self.conservative_nearest_distance_m is not None:
            _number(self.conservative_nearest_distance_m,'nearest-distance-bound')
        _number(self.uncertainty_m,'range-uncertainty',0.)
        _number(self.lateral_uncertainty_m,'lateral-uncertainty',0.)
        _probability(self.confidence,'object-confidence')


@dataclass(frozen=True)
class TrafficObservation:
    frame: int
    timestamp: float
    objects: tuple[TrackedObject, ...] = ()
    signal: str = 'none'
    signal_confidence: float = 0.
    stop_line_distance_m: float | None = None
    stop_line_confidence: float = 0.
    stop_required: bool = False
    perception_valid: bool = True
    stop_line_uncertainty_m: float = .5
    stop_sign_id: str | None = None

    def __post_init__(self):
        if type(self.frame) is not int or self.frame < 0:
            raise TrafficError('invalid-traffic-frame')
        _number(self.timestamp,'timestamp',0.)
        if type(self.objects) is not tuple or any(type(item) is not TrackedObject for item in self.objects):
            raise TrafficError('invalid-traffic-typed-objects')
        if len({item.track_id for item in self.objects}) != len(self.objects):
            raise TrafficError('invalid-traffic-duplicate-track')
        if self.signal not in ('none','unknown','red','yellow','green'):
            raise TrafficError('invalid-traffic-signal')
        if type(self.stop_required) is not bool or type(self.perception_valid) is not bool:
            raise TrafficError('invalid-traffic-observation-flags')
        _probability(self.signal_confidence,'signal-confidence')
        _probability(self.stop_line_confidence,'line-confidence')
        _number(self.stop_line_uncertainty_m,'line-uncertainty',0.)
        if self.stop_line_distance_m is not None: _number(self.stop_line_distance_m,'line-distance')
        if self.stop_sign_id is not None and (type(self.stop_sign_id) is not str or not self.stop_sign_id):
            raise TrafficError('invalid-traffic-stop-sign-track')


@dataclass(frozen=True)
class TrafficConfig:
    fixed_delta_seconds: float = .05
    comfortable_deceleration_mps2: float = 2.
    object_standstill_gap_m: float = 2.
    following_time_headway_seconds: float = 1.5
    following_gain_per_second: float = .5
    stop_line_buffer_m: float = .75
    stop_position_tolerance_m: float = .75
    emergency_ttc_seconds: float = 1.5
    emergency_clearance_m: float = .5
    reaction_time_seconds: float = .2
    minimum_confidence: float = .5
    clear_hold_seconds: float = .5
    green_hold_seconds: float = .3
    complete_stop_speed_mps: float = .1
    mandatory_stop_hold_seconds: float = 1.
    perception_max_age_seconds: float = .15
    served_stop_reset_distance_m: float = 5.

    def __post_init__(self):
        for name,value in asdict(self).items():
            _number(value,name,0.)
        if self.fixed_delta_seconds != .05 or self.comfortable_deceleration_mps2 <= 0 or not 0 < self.minimum_confidence <= 1:
            raise TrafficError('invalid-traffic-supervisor-config')
        if self.complete_stop_speed_mps <= 0 or self.perception_max_age_seconds < self.fixed_delta_seconds:
            raise TrafficError('invalid-traffic-supervisor-timing')


class SpeedPI:
    """The existing PP gains, deadband and conditional integration unchanged."""
    def __init__(self, config: ControllerConfig | None = None):
        self.config=config or ControllerConfig();self.reset()

    def reset(self):
        self.integral=0.

    def step(self,target_speed_mps,speed_mps):
        cfg=self.config;target=_number(target_speed_mps,'target-speed',0.);speed=_number(speed_mps,'ego-speed',0.)
        error=target-speed
        integration_error=0. if abs(error)<=cfg.speed_deadband_mps else error
        candidate=_clip(self.integral+integration_error*cfg.fixed_delta_seconds,-cfg.integral_limit_m,cfg.integral_limit_m)
        candidate_effort=cfg.speed_kp*error+cfg.speed_ki*candidate
        antiwindup=((candidate_effort>cfg.throttle_max and integration_error>0.) or
                    (candidate_effort<-cfg.brake_max and integration_error<0.))
        if not antiwindup:self.integral=candidate
        effort=cfg.speed_kp*error+cfg.speed_ki*self.integral
        bounded=_clip(effort,-cfg.brake_max,cfg.throttle_max)
        return {'throttle':max(0.,bounded),'brake':max(0.,-bounded)}, {
            'speed_error_mps':error,'speed_integral_m':self.integral,'speed_pi_effort':effort,
            'speed_pi_antiwindup':antiwindup}


class TrafficSupervisor:
    """RGB traffic FSM with explicit uncertainty and finite-release hysteresis.

Unknown range or stale perception yields a stop request. Whether a vehicle
subsequently resumes and completes is judged independently from ground truth.
No fallback declares an indefinitely stopped episode successful.
"""
    def __init__(self,config: TrafficConfig | None = None,controller_config: ControllerConfig | None = None):
        self.config=config or TrafficConfig();self.pi=SpeedPI(controller_config);self.reset()

    def reset(self):
        self.pi.reset();self._time=None;self._frame=None;self._state='cruise'
        self._clear_since=None;self._green_since=None;self._red_latched=False
        self._mandatory=False;self._served=False;self._stop_since=None
        self._mandatory_id=None;self._served_id=None
        self._served_distance=0.;self._last_target=0.;self._feature_absent_since=None

    def _braking_cap(self,distance,uncertainty,buffer,speed):
        remaining=max(0.,distance-uncertainty-buffer-speed*self.config.reaction_time_seconds)
        return math.sqrt(2*self.config.comfortable_deceleration_mps2*remaining)

    def step(self,observation: TrafficObservation,ego_speed_mps,lane_command,*,timestamp):
        if type(observation) is not TrafficObservation:
            raise TrafficError('traffic-observation-must-be-typed-RGB-only')
        speed=_number(ego_speed_mps,'ego-speed',0.);now=_number(timestamp,'control-time',0.)
        if self._time is not None and abs(now-self._time-self.config.fixed_delta_seconds)>1e-6:
            raise TrafficError('nonconsecutive-traffic-control-timestamp')
        if type(lane_command) is not dict or any(k not in lane_command for k in ('steer','throttle','brake','hand_brake','reverse','diagnostics')):
            raise TrafficError('invalid-traffic-lane-command')
        steer=_number(lane_command['steer'],'lane-steer')
        if abs(steer)>1. or lane_command['reverse'] is not False or lane_command['hand_brake'] is not False:
            raise TrafficError('invalid-traffic-forward-lane-command')
        nominal=_number(lane_command['diagnostics'].get('effective_target_speed_mps',
            lane_command['diagnostics'].get('target_speed_mps')),'lane-target-speed',0.)
        nominal=min(nominal,self.pi.config.target_speed_kmh/3.6)
        cfg=self.config;reasons=[];caps=[];emergency=False;hazard_state=None
        stale=(now-observation.timestamp>cfg.perception_max_age_seconds+1e-9 or observation.timestamp>now+1e-7
               or self._frame is not None and observation.frame<=self._frame)
        usable=observation.perception_valid and not stale
        if usable:self._frame=observation.frame
        self._time=now
        if self._served:self._served_distance+=speed*cfg.fixed_delta_seconds
        if not usable:
            caps.append(0.);hazard_state='uncertain';reasons.append('stale_perception' if stale else 'invalid_perception')
            self._green_since=None
        else:
            for obj in observation.objects:
                if not obj.in_path:continue
                nearest=(obj.conservative_nearest_distance_m if obj.conservative_nearest_distance_m is not None
                         else None if obj.distance_m is None else obj.distance_m-obj.uncertainty_m)
                if obj.confidence<cfg.minimum_confidence:
                    caps.append(0. if nearest is None else self._braking_cap(
                        nearest,0.,cfg.object_standstill_gap_m,speed))
                    hazard_state='uncertain';reasons.append('uncertain_in_path_object:'+obj.track_id)
                    continue
                if nearest is None:
                    caps.append(0.);hazard_state=hazard_state or 'uncertain';reasons.append('unknown_object_range:'+obj.track_id);continue
                signed_closing=speed if obj.closing_speed_mps is None else obj.closing_speed_mps
                closing=max(0.,signed_closing)
                ttc=nearest/closing if closing>1e-6 else math.inf
                if nearest<=cfg.emergency_clearance_m or ttc<=cfg.emergency_ttc_seconds:
                    emergency=True;reasons.append('emergency_object:'+obj.track_id)
                cap=self._braking_cap(nearest,0.,cfg.object_standstill_gap_m,speed)
                if obj.kind=='pedestrian':
                    state='yield_person';reasons.append('pedestrian_in_path:'+obj.track_id)
                else:
                    lead_speed=max(0.,speed-signed_closing)
                    follow=max(0.,lead_speed+cfg.following_gain_per_second*(nearest-cfg.object_standstill_gap_m-cfg.following_time_headway_seconds*speed))
                    cap=min(cap,follow);state='follow_vehicle';reasons.append('vehicle_in_path:'+obj.track_id)
                caps.append(cap)
                if hazard_state not in ('yield_person','uncertain'):hazard_state=state
            signal=observation.signal
            if signal in ('red','yellow','unknown') or signal=='green' and observation.signal_confidence<cfg.minimum_confidence:
                self._red_latched=True;self._green_since=None
            elif signal=='green' and observation.signal_confidence>=cfg.minimum_confidence:
                if self._green_since is None:self._green_since=now
                if now-self._green_since>=cfg.green_hold_seconds-1e-9:self._red_latched=False
            else:self._green_since=None
            sign_id=observation.stop_sign_id or 'unidentified-stop'
            if observation.stop_required and (not self._served or sign_id!=self._served_id):
                if self._mandatory_id!=sign_id:self._stop_since=None
                self._mandatory=True;self._mandatory_id=sign_id
            line_known=(observation.stop_line_distance_m is not None and observation.stop_line_confidence>=cfg.minimum_confidence)
            if self._red_latched or self._mandatory:
                line_cap=(self._braking_cap(observation.stop_line_distance_m,observation.stop_line_uncertainty_m,cfg.stop_line_buffer_m,speed)
                          if line_known else 0.)
                caps.append(line_cap);reasons.append('red_or_unknown_signal' if self._red_latched else 'mandatory_stop')
                if hazard_state not in ('yield_person','uncertain'):hazard_state='stop_red' if self._red_latched else 'stop_mandatory'
            if self._mandatory:
                at_line=(line_known and -cfg.stop_position_tolerance_m<=observation.stop_line_distance_m
                         <=cfg.stop_line_buffer_m+cfg.stop_position_tolerance_m+observation.stop_line_uncertainty_m)
                if speed<=cfg.complete_stop_speed_mps and at_line:
                    if self._stop_since is None:self._stop_since=now
                    if now-self._stop_since>=cfg.mandatory_stop_hold_seconds-1e-9:
                        self._served=True;self._mandatory=False;self._served_distance=0.;self._stop_since=None
                        self._served_id=self._mandatory_id;self._mandatory_id=None
                else:self._stop_since=None
            if self._served and not observation.stop_required and observation.stop_line_distance_m is None:
                if self._feature_absent_since is None:self._feature_absent_since=now
                if now-self._feature_absent_since>=cfg.clear_hold_seconds-1e-9 and self._served_distance>=cfg.served_stop_reset_distance_m:
                    self._served=False;self._feature_absent_since=None
                    self._served_id=None
            else:self._feature_absent_since=None
        target=min([nominal]+caps)
        previous_state=self._state
        if emergency:
            target=0.;state='emergency';self._clear_since=None
        elif hazard_state:
            state=hazard_state;self._clear_since=None
        elif previous_state!='cruise':
            if self._clear_since is None:self._clear_since=now
            if now-self._clear_since<cfg.clear_hold_seconds-1e-9:
                target=min(target,self._last_target);state='resume_wait';reasons.append('clear_hysteresis')
            else:state='cruise';self._clear_since=None
        else:state='cruise'
        command,pi=self.pi.step(target,speed)
        hold=(target<=1e-9 and speed<=cfg.complete_stop_speed_mps)
        pi['integrator_reset_on_hold']=hold
        pi['integral_before_hold_reset']=self.pi.integral
        if emergency or hold:
            command.update(throttle=0.,brake=1.)
            # A stopped wait must not retain a positive integrator that causes
            # an avoidable jump on release. Saturated motion uses normal PI.
            if hold:self.pi.reset();pi['speed_integral_m']=0.
        result=copy.deepcopy(lane_command);result.update(command)
        result.update(steer=steer,hand_brake=False,reverse=False)
        result['diagnostics'].update(effective_target_speed_mps=target,effective_target_speed_kmh=target*3.6,
            speed_mps=speed,speed_error_mps=pi['speed_error_mps'],speed_integral_m=pi['speed_integral_m'],
            speed_pi_effort=pi['speed_pi_effort'],speed_pi_antiwindup=pi['speed_pi_antiwindup'])
        result['diagnostics']['traffic_supervisor']={
            'version':'RGB-traffic-supervisor-v1','state':state,'previous_state':previous_state,
            'reasons':reasons,'target_speed_mps':target,'lane_target_speed_mps':nominal,
            'ego_speed_mps':speed,'observation_frame':observation.frame,'observation_timestamp':observation.timestamp,
            'control_timestamp':now,'perception_usable':usable,'emergency_brake':emergency,
            'red_latched':self._red_latched,'mandatory_stop_pending':self._mandatory,'mandatory_stop_served':self._served,
            'original_lane_throttle_not_used':lane_command['throttle'],'original_lane_brake_not_used':lane_command['brake'],
            'longitudinal_brake_override':emergency or hold,
            'steering_changed':False,'GT_inputs':False,'PI':pi}
        self._state=state;self._last_target=target
        return result
