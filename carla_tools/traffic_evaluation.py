"""Same-frame ground-truth judging for six bounded traffic functional scenes.

This sidecar is an evaluator input, never a camera policy observation. It judges
completion, road containment, rules, waiting and restart together; a missing
hazard or an indefinitely stopped car cannot pass. Lateral error is descriptive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


SCENARIOS=('clear_road','leading_vehicle','cut_in','crossing_pedestrian','red_to_green','mandatory_stop')


class TrafficEvaluationError(ValueError):
    pass


def _finite(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise TrafficEvaluationError('invalid-traffic-GT-'+name)
    return float(value)


@dataclass(frozen=True)
class TrafficGroundTruth:
    frame: int
    timestamp: float
    speed_mps: float
    route_progress_m: float
    body_lane_margin_m: float
    collision_events: int = 0
    lane_invasion_events: int = 0
    minimum_object_clearance_m: float | None = None
    front_bumper_line_distance_m: float | None = None
    signal_state: str = 'none'
    stimulus_present: bool = False
    stimulus_visible: bool = False
    hazard_cleared: bool = False
    expected_wait: bool = False
    phase: str = 'driving'
    lateral_error_m: float | None = None

    def __post_init__(self):
        if type(self.frame) is not int or self.frame<0:
            raise TrafficEvaluationError('invalid-traffic-GT-frame')
        for key in ('timestamp','speed_mps','route_progress_m','body_lane_margin_m'):_finite(getattr(self,key),key)
        if self.timestamp<0 or self.speed_mps<0 or self.route_progress_m<0:
            raise TrafficEvaluationError('invalid-traffic-GT-negative-time-speed-progress')
        for key in ('minimum_object_clearance_m','front_bumper_line_distance_m','lateral_error_m'):
            if getattr(self,key) is not None:_finite(getattr(self,key),key)
        for key in ('collision_events','lane_invasion_events'):
            if type(getattr(self,key)) is not int or getattr(self,key)<0:
                raise TrafficEvaluationError('invalid-traffic-GT-event-count')
        for key in ('stimulus_present','stimulus_visible','hazard_cleared','expected_wait'):
            if type(getattr(self,key)) is not bool:raise TrafficEvaluationError('invalid-traffic-GT-flag')
        if self.signal_state not in ('none','red','yellow','green','unknown') or self.phase not in ('driving','stopping'):
            raise TrafficEvaluationError('invalid-traffic-GT-state')


@dataclass(frozen=True)
class TrafficEvaluationConfig:
    scenario: str
    route_length_m: float
    fixed_delta_seconds: float = .05
    completion_tolerance_m: float = .5
    timeout_seconds: float = 45.
    stopped_speed_mps: float = .1
    mandatory_stop_hold_seconds: float = 1.
    stop_zone_max_distance_m: float = 3.
    maximum_restart_delay_seconds: float = 2.
    restart_progress_m: float = .25
    maximum_false_stop_seconds: float = 2.
    maximum_legal_wait_seconds: float = 15.
    terminal_stop_hold_seconds: float = .5
    numerical_tolerance_m: float = 1e-6

    def __post_init__(self):
        if self.scenario not in SCENARIOS:raise TrafficEvaluationError('invalid-traffic-scenario')
        for key,value in asdict(self).items():
            if key=='scenario':continue
            if _finite(value,key)<=0:raise TrafficEvaluationError('invalid-traffic-evaluation-config')
        if self.fixed_delta_seconds!=.05 or self.completion_tolerance_m>=self.route_length_m:
            raise TrafficEvaluationError('invalid-traffic-evaluation-geometry-or-timestep')


class TrafficEvaluator:
    def __init__(self,config:TrafficEvaluationConfig):
        if type(config) is not TrafficEvaluationConfig:raise TrafficEvaluationError('typed-traffic-evaluation-config-required')
        self.config=config;self.rows=[]

    def update(self,ground_truth:TrafficGroundTruth):
        if type(ground_truth) is not TrafficGroundTruth:
            raise TrafficEvaluationError('typed-GT-sidecar-required-not-policy-observation')
        if self.rows and (ground_truth.frame!=self.rows[-1].frame+1 or
                abs(ground_truth.timestamp-self.rows[-1].timestamp-self.config.fixed_delta_seconds)>1e-6):
            raise TrafficEvaluationError('nonconsecutive-traffic-GT-frame-time')
        self.rows.append(ground_truth)

    def finalize(self):
        cfg=self.config;rows=self.rows
        if not rows:raise TrafficEvaluationError('empty-traffic-GT-episode')
        failures=[];scenario_errors=[];red_crossings=[];mandatory_crossings=[]
        stop_start=None;stop_completed=False;full_stop_holds=[]
        false_since=None;false_intervals=[];legal_wait=0.;release=None;release_progress=None;restart=None
        terminal_since=None;previous=None;stimulus_seen=False;visible_seen=False;clear_seen=False
        stop_position=None;signal_states=set()
        for row in rows:
            signal_states.add(row.signal_state)
            stimulus_seen|=row.stimulus_present;visible_seen|=row.stimulus_present and row.stimulus_visible
            if stimulus_seen and row.hazard_cleared:clear_seen=True
            # A restart timer starts at the externally observed release after
            # an actual scenario stimulus, never at the controller's self-report.
            if (cfg.scenario in ('leading_vehicle','cut_in','crossing_pedestrian','red_to_green')
                    and release is None and stimulus_seen and row.hazard_cleared and not row.expected_wait):
                release=row.timestamp;release_progress=row.route_progress_m
            if release is not None and restart is None and row.speed_mps>cfg.stopped_speed_mps and row.route_progress_m-release_progress>=cfg.restart_progress_m:
                restart=row.timestamp
            line=row.front_bumper_line_distance_m
            at_stop=(line is not None and -cfg.numerical_tolerance_m<=line<=cfg.stop_zone_max_distance_m
                     and row.speed_mps<=cfg.stopped_speed_mps)
            if at_stop:
                if stop_start is None:stop_start=row.timestamp;stop_position=line
                if row.timestamp-stop_start>=cfg.mandatory_stop_hold_seconds-1e-9 and not stop_completed:
                    stop_completed=True;full_stop_holds.append({'start':stop_start,'completed':row.timestamp,'front_bumper_gap_m':stop_position})
                    if cfg.scenario=='mandatory_stop' and release is None:
                        release=row.timestamp;release_progress=row.route_progress_m
            else:stop_start=None
            crossing=(line is not None and line < -cfg.numerical_tolerance_m and
                      (previous is None or previous.front_bumper_line_distance_m is not None and
                       previous.front_bumper_line_distance_m>=-cfg.numerical_tolerance_m))
            if crossing:
                if row.signal_state=='red' or previous is not None and previous.signal_state=='red':red_crossings.append(row.frame)
                if cfg.scenario=='mandatory_stop' and not stop_completed:mandatory_crossings.append(row.frame)
            complete=row.route_progress_m>=cfg.route_length_m-cfg.completion_tolerance_m
            unexpected=(row.phase=='driving' and not complete and row.speed_mps<=cfg.stopped_speed_mps and not row.expected_wait)
            if unexpected:
                if false_since is None:false_since=row.timestamp
            elif false_since is not None:
                false_intervals.append({'start':false_since,'end':row.timestamp,'duration_seconds':row.timestamp-false_since});false_since=None
            if row.expected_wait and row.speed_mps<=cfg.stopped_speed_mps:legal_wait+=cfg.fixed_delta_seconds
            if complete and row.phase=='stopping' and row.speed_mps<=cfg.stopped_speed_mps:
                if terminal_since is None:terminal_since=row.timestamp
            else:terminal_since=None
            previous=row
        if false_since is not None:false_intervals.append({'start':false_since,'end':rows[-1].timestamp,'duration_seconds':rows[-1].timestamp-false_since})
        completed=max(row.route_progress_m for row in rows)>=cfg.route_length_m-cfg.completion_tolerance_m
        terminal_stopped=terminal_since is not None and rows[-1].timestamp-terminal_since>=cfg.terminal_stop_hold_seconds-1e-9
        collisions=sum(row.collision_events for row in rows);invasions=sum(row.lane_invasion_events for row in rows)
        minimum_margin=min(row.body_lane_margin_m for row in rows)
        clearances=[row.minimum_object_clearance_m for row in rows if row.minimum_object_clearance_m is not None]
        minimum_clearance=min(clearances) if clearances else None
        if collisions:failures.append('collision')
        if invasions:failures.append('lane_invasion')
        if minimum_margin < -cfg.numerical_tolerance_m:failures.append('body_outside_lane')
        if minimum_clearance is not None and minimum_clearance < -cfg.numerical_tolerance_m:failures.append('object_footprint_overlap')
        if red_crossings:failures.append('red_light_stop_line_crossing')
        if mandatory_crossings:failures.append('mandatory_line_crossed_without_full_stop')
        if cfg.scenario=='mandatory_stop' and not stop_completed:failures.append('mandatory_full_stop_not_observed')
        if not completed:failures.append('route_not_completed')
        if not terminal_stopped:failures.append('terminal_stop_not_observed')
        duration=rows[-1].timestamp-rows[0].timestamp
        if duration>cfg.timeout_seconds+1e-9:failures.append('scenario_deadline_exceeded')
        if any(item['duration_seconds']>cfg.maximum_false_stop_seconds+1e-9 for item in false_intervals):failures.append('unnecessary_or_permanent_stop')
        if legal_wait>cfg.maximum_legal_wait_seconds+1e-9:failures.append('legal_wait_budget_exceeded')
        requires_release=cfg.scenario in ('leading_vehicle','cut_in','crossing_pedestrian','red_to_green')
        if cfg.scenario!='clear_road':
            if not stimulus_seen:scenario_errors.append('required_stimulus_never_present')
            if not visible_seen:scenario_errors.append('required_stimulus_never_visible_in_RGB_FOV')
        if cfg.scenario in ('red_to_green','mandatory_stop') and any(row.front_bumper_line_distance_m is None for row in rows):
            scenario_errors.append('required_same_frame_stop_line_GT_missing')
        if cfg.scenario in ('leading_vehicle','cut_in','crossing_pedestrian') and any(
                row.stimulus_present and row.minimum_object_clearance_m is None for row in rows):
            scenario_errors.append('required_same_frame_object_clearance_GT_missing')
        if requires_release and not clear_seen:scenario_errors.append('required_stimulus_never_cleared')
        if cfg.scenario=='red_to_green' and not {'red','green'}<=signal_states:scenario_errors.append('red_to_green_transition_not_observed')
        if cfg.scenario=='red_to_green' and not stop_completed:failures.append('red_full_stop_not_observed')
        if requires_release or cfg.scenario=='mandatory_stop':
            if release is None or restart is None:failures.append('restart_not_observed')
            elif restart-release>cfg.maximum_restart_delay_seconds+1e-9:failures.append('restart_deadline_exceeded')
        lateral=[abs(row.lateral_error_m) for row in rows if row.lateral_error_m is not None]
        return {'status':'PASS','scenario':cfg.scenario,'functional_success':not failures and not scenario_errors,
                'scenario_valid':not scenario_errors,'failure_reasons':failures,'scenario_errors':scenario_errors,
                'sample_count':len(rows),'route_completed':completed,'terminal_stopped':terminal_stopped,
                'duration_seconds':duration,'maximum_progress_m':max(row.route_progress_m for row in rows),
                'collision_count':collisions,'lane_invasion_count':invasions,'minimum_body_lane_margin_m':minimum_margin,
                'minimum_object_clearance_m':minimum_clearance,'red_line_crossing_frames':red_crossings,
                'mandatory_line_violation_frames':mandatory_crossings,'complete_stop_holds':full_stop_holds,
                'legal_wait_seconds':legal_wait,'false_stop_intervals':false_intervals,
                'release_timestamp':release,'restart_timestamp':restart,
                'restart_delay_seconds':None if release is None or restart is None else restart-release,
                'lateral_abs_mean_m':sum(lateral)/len(lateral) if lateral else None,
                'CTE_used_as_acceptance_gate':False,'GT_sidecar_used_only_for_evaluation':True,
                'actual_simulator_execution_claimed':False}
