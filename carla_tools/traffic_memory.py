"""Causal RGB object/stop-line memory with bounded vehicle range evidence.

Inputs contain image-derived observations, frame/time, speed and a fixed nominal
camera mounting. This module has no world, map, actor or route inputs.
"""
import itertools
from carla_tools.traffic_perception import (
    TrafficPerceptionConfig, _ground_point, validate_nominal_calibration,
)

import copy
import math


class RGBTemporalBridge:
    def __init__(self,*,front_bumper_x_m=2.4251085,ego_half_width_m=.95):
        self.front=front_bumper_x_m;self.half=ego_half_width_m;self.reset()
    def reset(self):
        self.time=None;self.speed=None;self.line=None;self.tracks={};self.sign=None;self.sign_counter=0
        self.sign_last_seen=None;self.sign_served=False;self.sign_travel=0.
    def acknowledge(self,command):
        served=command['diagnostics']['traffic_supervisor']['mandatory_stop_served']
        if served and not self.sign_served:self.sign_served=True;self.sign_travel=0.

    def _path_membership(self,item,points):
        if not item.get('ground_valid') or item.get('distance_m') is None or item.get('lateral_y_m') is None or not points:
            return item['in_path'],'nominal-unknown-geometry'
        x=item['distance_m']+self.front;y=item['lateral_y_m']
        if not min(p[0] for p in points)<=x<=max(p[0] for p in points):return item['in_path'],'nominal-outside-NN-path-support'
        distances=[]
        for a,b in zip(points,points[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1];norm=dx*dx+dy*dy
            if norm<=1e-12:continue
            f=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/norm))
            distances.append(math.hypot(x-a[0]-f*dx,y-a[1]-f*dy))
        if not distances:return True,'uncertain-invalid-NN-path-segments'
        object_half=.35 if item['kind']=='pedestrian' else 1.
        band=self.half+.35+object_half+item.get('lateral_uncertainty_m',0.)
        return min(distances)<=band,'RGB-future-polyline-distance-plus-uncertainty'

    def update(self,raw,prediction,observation):
        if raw.get('GT_inputs') is not False or raw['frame']!=observation.frame or abs(raw['timestamp']-observation.timestamp)>1e-7:
            raise ValueError('RGB-memory-input-provenance-or-age-mismatch')
        result=copy.deepcopy(raw);now=observation.timestamp;speed=observation.ego_speed_mps
        dt=0. if self.time is None else now-self.time
        if self.time is not None and (dt<=0 or abs(dt-.05)>1e-6):raise ValueError('RGB-memory-nonconsecutive-time')
        travel=0. if self.speed is None else .5*(self.speed+speed)*dt
        self.time=now;self.speed=speed
        if self.sign_served:self.sign_travel+=travel
        notes={'GT_inputs':False,'raw_result_modified':False,'line':None,'objects':[],'unresolved_lost_tracks':[],
               'line_maximum_age_seconds':2.,'object_propagation_maximum_age_seconds':.5,
               'positive_out_of_path_clear_hold_seconds':.2}
        if self.line:
            self.line['distance_m']-=travel
            self.line['uncertainty_m']+=.1*dt+.05*travel
        if raw.get('stop_line_distance_m') is not None and raw.get('stop_line_confidence',0.)>=.5:
            self.line={'distance_m':raw['stop_line_distance_m'],
                'uncertainty_m':raw.get('stop_line',{}).get('ground_estimate',{}).get('distance_uncertainty_m',.5),
                'confidence':raw['stop_line_confidence'],'last_seen_frame':observation.frame,'last_seen_timestamp':now}
        elif self.line:
            age=now-self.line['last_seen_timestamp']
            if age<=2.+1e-9:
                result['stop_line_distance_m']=self.line['distance_m'];result['stop_line_confidence']=self.line['confidence']
                result.setdefault('stop_line',{}).update(observed=False,status='causal-RGB-memory-estimate',
                    ground_estimate={'distance_uncertainty_m':self.line['uncertainty_m']})
            else:self.line=None
        if self.line:notes['line']={**self.line,'age_seconds':now-self.line['last_seen_timestamp'],
            'source':'last-actual-RGB-line-minus-measured-ego-travel;not-current-observation'}
        if raw.get('stop_required'):
            if self.sign is None:self.sign_counter+=1;self.sign=f'RGB-stop-context-{self.sign_counter}';self.sign_served=False
            self.sign_last_seen=now;result['stop_sign_id']=self.sign
        elif (self.sign and self.sign_served and self.sign_last_seen is not None and now-self.sign_last_seen>=.5
                and self.sign_travel>=5.):
            self.sign=None;self.sign_served=False;self.sign_travel=0.
        points=prediction.get('waypoints_ego_m') if prediction.get('status')=='PASS' else None
        observed=set();objects=[]
        for original in raw['objects']:
            item=copy.deepcopy(original);identifier=item['track_id'];observed.add(identifier)
            in_path,method=self._path_membership(item,points);item['in_path']=in_path
            old=self.tracks.get(identifier)
            if in_path and item['confidence']>=.5:
                nearest=None if item['distance_m'] is None else item['distance_m']-item['distance_uncertainty_m']
                relevant=nearest is None or nearest<=2.+speed*speed/4.+.2*speed
                self.tracks[identifier]={'item':copy.deepcopy(item),'last_seen':now,'last_seen_frame':observation.frame,
                                        'clear_since':None,'threat_relevant':relevant}
            elif old:
                positive_clear=not in_path and item.get('ground_valid') and item['confidence']>=.5
                if positive_clear:
                    if old['clear_since'] is None:old['clear_since']=now
                    if now-old['clear_since']>=.2-1e-9:del self.tracks[identifier]
                    else:item['in_path']=True
                else:
                    old['clear_since']=None;item['in_path']=True
            notes['objects'].append({'track_id':identifier,'membership_source':method,'source':'current-RGB'})
            objects.append(item)
        for identifier,memory in list(self.tracks.items()):
            if identifier in observed:continue
            age=now-memory['last_seen'];item=copy.deepcopy(memory['item'])
            # A missing ground footpoint is an explicit unknown, not a numeric
            # estimate. Keep None until the typed boundary supplies its declared
            # unknown-range uncertainty; never invent a confident range.
            closing=item.get('closing_speed_mps');closing=speed if closing is None else closing
            if item.get('distance_m') is not None:item['distance_m']-=closing*age
            if item.get('distance_uncertainty_m') is not None:item['distance_uncertainty_m']+=2.*age
            if item.get('lateral_uncertainty_m') is not None:item['lateral_uncertainty_m']+=age
            nearest=(None if item.get('distance_m') is None or item.get('distance_uncertainty_m') is None
                else item['distance_m']-item['distance_uncertainty_m'])
            # Risk can grow during a dropout. Reassess the propagated lower
            # bound before expiry and latch danger until positive clearance.
            if nearest is None or nearest<=2.+speed*speed/4.+.2*speed:
                memory['threat_relevant']=True
            if age<=.5+1e-9:
                item['in_path']=True;objects.append(item)
                notes['objects'].append({'track_id':identifier,'source':'last-RGB-track-propagated',
                    'last_seen_timestamp':memory['last_seen'],'last_seen_frame':memory['last_seen_frame'],'age_seconds':age})
            elif memory['threat_relevant']:
                # Empty detections are not positive evidence that a nearby
                # pedestrian vanished safely. Hold uncertainty until a real
                # observation of this track establishes lateral clearance.
                result['perception_valid']=False;notes['unresolved_lost_tracks'].append(identifier)
            else:del self.tracks[identifier]
        result['objects']=objects
        return result,notes


ROADSIDE_CONFIG={'version':'bounded-RGB-roadside-car-border-certificate-v1',
    'minimum_positive_clear_seconds':.2,'minimum_clearance_m':.1,
    'maximum_certificate_age_seconds':1.5,'maximum_ego_travel_m':3.,
    'border_fraction':.025,'maximum_path_lateral_change_m':.25,
    'maximum_inward_bbox_drift_px':2.,'confidence_minimum':.5,
    'class':'car-only;never-pedestrian','existing_unresolved_threat_can_be_erased':False}

class RoadsideBridge(RGBTemporalBridge):
    def reset(self):
        super().reset();self.outside={};self.travel=0.;self.observation=None;self.current_points=None

    def update(self,raw,prediction,observation):
        if self.time is not None:
            self.travel+=.5*(self.speed+observation.ego_speed_mps)*(observation.timestamp-self.time)
        self.observation=observation
        self.current_points=prediction.get('waypoints_ego_m') if prediction.get('status')=='PASS' else None
        return super().update(raw,prediction,observation)

    def _path_membership(self,item,points):
        inside,method=super()._path_membership(item,points)
        if item.get('label')!='car' or item['kind']!='vehicle' or item['confidence']<.5 or not points:
            return inside,method
        box=item.get('bbox_xyxy');w=self.observation.width
        if not isinstance(box,list) or len(box)!=4:return inside,method
        side='right' if box[0]>w/2 else 'left' if box[2]<w/2 else None
        border=(side=='right' and box[2]>=w*(1.-.025) or side=='left' and box[0]<=w*.025)
        now=self.observation.timestamp;identifier=item['track_id'];old=self.outside.get(identifier)
        if item.get('ground_valid') and not inside and not border and side:
            x=item['distance_m']+self.front;y=item['lateral_y_m'];distances=[]
            for a,b in zip(points,points[1:]):
                dx,dy=b[0]-a[0],b[1]-a[1];norm=dx*dx+dy*dy
                if norm<=1e-12:continue
                f=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/norm))
                distances.append(math.hypot(x-a[0]-f*dx,y-a[1]-f*dy))
            clearance=min(distances)-(self.half+.35+1.+item['lateral_uncertainty_m']) if distances else -math.inf
            if clearance>=.1:
                since=old['since'] if old and old['side']==side and now-old['last']<=.1 else now
                self.outside[identifier]={'since':since,'last':now,'side':side,'box':list(box),
                    'points':[list(p) for p in points],'travel':self.travel,'clearance_m':clearance,
                    'positive_seconds':now-since}
                return inside,method
        if border and old and old['side']==side and old['positive_seconds']>=.2-1e-9:
            inner=box[0] if side=='right' else -box[2]
            previous_inner=old['box'][0] if side=='right' else -old['box'][2]
            path_change=max(abs(a[1]-b[1]) for a,b in zip(points,old['points']))
            supported=(now-old['last']<=1.5+1e-9 and self.travel-old['travel']<=3.+1e-9
                and inner>=previous_inner-2. and path_change<=.25
                and identifier not in self.tracks)
            if supported:
                return False,'bounded-prior-positive-RGB-car-clearance-plus-outward-border-bbox'
            if not item.get('ground_valid'):
                return True,'expired-or-inconsistent-RGB-border-evidence-uncertain'
        if not border and inside:self.outside.pop(identifier,None)
        return inside,method


DIRECTED_RANGE_CONFIG = {
    'version': 'explicit-directed-far-ground-range-bounds-v3',
    'applies_to': 'vehicle-only; pedestrian-fields-and-memory-unchanged',
    'far_unbounded_encoding': None,
    'nominal_RGB_fields_modified': False,
    'bounded_assumptions': {'footpoint_px': 3., 'mount_height_m': .15,
                            'mount_pitch_deg': 2., 'level_ground': True},
    'missing_vehicle_lower_bound': 'subtract-max-ego-or-known-closing-and-2mps-growth',
    'new_range_model_or_training': False,
    'far_bound_minimum_m': 7.85,
    'far_bound_minimum_derivation': '2m-gap+v^2/4+.2s*v+.6s*(max(v,known-closing,0)+2mps-growth);v=max(3mps,measured-speed)',
    'control_bound_applies_only_to_current_in_path_far_vehicle': True,
}

def vehicle_range_interval(item, calibration, config):
    """Bound the projected ground intersection, allowing an unbounded far end.

    Ground rays crossing the horizon make the *upper* distance unbounded. They
    do not imply that the nearest possible intersection is at the ego bumper.
    All finite corner projections are retained. This is a bound under the fixed
    interval projection model, not a sensor-measured distance or road guarantee.
    """
    if item.get('kind') != 'vehicle':
        return None
    box = item.get('bbox_xyxy')
    w, h = calibration['width'], calibration['height']
    if not isinstance(box, list) or len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x0, y0, x1, y1 = box
    if x0 <= 1. or x1 >= w-1. or y1 >= h-1.:
        return None  # a clipped footpoint has no bounded pixel interval
    u, v = .5*(x0+x1), y1
    points = [_ground_point(u+du, v+dv, calibration, height_shift=dh, pitch_shift=dp)
              for du, dv, dh, dp in itertools.product(
                  (-config.bbox_footpoint_uncertainty_px, config.bbox_footpoint_uncertainty_px),
                  (-config.bbox_footpoint_uncertainty_px, config.bbox_footpoint_uncertainty_px),
                  (-config.camera_height_uncertainty_m, config.camera_height_uncertainty_m),
                  (-config.camera_pitch_uncertainty_deg, config.camera_pitch_uncertainty_deg))]
    finite = [p[0]-config.front_bumper_x_m for p in points if p is not None]
    if not finite:
        return None
    return {'near_m': min(finite), 'far_m': max(finite) if all(p is not None for p in points) else None,
            'nominal_m': item.get('distance_m'),
            'nominal_ground_valid': item.get('ground_valid'),
            'source': 'RGB-bbox-footpoint+fixed-mount-bounded-level-ground-projection',
            'bound_semantics': 'one-sided-longitudinal;far-None-means-unbounded',
            'conditioned_on_bounded_mount_and_level_ground': True,
            'GT_inputs': False}


def _nearest(item):
    bound = item.get('conservative_nearest_distance_m')
    if bound is not None:
        return bound
    if item.get('distance_m') is None or item.get('distance_uncertainty_m') is None:
        return None
    return item['distance_m']-item['distance_uncertainty_m']


class DirectedRangeBridge(RoadsideBridge):
    def __init__(self, *, nominal_calibration, front_bumper_x_m=2.4251085, ego_half_width_m=.95):
        self.calibration = validate_nominal_calibration(nominal_calibration)
        self.range_config = TrafficPerceptionConfig(front_bumper_x_m=front_bumper_x_m,
                                                    ego_half_width_m=ego_half_width_m)
        super().__init__(front_bumper_x_m=front_bumper_x_m, ego_half_width_m=ego_half_width_m)

    def update(self, raw, prediction, observation):
        enriched = copy.deepcopy(raw)
        for item in enriched['objects']:
            interval = vehicle_range_interval(item, self.calibration, self.range_config)
            if interval is not None:
                item['range_interval'] = interval
                speed=max(3.,observation.ego_speed_mps)
                closing=item.get('closing_speed_mps')
                closing=speed if closing is None else max(speed,closing,0.)
                minimum=2.+speed*speed/4.+.2*speed+.6*(closing+2.)
                interval['minimum_control_near_m']=minimum
                if item.get('in_path') and item['confidence']>=.5 and interval['near_m']>=minimum:
                    item['conservative_nearest_distance_m'] = interval['near_m']
        if self.time is not None:
            self.travel += .5*(self.speed+observation.ego_speed_mps)*(observation.timestamp-self.time)
        self.observation = observation
        self.current_points = prediction.get('waypoints_ego_m') if prediction.get('status') == 'PASS' else None
        result, notes = self._memory_update(enriched, prediction, observation)
        notes['directed_range'] = {'version': DIRECTED_RANGE_CONFIG['version'], 'GT_inputs': False,
            'nominal_fields_preserved': True,
            'objects_with_explicit_lower_bound': sum(o.get('conservative_nearest_distance_m') is not None for o in result['objects'])}
        return result, notes

    # Causal memory update with explicit lower-bound risk tests.
    def _memory_update(self,raw,prediction,observation):
        if raw.get('GT_inputs') is not False or raw['frame']!=observation.frame or abs(raw['timestamp']-observation.timestamp)>1e-7:
            raise ValueError('RGB-memory-input-provenance-or-age-mismatch')
        result=copy.deepcopy(raw);now=observation.timestamp;speed=observation.ego_speed_mps
        dt=0. if self.time is None else now-self.time
        if self.time is not None and (dt<=0 or abs(dt-.05)>1e-6):raise ValueError('RGB-memory-nonconsecutive-time')
        travel=0. if self.speed is None else .5*(self.speed+speed)*dt
        self.time=now;self.speed=speed
        if self.sign_served:self.sign_travel+=travel
        notes={'GT_inputs':False,'raw_result_modified':False,'line':None,'objects':[],'unresolved_lost_tracks':[],
               'line_maximum_age_seconds':2.,'object_propagation_maximum_age_seconds':.5,
               'positive_out_of_path_clear_hold_seconds':.2}
        if self.line:
            self.line['distance_m']-=travel
            self.line['uncertainty_m']+=.1*dt+.05*travel
        if raw.get('stop_line_distance_m') is not None and raw.get('stop_line_confidence',0.)>=.5:
            self.line={'distance_m':raw['stop_line_distance_m'],
                'uncertainty_m':raw.get('stop_line',{}).get('ground_estimate',{}).get('distance_uncertainty_m',.5),
                'confidence':raw['stop_line_confidence'],'last_seen_frame':observation.frame,'last_seen_timestamp':now}
        elif self.line:
            age=now-self.line['last_seen_timestamp']
            if age<=2.+1e-9:
                result['stop_line_distance_m']=self.line['distance_m'];result['stop_line_confidence']=self.line['confidence']
                result.setdefault('stop_line',{}).update(observed=False,status='causal-RGB-memory-estimate',
                    ground_estimate={'distance_uncertainty_m':self.line['uncertainty_m']})
            else:self.line=None
        if self.line:notes['line']={**self.line,'age_seconds':now-self.line['last_seen_timestamp'],
            'source':'last-actual-RGB-line-minus-measured-ego-travel;not-current-observation'}
        if raw.get('stop_required'):
            if self.sign is None:self.sign_counter+=1;self.sign=f'RGB-stop-context-{self.sign_counter}';self.sign_served=False
            self.sign_last_seen=now;result['stop_sign_id']=self.sign
        elif (self.sign and self.sign_served and self.sign_last_seen is not None and now-self.sign_last_seen>=.5
                and self.sign_travel>=5.):
            self.sign=None;self.sign_served=False;self.sign_travel=0.
        points=prediction.get('waypoints_ego_m') if prediction.get('status')=='PASS' else None
        observed=set();objects=[]
        for original in raw['objects']:
            item=copy.deepcopy(original);identifier=item['track_id'];observed.add(identifier)
            in_path,method=self._path_membership(item,points);item['in_path']=in_path
            old=self.tracks.get(identifier)
            if in_path and item['confidence']>=.5:
                nearest=_nearest(item)
                relevant=nearest is None or nearest<=2.+speed*speed/4.+.2*speed
                self.tracks[identifier]={'item':copy.deepcopy(item),'last_seen':now,'last_seen_frame':observation.frame,
                                        'clear_since':None,'threat_relevant':relevant}
            elif old:
                positive_clear=not in_path and item.get('ground_valid') and item['confidence']>=.5
                if positive_clear:
                    if old['clear_since'] is None:old['clear_since']=now
                    if now-old['clear_since']>=.2-1e-9:del self.tracks[identifier]
                    else:item['in_path']=True
                else:
                    old['clear_since']=None;item['in_path']=True
            notes['objects'].append({'track_id':identifier,'membership_source':method,'source':'current-RGB'})
            objects.append(item)
        for identifier,memory in list(self.tracks.items()):
            if identifier in observed:continue
            age=now-memory['last_seen'];item=copy.deepcopy(memory['item'])
            # A missing ground footpoint is an explicit unknown, not a numeric
            # estimate. Keep None until the typed boundary supplies its declared
            # unknown-range uncertainty; never invent a confident range.
            closing=item.get('closing_speed_mps');closing=speed if closing is None else closing
            if item.get('distance_m') is not None:item['distance_m']-=closing*age
            if item.get('distance_uncertainty_m') is not None:item['distance_uncertainty_m']+=2.*age
            if item.get('lateral_uncertainty_m') is not None:item['lateral_uncertainty_m']+=age
            if item.get('conservative_nearest_distance_m') is not None:
                # A missing vehicle cannot gain clearance from a noisy receding
                # derivative. Bound at least a stationary object plus growth.
                decrease=(max(speed,closing,0.)+2.)*age
                item['conservative_nearest_distance_m']-=decrease
                item['range_interval']=copy.deepcopy(item['range_interval'])
                item['range_interval']['near_m']=item['conservative_nearest_distance_m']
                item['range_interval']['far_m']=None
                item['range_interval']['source']='last-RGB-interval-conservative-propagation'
                item['range_interval']['nominal_m']=item.get('distance_m')
            nearest=_nearest(item)
            # Risk can grow during a dropout. Reassess the propagated lower
            # bound before expiry and latch danger until positive clearance.
            if nearest is None or nearest<=2.+speed*speed/4.+.2*speed:
                memory['threat_relevant']=True
            if age<=.5+1e-9:
                item['in_path']=True;objects.append(item)
                notes['objects'].append({'track_id':identifier,'source':'last-RGB-track-propagated',
                    'last_seen_timestamp':memory['last_seen'],'last_seen_frame':memory['last_seen_frame'],'age_seconds':age})
            elif memory['threat_relevant']:
                # Empty detections are not positive evidence that a nearby
                # pedestrian vanished safely. Hold uncertainty until a real
                # observation of this track establishes lateral clearance.
                result['perception_valid']=False;notes['unresolved_lost_tracks'].append(identifier)
            else:del self.tracks[identifier]
        result['objects']=objects
        return result,notes
