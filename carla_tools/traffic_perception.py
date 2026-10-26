"""RGB traffic observations with explicit monocular uncertainty.

World actors, maps, simulator labels and vehicle world transforms are deliberately
absent from the input API. Ground estimates assume a level road and the supplied
nominal, static camera mounting. They are estimates, not range measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import colorsys
import hashlib
import math
import time
from typing import Any

SSD_WEIGHTS_URL = 'https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
SSD_WEIGHTS_SHA256_PREFIX = 'a79551df'
FRCNN_WEIGHTS_URL = 'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
FRCNN_WEIGHTS_SHA256_PREFIX = 'fb6a3cc7'
COCO_TRAFFIC_LABELS = {1:'person',3:'car',6:'bus',8:'truck',10:'traffic light',13:'stop sign'}

def _require(value: bool, code: str) -> None:
    if not value: raise ValueError(code)

def _finite(value: Any) -> bool:
    return type(value) in (int,float) and math.isfinite(value)

@dataclass(frozen=True)
class RGBObservation:
    rgb: bytes
    width: int
    height: int
    frame: int
    timestamp: float
    ego_speed_mps: float

    def __post_init__(self):
        _require(type(self.width) is int and type(self.height) is int and
            1<=self.width<=4096 and 1<=self.height<=2160,'traffic-invalid-RGB-size')
        _require(type(self.rgb) is bytes and len(self.rgb)==self.width*self.height*3,'traffic-invalid-RGB-bytes')
        _require(type(self.frame) is int and self.frame>=0 and _finite(self.timestamp) and self.timestamp>=0,
            'traffic-invalid-frame-time')
        _require(_finite(self.ego_speed_mps) and 0<=self.ego_speed_mps<=100,'traffic-invalid-ego-speed')

@dataclass(frozen=True)
class TrafficPerceptionConfig:
    score_threshold: float = .35
    minimum_signal_confidence: float = .55
    front_bumper_x_m: float = 2.3
    ego_half_width_m: float = .95
    path_margin_m: float = .35
    maximum_ground_distance_m: float = 60.
    bbox_footpoint_uncertainty_px: float = 3.
    camera_height_uncertainty_m: float = .15
    camera_pitch_uncertainty_deg: float = 2.
    track_max_age_s: float = .5
    track_min_iou: float = .1
    stop_line_detection: bool = True

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key=='stop_line_detection':_require(type(value) is bool,'traffic-invalid-config');continue
            _require(_finite(value) and value>=0,'traffic-invalid-config')
        _require(0<self.score_threshold<=1 and 0<self.minimum_signal_confidence<=1
            and 0<=self.track_min_iou<=1 and self.track_max_age_s>0,'traffic-invalid-config')

def validate_nominal_calibration(calibration: dict) -> dict:
    """Accept only static pinhole camera calibration, never a world pose."""
    _require(type(calibration) is dict and set(calibration)=={'width','height','intrinsics','camera_to_actor_matrix'},
        'traffic-static-calibration-fields-required')
    w,h=calibration['width'],calibration['height'];k=calibration['intrinsics'];m=calibration['camera_to_actor_matrix']
    _require(type(w) is int and type(h) is int and w>0 and h>0 and type(k) is dict
        and set(k)=={'fx','fy','cx','cy'} and all(_finite(v) for v in k.values())
        and k['fx']>0 and k['fy']>0,'traffic-invalid-camera-intrinsics')
    _require(type(m) in (list,tuple) and len(m)==4 and all(type(r) in (list,tuple) and len(r)==4 for r in m)
        and all(_finite(v) for r in m for v in r),'traffic-invalid-static-mount')
    _require(all(abs(m[3][i]-(1. if i==3 else 0.))<1e-6 for i in range(4))
        and m[2][3]>.1,'traffic-nonrigid-or-below-ground-camera')
    _require(all(abs(sum(m[k][i]*m[k][j] for k in range(3))-(1. if i==j else 0.))<1e-5
        for i in range(3) for j in range(3)),'traffic-nonrigid-static-mount')
    determinant=(m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
        -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    _require(abs(determinant-1.)<1e-5,'traffic-reflected-static-mount')
    return {'width':w,'height':h,'intrinsics':dict(k),'camera_to_actor_matrix':[list(r) for r in m]}

def _ground_point(u:float,v:float,calibration:dict,*,height_shift=0.,pitch_shift=0.) -> tuple[float,float]|None:
    k=calibration['intrinsics'];m=calibration['camera_to_actor_matrix']
    local=[1.,(u-k['cx'])/k['fx'],-(v-k['cy'])/k['fy']]
    ray=[sum(m[r][c]*local[c] for c in range(3)) for r in range(3)]
    # Pitch uncertainty rotates the nominal ray around actor y, without using
    # any actual actor/camera pose from the simulator.
    angle=math.radians(pitch_shift);cs,sn=math.cos(angle),math.sin(angle)
    ray=[cs*ray[0]+sn*ray[2],ray[1],-sn*ray[0]+cs*ray[2]]
    height=m[2][3]+height_shift
    if ray[2]>=-.02 or height<=0:return None
    factor=-height/ray[2]
    x=m[0][3]+factor*ray[0];y=m[1][3]+factor*ray[1]
    return (x,y) if math.isfinite(x) and math.isfinite(y) and factor>0 else None

def ground_estimate(u:float,v:float,calibration:dict,config:TrafficPerceptionConfig,*,truncated=False)->dict:
    result={'valid':False,'distance_m':None,'lateral_y_m':None,'distance_uncertainty_m':None,
        'lateral_uncertainty_m':None,'reason':'unobservable_ground_footpoint',
        'source':'RGB-footpoint+nominal-level-ground-camera;not-depth-sensor',
        'distance_origin':'signed-forward-from-static-front-bumper'}
    if truncated:return dict(result,reason='image_border_truncation')
    nominal=_ground_point(u,v,calibration)
    if nominal is None:return result
    x,y=nominal;distance=x-config.front_bumper_x_m
    if distance>config.maximum_ground_distance_m or x<=0:return dict(result,reason='outside_ground_estimate_range')
    alternatives=[]
    for du in (-config.bbox_footpoint_uncertainty_px,config.bbox_footpoint_uncertainty_px):
        for dv in (-config.bbox_footpoint_uncertainty_px,config.bbox_footpoint_uncertainty_px):
            for dh in (-config.camera_height_uncertainty_m,config.camera_height_uncertainty_m):
                for dp in (-config.camera_pitch_uncertainty_deg,config.camera_pitch_uncertainty_deg):
                    point=_ground_point(u+du,v+dv,calibration,height_shift=dh,pitch_shift=dp)
                    if point is None:return dict(result,reason='uncertainty_interval_crosses_horizon')
                    alternatives.append(point)
    return dict(result,valid=True,distance_m=distance,lateral_y_m=y,
        distance_uncertainty_m=max(.25,max(abs(p[0]-x) for p in alternatives)),
        lateral_uncertainty_m=max(.1,max(abs(p[1]-y) for p in alternatives)),reason=None)

class SSDLiteBackend:
    """Strict local checkpoint loading; downloading is the caller's job."""
    def __init__(self,weights_path:str|Path,*,expected_sha256:str,device='cpu',architecture='ssdlite'):
        _require(architecture in ('ssdlite','fasterrcnn_mobilenet'),'traffic-unsupported-official-architecture')
        prefix=SSD_WEIGHTS_SHA256_PREFIX if architecture=='ssdlite' else FRCNN_WEIGHTS_SHA256_PREFIX
        url=SSD_WEIGHTS_URL if architecture=='ssdlite' else FRCNN_WEIGHTS_URL
        digest=hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
        _require(type(expected_sha256) is str and len(expected_sha256)==64 and digest==expected_sha256
            and digest.startswith(prefix),'traffic-official-weight-SHA')
        _require(device in ('cpu','cuda'),'traffic-invalid-inference-device')
        import torch
        import torchvision
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
        _require(torch.__version__.split('+')[0]=='2.7.1' and torchvision.__version__.split('+')[0]=='0.22.1',
            'traffic-pinned-torch-vision-required')
        categories=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.meta['categories']
        _require(all(categories[key]==value for key,value in COCO_TRAFFIC_LABELS.items()),'traffic-COCO-category-binding')
        self.torch=torch;self.device=device
        if architecture=='ssdlite':self.model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None)
        else:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
            self.model=fasterrcnn_mobilenet_v3_large_fpn(weights=None,weights_backbone=None)
        self.model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True)
        self.model.to(device).eval();self.weights_sha256=digest
        self.identity={'architecture':architecture,'weights':'COCO_V1',
            'weights_url':url,'weights_sha256':digest,'weight_bytes':Path(weights_path).stat().st_size,
            'torch':torch.__version__,'torchvision':torchvision.__version__,'device':device,'trained_locally':False}
    def predict(self,observation:RGBObservation)->list[dict]:
        torch=self.torch
        # Writable bytearray avoids a non-writable tensor alias warning.
        tensor=torch.frombuffer(bytearray(observation.rgb),dtype=torch.uint8).reshape(observation.height,observation.width,3)
        tensor=tensor.permute(2,0,1).to(device=self.device,dtype=torch.float32)/255.
        with torch.inference_mode():output=self.model([tensor])[0]
        boxes=output['boxes'].detach().cpu().tolist();scores=output['scores'].detach().cpu().tolist();labels=output['labels'].detach().cpu().tolist()
        return [{'bbox_xyxy':box,'score':score,'label_id':label} for box,score,label in zip(boxes,scores,labels)]
    def synchronize(self):
        if self.device=='cuda':self.torch.cuda.synchronize()

class FasterRCNNMobileNetBackend(SSDLiteBackend):
    def __init__(self,weights_path:str|Path,*,expected_sha256:str,device='cpu'):
        super().__init__(weights_path,expected_sha256=expected_sha256,device=device,architecture='fasterrcnn_mobilenet')

def _iou(a,b):
    ix=max(0.,min(a[2],b[2])-max(a[0],b[0]));iy=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection=ix*iy;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.

class CausalObjectTracker:
    def __init__(self,config:TrafficPerceptionConfig):self.config=config;self.reset()
    def reset(self):self.tracks={};self.sequence=0
    def update(self,objects:list[dict],timestamp:float)->list[dict]:
        self.tracks={key:value for key,value in self.tracks.items() if 0<timestamp-value['timestamp']<=self.config.track_max_age_s}
        unmatched=set(self.tracks);result=[]
        for item in sorted(objects,key=lambda r:-r['confidence']):
            matches=[(key,_iou(item['bbox_xyxy'],self.tracks[key]['bbox'])) for key in unmatched
                if self.tracks[key]['label']==item['label']]
            key,overlap=max(matches,key=lambda p:p[1],default=(None,0.))
            if overlap<self.config.track_min_iou:key=None
            if key is None:self.sequence+=1;key=f'traffic_{self.sequence:06d}';past=None
            else:past=self.tracks[key];unmatched.remove(key)
            closing=None;history=[]
            if past and item['distance_m'] is not None and past['distance_m'] is not None:
                dt=timestamp-past['timestamp']
                if dt>=.025:
                    instantaneous=(past['distance_m']-item['distance_m'])/dt
                    if abs(instantaneous)<=30.:
                        history=(past.get('closing_history',[])+[instantaneous])[-3:]
                        if len(history)>=2:closing=sorted(history)[len(history)//2]
            item=dict(item,track_id=key,closing_speed_mps=closing)
            self.tracks[key]={'bbox':item['bbox_xyxy'],'label':item['label'],'distance_m':item['distance_m'],
                'timestamp':timestamp,'closing_history':history}
            result.append(item)
        return result

def signal_color(rgb:bytes,width:int,height:int,bbox:list[float])->dict:
    counts={'red':0,'yellow':0,'green':0};total=0
    x0,y0,x1,y1=[int(v) for v in bbox]
    for y in range(max(0,y0),min(height,y1)):
        for x in range(max(0,x0),min(width,x1)):
            index=(y*width+x)*3;r,g,b=[v/255. for v in rgb[index:index+3]]
            hue,saturation,value=colorsys.rgb_to_hsv(r,g,b)
            if saturation<.5 or value<.45:continue
            degrees=hue*360.;total+=1
            if degrees<=20. or degrees>=340.:counts['red']+=1
            elif 35.<=degrees<=75.:counts['yellow']+=1
            elif 80.<=degrees<=165.:counts['green']+=1
    color=max(counts,key=counts.get);count=counts[color];confidence=count/max(1,total)
    return {'color':color if count>=3 and confidence>=.55 else 'unknown',
        'confidence':confidence if count>=3 else 0.,'colored_pixels':total,'counts':counts,
        'source':'HSV-only-within-RGB-detected-traffic-light-box'}

def detect_stop_line(observation:RGBObservation,calibration:dict,config:TrafficPerceptionConfig)->dict:
    """A paint-bar candidate, explicitly distinct from a map stop waypoint.

    This inexpensive detector cannot establish the legal meaning of a white
    stripe. Multiple close stripes are ambiguous (for example a crosswalk).
    """
    result={'status':'not_observed','distance_m':None,'confidence':0.,'uv_segment':None,
        'source':'RGB-white-paint-horizontal-bar;no-map-waypoint','legal_stopline_verified':False}
    if not config.stop_line_detection:return result
    import numpy as np
    import cv2
    image=np.frombuffer(observation.rgb,dtype=np.uint8).reshape(observation.height,observation.width,3)
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);mask=cv2.inRange(hsv,(0,0,175),(179,65,255))
    h,w=observation.height,observation.width;mask[:int(h*.5)]=0;mask[int(h*.96):]=0
    mask[:,:int(w*.2)]=0;mask[:,int(w*.8):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,max(9,int(w*.08))),dtype=np.uint8))
    count,_,stats,centroids=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for index in range(1,count):
        x,y,bw,bh,area=stats[index];cx,cy=centroids[index]
        if bw>=w*.28 and 2<=bh<=h*.055 and bw/max(1,bh)>=6. and area/(bw*bh)>=.55:
            estimate=ground_estimate(float(cx),float(cy),calibration,config)
            if estimate['valid']:candidates.append((int(x),int(y),int(bw),int(bh),estimate))
    if not candidates:return result
    candidates.sort(key=lambda row:row[1],reverse=True)
    x,y,bw,bh,estimate=candidates[0]
    if len(candidates)>1:return dict(result,status='ambiguous_paint_stripes')
    return dict(result,status='paint_bar_candidate',distance_m=estimate['distance_m'],confidence=.55,
        uv_segment=[[x,y+bh/2],[x+bw,y+bh/2]],ground_estimate=estimate)

class TrafficPerception:
    def __init__(self,backend,nominal_calibration:dict,config:TrafficPerceptionConfig|None=None):
        self.backend=backend;self.calibration=validate_nominal_calibration(nominal_calibration)
        self.config=config or TrafficPerceptionConfig();self.tracker=CausalObjectTracker(self.config)
        self.sign_tracker=CausalObjectTracker(self.config);self.reset()
    def reset(self):
        self.previous_frame=None;self.previous_timestamp=None;self.tracker.reset();self.sign_tracker.reset()
    def process(self,observation:RGBObservation)->dict:
        _require(type(observation) is RGBObservation,'traffic-RGB-only-observation-required')
        _require((observation.width,observation.height)==(self.calibration['width'],self.calibration['height']),
            'traffic-nominal-calibration-size-mismatch')
        _require(self.previous_frame is None or (observation.frame>self.previous_frame and observation.timestamp>self.previous_timestamp),
            'traffic-noncausal-frame-or-timestamp')
        started=time.perf_counter()
        if hasattr(self.backend,'synchronize'):self.backend.synchronize()
        detections=self.backend.predict(observation);objects=[];signals=[];signs=[];raw=[]
        for detection in detections:
            label_id=detection['label_id'];score=detection['score'];bbox=detection['bbox_xyxy']
            _require(type(label_id) is int and _finite(score) and 0<=score<=1 and type(bbox) is list and len(bbox)==4
                and all(_finite(v) for v in bbox),'traffic-invalid-backend-output')
            if label_id not in COCO_TRAFFIC_LABELS or score<self.config.score_threshold:continue
            x0,y0,x1,y1=bbox
            if not (0<=x0<x1<=observation.width and 0<=y0<y1<=observation.height):continue
            label=COCO_TRAFFIC_LABELS[label_id];item={'label':label,'label_id':label_id,'confidence':score,'bbox_xyxy':bbox}
            raw.append(dict(item))
            if label=='traffic light':
                color=signal_color(observation.rgb,observation.width,observation.height,bbox)
                if color['confidence']<self.config.minimum_signal_confidence:color['color']='unknown'
                signals.append(dict(item,**color));continue
            if label=='stop sign':signs.append(item);continue
            ground=ground_estimate((x0+x1)/2,y1,self.calibration,self.config,truncated=y1>=observation.height-1 or x0<=1 or x1>=observation.width-1)
            item.update(kind='pedestrian' if label=='person' else 'vehicle',ground_valid=ground['valid'],
                distance_m=ground['distance_m'],lateral_y_m=ground['lateral_y_m'],
                distance_uncertainty_m=ground['distance_uncertainty_m'],lateral_uncertainty_m=ground['lateral_uncertainty_m'],
                ground_estimate=ground,
                in_path=bool(abs(ground['lateral_y_m'])<=self.config.ego_half_width_m+self.config.path_margin_m+ground['lateral_uncertainty_m'])
                    if ground['valid'] else x0<observation.width*.65 and x1>observation.width*.35)
            objects.append(item)
        objects=self.tracker.update(objects,observation.timestamp)
        signs=self.sign_tracker.update([dict(s,distance_m=None) for s in signs],observation.timestamp)
        # Several physical signal heads can be visible. Never assume that a
        # green on one approach cancels a visible red/unknown on another.
        selected=next((s for color in ('red','yellow','unknown','green') for s in signals if s['color']==color),None)
        line=detect_stop_line(observation,self.calibration,self.config)
        if hasattr(self.backend,'synchronize'):self.backend.synchronize()
        result={'schema_version':1,'frame':observation.frame,'timestamp':observation.timestamp,
            'objects':objects,'detections':raw,'signals':signals,'stop_signs':signs,
            'signal':selected['color'] if selected else 'none','signal_confidence':selected['confidence'] if selected else 0.,
            'stop_line':line,'stop_line_distance_m':line['distance_m'],'stop_line_confidence':line['confidence'],
            'stop_required':bool(signs),'stop_sign_id':signs[0]['track_id'] if signs else None,'perception_valid':True,
            'GT_inputs':False,'input_contract':'RGB-bytes/frame/timestamp/ego-speed;nominal-static-calibration-only',
            'in_path_contract':'nominal-ego-forward-corridor;not-route-or-lane-aware',
            'latency_ms':(time.perf_counter()-started)*1000.}
        self.previous_frame=observation.frame;self.previous_timestamp=observation.timestamp
        return result

def to_traffic_observation(result:dict):
    """Narrow the image-derived result to the decision module's frozen types."""
    from .traffic_behavior import TrafficObservation,TrackedObject
    _require(result.get('GT_inputs') is False,'traffic-camera-result-required')
    objects=tuple(TrackedObject(track_id=item['track_id'],kind=item['kind'],distance_m=item['distance_m'],
        lateral_y_m=item['lateral_y_m'],uncertainty_m=item['distance_uncertainty_m'] if item['ground_valid'] else 60.,
        lateral_uncertainty_m=item['lateral_uncertainty_m'] if item['ground_valid'] else 3.,
        closing_speed_mps=item['closing_speed_mps'],in_path=item['in_path'],confidence=item['confidence'],
        conservative_nearest_distance_m=item.get('conservative_nearest_distance_m'))
        for item in result['objects'])
    return TrafficObservation(frame=result['frame'],timestamp=result['timestamp'],objects=objects,
        signal=result['signal'],signal_confidence=result['signal_confidence'],
        stop_line_distance_m=result['stop_line_distance_m'],stop_line_confidence=result['stop_line_confidence'],
        stop_required=result['stop_required'],perception_valid=result['perception_valid'],
        stop_sign_id=result.get('stop_sign_id'),
        stop_line_uncertainty_m=result['stop_line'].get('ground_estimate',{}).get('distance_uncertainty_m',.5))
