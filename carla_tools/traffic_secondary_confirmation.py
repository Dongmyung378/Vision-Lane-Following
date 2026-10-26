"""Explicit secondary-person agreement for fixed-camera simulation experiments.

Importing this module loads no neural model, Torch runtime or GPU. Construct
ResNet50V2Backend with an explicitly chosen device and an already downloaded,
hash-verified official weight file, or inject a recorded backend for CPU replay.

The scheduler examines only initially weak, far primary person observations.
It never suppresses a current primary person, adds a secondary object, changes
signals, authorizes memory retirement or supplies a physical absence proof.
Scores and finite ground intervals do not certify a visible person's range.
"""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import math
import time
from .traffic_perception import _ground_point,validate_nominal_calibration

VERSION='primary-ROI-positive-secondary-person-confirmation-v1'
PRIMARY_SHA='fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533'
SECONDARY_SHA='dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'


def require(value,reason):
    if not value:raise ValueError(reason)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def overlap(a,b):
    cross=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-cross
    return cross/union if union>0 else 0.
def valid_box(box,width,height):return type(box)is list and len(box)==4 and all(type(v)in(int,float) and math.isfinite(v) for v in box) and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height


def implementation_source_sha256():
    """Identify this installed optional implementation without loading a model."""
    with Path(__file__).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class SecondaryPersonConfirmationScheduler:
    """Schedule bounded positive re-observations, never negative clearance.

    Call once for every fixed 0.05-second observation, including frames without
    people. Primary IDs must retain their lifecycle identity until reset. The
    nominal camera and static front geometry must match the injected bridge.
    A finite ground footprint is only a scheduling heuristic under occlusion.
    """
    def __init__(self,backend,nominal_calibration,front_bumper_x_m=2.4251085,primary_source_identity=None):
        self.backend=backend;self.calibration=validate_nominal_calibration(nominal_calibration);self.front=front_bumper_x_m
        require(type(primary_source_identity)is dict and type(backend.identity)is dict,'explicit-primary-and-secondary-source-identities')
        self.primary=copy.deepcopy(primary_source_identity);self.secondary=copy.deepcopy(backend.identity)
        require(type(self.front)in(int,float) and math.isfinite(self.front),'secondary-static-front-bumper')
        require(self.primary.get('architecture')=='fasterrcnn_mobilenet' and self.primary.get('weights_sha256')==PRIMARY_SHA,'fixed-primary-MobileNet-source')
        require(self.secondary.get('architecture')=='fasterrcnn_resnet50_fpn_v2' and self.secondary.get('weights_sha256')==SECONDARY_SHA
            and self.secondary.get('implementation_source_sha256')==implementation_source_sha256(),'fixed-secondary-official-source')
        self.configuration=dict(version=VERSION,minimum_primary_score=.5,maximum_primary_score_exclusive=.85,minimum_secondary_score=.5,minimum_IoU=.3,
            minimum_one_sided_ground_near_m_exclusive=8.,maximum_observed_person_sightings=2,maximum_calls_per_track=2,
            one_shared_secondary_call_per_frame=True,positive_confirmation_only=True,GT_inputs=False,
            nominal_calibration_sha256=digest(self.calibration),front_bumper_x_m=self.front)
        self.reset()
    def reset(self):self.tracks={};self.previous=None;self.scheduled_calls=0;self.fresh_NN_calls=0
    def _footprint(self,item):
        b=item.get('bbox_xyxy')
        if not item.get('ground_valid') or not valid_box(b,self.calibration['width'],self.calibration['height']):return None
        if b[0]<=1 or b[2]>=self.calibration['width']-1 or b[3]>=self.calibration['height']-1:return None
        points=[_ground_point((b[0]+b[2])*.5+du,b[3]+dv,self.calibration,height_shift=dh,pitch_shift=dp)
            for du,dv,dh,dp in itertools.product((-3.,3.),(-3.,3.),(-.15,.15),(-2.,2.))]
        if any(p is None or not all(math.isfinite(v) for v in p) for p in points):return None
        return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
    def update(self,raw,observation):
        require(raw.get('GT_inputs')is False and raw['frame']==observation.frame and abs(raw['timestamp']-observation.timestamp)<=1e-7,'secondary-current-RGB-source')
        require(type(observation.rgb)is bytes and len(observation.rgb)==observation.width*observation.height*3
            and (observation.width,observation.height)==(self.calibration['width'],self.calibration['height']),'secondary-original-RGB-contract')
        if self.previous is not None:require(observation.frame>self.previous[0] and abs(observation.timestamp-self.previous[1]-.05)<=1e-6,'secondary-consecutive-time')
        self.previous=(observation.frame,observation.timestamp);image_sha=hashlib.sha256(observation.rgb).hexdigest();eligible=[];skips=[];seen_ids=set()
        for item in raw['objects']:
            if item.get('kind')!='pedestrian' or item.get('confidence',0)<.5:continue
            key=item['track_id'];require(key not in seen_ids,'secondary-primary-person-ID-unique');seen_ids.add(key)
            track=self.tracks.setdefault(key,dict(seen=0,strong=False,calls=0,positive_confirmed=False));track['seen']+=1;foot=self._footprint(item)
            track['strong']|=(track['seen']>=3 or item['confidence']>=.85 or foot is None or foot[0]-self.front<=8.)
            reasons=[]
            if raw.get('perception_valid')is not True:reasons.append('primary-perception-invalid')
            if track['strong']:reasons.append('strong-near-unknown-or-third-observation')
            if not item.get('in_path'):reasons.append('outside-primary-nominal-corridor')
            if track['calls']>=2:reasons.append('track-two-call-budget')
            if track['positive_confirmed']:reasons.append('positive-secondary-already-confirmed')
            if reasons:skips.append(dict(primary_track_id=key,observation_count=track['seen'],reasons=reasons));continue
            eligible.append((item,track,foot))
        result=dict(version=VERSION,frame=observation.frame,timestamp=observation.timestamp,RGB_sha256=image_sha,GT_inputs=False,
            primary_source_identity=copy.deepcopy(self.primary),secondary_source_identity=copy.deepcopy(self.secondary),
            configuration_sha256=digest(self.configuration),new_NN_calls=0,scheduled_inference_calls=0,inference_receipt=None,confirmations=[],skips=skips)
        if not eligible:return result
        if hasattr(self.backend,'synchronize'):self.backend.synchronize()
        began=time.perf_counter();detections=self.backend.predict(observation)
        if hasattr(self.backend,'synchronize'):self.backend.synchronize()
        elapsed=(time.perf_counter()-began)*1000
        require(type(detections)is list,'secondary-list-of-original-detections')
        for item in detections:
            require(type(item.get('label_id'))is int and type(item.get('score'))in(int,float) and math.isfinite(item['score']) and 0<=item['score']<=1
                and valid_box(item.get('bbox_xyxy'),observation.width,observation.height),'secondary-finite-original-detection')
        self.scheduled_calls+=1;recorded=bool(getattr(self.backend,'recorded_only',False));self.fresh_NN_calls+=not recorded
        receipt=dict(version='full-secondary-RGB-inference-receipt-v1',frame=observation.frame,timestamp=observation.timestamp,RGB_sha256=image_sha,
            width=observation.width,height=observation.height,secondary_source_identity=copy.deepcopy(self.secondary),raw_detections=copy.deepcopy(detections),
            synchronized_pipeline_ms=elapsed,source_RGB_only=True,GT_inputs=False,fresh_NN=not recorded)
        previous=getattr(self.backend,'last_recorded_receipt',None)
        if previous is not None:
            require(previous['frame']==receipt['frame'] and previous['RGB_sha256']==image_sha and previous['raw_detections']==detections,'secondary-recorded-receipt-output-binding')
            receipt=copy.deepcopy(previous)
        result.update(new_NN_calls=int(not recorded),scheduled_inference_calls=1,inference_receipt=receipt)
        people=[d for d in detections if d['label_id']==1 and d['score']>=.5]
        for item,track,foot in eligible:
            track['calls']+=1;matches=[dict(bbox_xyxy=copy.deepcopy(d['bbox_xyxy']),score=d['score'],label_id=1,IoU=overlap(item['bbox_xyxy'],d['bbox_xyxy']))
                for d in people if overlap(item['bbox_xyxy'],d['bbox_xyxy'])>=.3]
            result['confirmations'].append(dict(primary_track_id=item['track_id'],primary_bbox_xyxy=copy.deepcopy(item['bbox_xyxy']),primary_score=item['confidence'],
                observation_count=track['seen'],secondary_person_confirmation=bool(matches),matched_secondary_persons=matches,
                fixed_mount_footprint_xy_bounds=foot,one_sided_near_m=foot[0]-self.front,scheduled_calls_for_track=track['calls']))
            track['positive_confirmed']|=bool(matches)
        return result


class RecordedSecondaryBackend:
    """Source-bound CPU replay adapter; no neural model is constructed."""
    recorded_only=True
    def __init__(self,receipts,identity,*,timestamp_by_frame=None):
        self.receipts=copy.deepcopy(receipts);self.identity=copy.deepcopy(identity);self.times=timestamp_by_frame or {};self.calls=0;self.last_recorded_receipt=None
        require(self.identity.get('weights_sha256')==SECONDARY_SHA,'recorded-secondary-official-weights')
    def synchronize(self):pass
    def predict(self,observation):
        require(observation.frame in self.receipts,'NOOBS-secondary-receipt-missing')
        source=self.receipts[observation.frame];rgbsha=hashlib.sha256(observation.rgb).hexdigest()
        require(source['frame']==observation.frame and source['RGB_sha256']==rgbsha,'recorded-secondary-same-frame-RGB')
        timestamp=source.get('timestamp',self.times.get(observation.frame))
        require(timestamp is not None and abs(timestamp-observation.timestamp)<=1e-7,'recorded-secondary-source-timestamp')
        require(source.get('weights_sha256',source.get('secondary_source_identity',{}).get('weights_sha256'))==SECONDARY_SHA,'recorded-secondary-weight-binding')
        boxes=source.get('raw_detections',source.get('detections'));require(type(boxes)is list,'recorded-secondary-original-boxes')
        if source.get('version')=='full-secondary-RGB-inference-receipt-v1':
            require(source.get('secondary_source_identity')==self.identity,'recorded-secondary-implementation-identity')
            require(source.get('GT_inputs')is False and source.get('source_RGB_only')is True
                and (source.get('width'),source.get('height'))==(observation.width,observation.height),'recorded-secondary-RGB-only-contract')
            self.last_recorded_receipt=copy.deepcopy(source)
        else:self.last_recorded_receipt=dict(version='full-secondary-RGB-inference-receipt-v1',frame=observation.frame,timestamp=timestamp,RGB_sha256=rgbsha,
            width=observation.width,height=observation.height,secondary_source_identity=copy.deepcopy(self.identity),raw_detections=copy.deepcopy(boxes),
            synchronized_pipeline_ms=source.get('synchronized_model_pipeline_ms',source.get('inference_ms')),source_RGB_only=True,GT_inputs=False,
            fresh_NN=False,recorded_source_receipt_sha256=digest(source))
        self.calls+=1;return copy.deepcopy(boxes)


from .traffic_perception import SSDLiteBackend, COCO_TRAFFIC_LABELS

WEIGHTS_URL='https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
WEIGHTS_SHA256='dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'


class ResNet50V2Backend(SSDLiteBackend):
    """Explicitly constructed official pretrained auxiliary detector.

    Loading is opt-in: the caller supplies a local weight file, its exact hash
    and device='cpu' or device='cuda'. No download or model fallback is attempted.
    """
    def __init__(self,weights_path,*,expected_sha256,device):
        if device not in ('cpu','cuda'):raise ValueError('explicit-valid-detection-device')
        path=Path(weights_path)
        with path.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=WEIGHTS_SHA256 or expected_sha256!=WEIGHTS_SHA256:
            raise ValueError('official-ResNet50-v2-full-weight-SHA')
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
        if torch.__version__.split('+')[0]!='2.7.1'or torchvision.__version__.split('+')[0]!='0.22.1':
            raise ValueError('pinned-torch-and-vision-required')
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        if weights.url!=WEIGHTS_URL or not all(weights.meta['categories'][key]==value for key,value in COCO_TRAFFIC_LABELS.items()):
            raise ValueError('official-COCO-category-binding')
        self.torch=torch;self.device=device;self.weights_sha256=digest
        # Passing both weights arguments as None prevents implicit downloads.
        self.model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=800,max_size=1333)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.model.to(device).eval()
        self.identity=dict(architecture='fasterrcnn_resnet50_fpn_v2',weights='COCO_V1',
            weights_url=WEIGHTS_URL,weights_sha256=digest,weight_bytes=path.stat().st_size,
            torch=torch.__version__,torchvision=torchvision.__version__,device=device,trained_locally=False,
            parameters=sum(p.numel()for p in self.model.parameters()),minimum_resize=800,maximum_resize=1333,
            default_profile_selected=False,implementation_source_sha256=implementation_source_sha256())
