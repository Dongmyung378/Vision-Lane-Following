"""Optional exclusive ownership of positive secondary person evidence.

Each auxiliary person box selects its unique best current primary person.
Strong and otherwise ineligible primaries still compete. Equal overlaps remain
ambiguous, and no leftover box is reassigned to a weaker primary. This does not
remove primary observations or authorize memory clearance. Two overlapping real
people can share one box and lose additional confirmation; no identity or safety
guarantee follows. Importing this module starts no model or device runtime.
"""
from pathlib import Path
import copy
import hashlib

from . import traffic_secondary_confirmation as _original_module
from .traffic_secondary_confirmation import (
    SecondaryPersonConfirmationScheduler as _OriginalScheduler,
    overlap, valid_box, require,
)

# Bind receipts to the installed base rather than an external producer path.
ORIGINAL_SCHEDULER_SHA256 = _original_module.implementation_source_sha256()

ASSIGNMENT = dict(
    version='secondary-person-exclusive-current-primary-owner-v1',
    minimum_primary_score=.5,minimum_secondary_score=.5,minimum_IoU=.3,
    IoU_tie_tolerance=1e-6,
    competitors='all-current-primary-persons-including-strong-near-outside-and-ineligible',
    assignment='each-secondary-first-selects-unique-best-primary-IoU',
    fallback_reassignment=False,ambiguous_confirmation=None,
    multiple_secondary_boxes_may_support_same_primary=True,
    one_secondary_box_may_support_multiple_primary=False,
    primary_objects_or_original_memory_clearance_modified=False,
    GT_inputs=False)


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def assign_secondary_people(raw,detections,width,height):
    """Freeze ownership before considering which primary may receive a veto.

    All secondary boxes independently compete against the complete primary set.
    There is no leftover-box assignment: duplicate secondary boxes can all stay
    owned by one better primary and never fall through to its weaker sub-box.
    Equal best overlap is unresolved, including two genuinely distinct people.
    """
    require(raw.get('GT_inputs') is False,'exclusive-assignment-no-GT-inputs')
    primary=[];keys=set()
    for p in raw['objects']:
        if p.get('kind')!='pedestrian' or p.get('confidence',0)<.5:
            continue
        require(p['track_id'] not in keys,'exclusive-primary-unique-ID')
        require(valid_box(p.get('bbox_xyxy'),width,height),'exclusive-current-primary-original-box')
        keys.add(p['track_id']);primary.append(p)
    primary.sort(key=lambda p:p['track_id'])
    owned={p['track_id']:[] for p in primary};ambiguous=set();decisions=[]
    for index,d in enumerate(detections):
        if d.get('label_id')!=1 or d.get('score',0)<.5:
            continue
        require(valid_box(d.get('bbox_xyxy'),width,height),'exclusive-secondary-original-box')
        edges=[dict(primary_track_id=p['track_id'],IoU=overlap(p['bbox_xyxy'],d['bbox_xyxy']),
            primary_score=p['confidence']) for p in primary]
        edges.sort(key=lambda edge:(-edge['IoU'],edge['primary_track_id']))
        eligible=[edge for edge in edges if edge['IoU']>=.3]
        best=[] if not eligible else [edge for edge in eligible if abs(edge['IoU']-eligible[0]['IoU'])<=1e-6]
        owner=best[0]['primary_track_id'] if len(best)==1 else None
        if owner is not None:
            owned[owner].append(dict(bbox_xyxy=copy.deepcopy(d['bbox_xyxy']),score=d['score'],label_id=1,
                IoU=best[0]['IoU']))
        if len(best)>1:
            ambiguous.update(edge['primary_track_id'] for edge in best)
        decisions.append(dict(secondary_detection_index=index,secondary_bbox_xyxy=copy.deepcopy(d['bbox_xyxy']),
            secondary_score=d['score'],all_primary_competition=edges,owner_primary_track_id=owner,
            ambiguous_primary_track_ids=[edge['primary_track_id'] for edge in best] if len(best)>1 else [],
            reason='unique-best-current-primary' if owner is not None else 'ambiguous-best-IoU' if best else 'no-primary-IoU-support',
            fallback_reassignment_performed=False))
    return owned,ambiguous,decisions


class ExclusiveSecondaryPersonConfirmationScheduler(_OriginalScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.configuration['positive_assignment']=copy.deepcopy(ASSIGNMENT)
        self.configuration['exclusive_assignment_source_sha256']=implementation_sha256()

    def update(self,raw,observation):
        prior={key:value['positive_confirmed'] for key,value in self.tracks.items()}
        result=super().update(raw,observation)
        audit=dict(configuration=copy.deepcopy(ASSIGNMENT),implementation_source_sha256=implementation_sha256(),
            original_scheduler_source_sha256=ORIGINAL_SCHEDULER_SHA256,secondary_ownership=[],
            confirmation_changes=[],source_RGB_only=True,GT_inputs=False)
        result['exclusive_person_assignment']=audit
        if result['inference_receipt'] is None:
            return result
        owned,ambiguous,decisions=assign_secondary_people(raw,result['inference_receipt']['raw_detections'],observation.width,observation.height)
        audit['secondary_ownership']=decisions
        for confirmation in result['confirmations']:
            key=confirmation['primary_track_id'];original=copy.deepcopy(confirmation)
            matches=owned.get(key,[])
            value=True if matches else None if key in ambiguous else False
            confirmation['matched_secondary_persons']=matches
            confirmation['secondary_person_confirmation']=value
            self.tracks[key]['positive_confirmed']=prior.get(key,False) or value is True
            if original['secondary_person_confirmation']!=value or original['matched_secondary_persons']!=matches:
                audit['confirmation_changes'].append(dict(primary_track_id=key,
                    original_confirmation=original['secondary_person_confirmation'],exclusive_confirmation=value,
                    reason='unique-positive-owner' if matches else 'ambiguous-IoU-no-additional-veto' if value is None else 'secondary-evidence-belongs-to-another-current-primary-or-no-support',
                    original_matches=original['matched_secondary_persons'],exclusive_matches=copy.deepcopy(matches),
                    current_primary_and_parent_memory_not_modified=True))
        return result
