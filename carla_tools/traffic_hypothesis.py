"""Optional RGB hypothesis memory for fixed-camera, low-speed simulation.

Vehicle alias retirement requires repeated simultaneous RGB support, a fresh
representative beyond the stopping budget, and no old unknown or urgent threat.
Every current person remains in the original control path. A weak, never-confirmed
far-person hypothesis may expire only after four valid, stable, unoccluded current
road observations at its projected old contact patch while speed is at most
0.2 m/s. This patch is not a complete pedestrian reachable region: a real briefly
seen person can move away. No universal collision-avoidance guarantee is claimed.
"""
import colorsys
import itertools
import json
import math

from .traffic_memory import DirectedRangeBridge, _nearest
from .traffic_perception import _ground_point

def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0.0 else 0.0

def duplicate_support(a, b):
    """Strict same-frame cross-label evidence, never a person/vehicle merge."""
    if a.get('kind') != 'vehicle' or b.get('kind') != 'vehicle' or a.get('label') == b.get('label'):
        return False
    if a.get('label') not in ('car', 'truck', 'bus') or b.get('label') not in ('car', 'truck', 'bus'):
        return False
    if min(a.get('confidence', 0.0), b.get('confidence', 0.0)) < 0.5:
        return False
    if not a.get('ground_valid') or not b.get('ground_valid') or (not a.get('in_path')) or (not b.get('in_path')):
        return False
    ba, bb = (a.get('bbox_xyxy'), b.get('bbox_xyxy'))
    if not isinstance(ba, list) or not isinstance(bb, list) or len(ba) != 4 or (len(bb) != 4):
        return False
    values = ba + bb + [a.get('distance_m'), b.get('distance_m')]
    if not all((type(v) in (int, float) and math.isfinite(v) for v in values)):
        return False
    return _iou(ba, bb) >= 0.9 and abs(ba[3] - bb[3]) <= 2.0 and (abs((ba[0] + ba[2] - bb[0] - bb[2]) * 0.5) <= 3.0) and (abs(a['distance_m'] - b['distance_m']) <= 0.25)

class DuplicateEvidenceBridge(DirectedRangeBridge):
    """Retire only an obsolete vehicle identity with repeated positive alias evidence."""

    def reset(self):
        super().reset()
        self.duplicate_evidence = {}
        self.person_evidence = {}

    def _memory_update(self, raw, prediction, observation):
        now = observation.timestamp
        objects = raw['objects']
        observed = {o['track_id']: o for o in objects}
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                if not duplicate_support(a, b):
                    continue
                key = tuple(sorted((a['track_id'], b['track_id'])))
                old = self.duplicate_evidence.get(key)
                stamps = [t for t in old['timestamps'] if now - t <= 1.0 + 1e-09] if old else []
                if not stamps or abs(stamps[-1] - now) > 1e-09:
                    stamps.append(now)
                self.duplicate_evidence[key] = {'timestamps': stamps, 'last': now, 'iou': _iou(a['bbox_xyxy'], b['bbox_xyxy']), 'distance_difference_m': abs(a['distance_m'] - b['distance_m'])}
        resolved = []
        for identifier, memory in list(self.tracks.items()):
            if identifier in observed or memory['item'].get('kind') != 'vehicle' or memory['threat_relevant']:
                continue
            previous_near = _nearest(memory['item'])
            previous_closing = memory['item'].get('closing_speed_mps')
            if previous_near is None or previous_closing is None or previous_closing > max(0.5, observation.ego_speed_mps) or (previous_closing > 0.0 and previous_near / previous_closing <= 2.0):
                continue
            for key, evidence in self.duplicate_evidence.items():
                if identifier not in key or now - evidence['last'] > 0.2 + 1e-09:
                    continue
                times = evidence['timestamps']
                if len(times) < 3 or times[-1] - times[0] < 0.1 - 1e-09:
                    continue
                other = key[1] if key[0] == identifier else key[0]
                current = observed.get(other)
                if current is None or not duplicate_support(memory['item'], current):
                    continue
                current_near = _nearest(current)
                closing = current.get('closing_speed_mps')
                speed = observation.ego_speed_mps
                stopping_gap = 2.0 + speed * speed / 4.0 + 0.2 * speed
                if current_near is None or min(previous_near, current_near) <= stopping_gap + 0.25 or closing is None or (closing > max(0.5, speed)):
                    continue
                del self.tracks[identifier]
                resolved.append({'obsolete_track_id': identifier, 'observed_track_id': other, 'source': 'repeated-current-RGB-duplicate-box-and-ground-footpoint-evidence', 'support_frames': len(times), 'support_span_seconds': times[-1] - times[0], 'last_support_age_seconds': now - evidence['last'], 'last_support_iou': evidence['iou'], 'fresh_nearest_m': current_near, 'required_stopping_gap_m': stopping_gap + 0.25, 'old_close_or_unknown_hazard_released': False})
                break
        for obj in objects:
            if obj.get('kind') != 'pedestrian' or obj.get('confidence', 0.0) < 0.5:
                continue
            identifier = obj['track_id']
            ledger = self.person_evidence.setdefault(identifier, {'seen': 0, 'first': now, 'last': now})
            ledger['seen'] += 1
            ledger['last'] = now
        result, notes = super()._memory_update(raw, prediction, observation)
        notes['identity_memory'] = {'version': 'positive-RGB-duplicate-vehicle-evidence-v2', 'GT_inputs': False, 'raw_boxes_and_track_ids_changed': False, 'resolved_duplicate_identities': resolved, 'pedestrian_evidence': {identifier: {'status': 'confirmed-multiple-frames' if ledger['seen'] >= 3 else 'tentative', 'RGB_observation_count': ledger['seen'], 'age_seconds': now - ledger['last'], 'safety_memory_unchanged': True} for identifier, ledger in self.person_evidence.items()}}
        return (result, notes)

def project(point, calibration):
    m = calibration['camera_to_actor_matrix']
    k = calibration['intrinsics']
    delta = [point[i] - m[i][3] for i in range(3)]
    local = [sum((m[r][c] * delta[r] for r in range(3))) for c in range(3)]
    if local[0] <= 0.2:
        return None
    return [k['cx'] + k['fx'] * local[1] / local[0], k['cy'] - k['fy'] * local[2] / local[0]]

def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

class HypothesisMemoryBridge(DuplicateEvidenceBridge):
    """Experimental weak-far-person existence update; not a reachable-region safety proof."""

    def reset(self):
        super().reset()
        self.tentative_people = {}
        self.previous_anchor = None
        self.low_speed_reference_anchor = None
        self.low_speed_camera_jump = False
        self.fixed_calibration_serialized = json.dumps(self.calibration, sort_keys=True)

    def acknowledge(self, command):
        super().acknowledge(command)
        diag = command.get('diagnostics', {}).get('traffic_supervisor', {})
        if diag.get('emergency_brake'):
            for reason in diag.get('reasons', []):
                if reason.startswith('emergency_object:'):
                    key = reason.split(':', 1)[1]
                    if key in self.tentative_people:
                        self.tentative_people[key]['strong'] = True
                        if key in self.tracks:
                            self.tracks[key]['threat_relevant'] = True

    def _footprint(self, item):
        b = item.get('bbox_xyxy')
        if not item.get('ground_valid') or not isinstance(b, list) or len(b) != 4:
            return None
        if b[0] <= 1 or b[2] >= self.calibration['width'] - 1 or b[3] >= self.calibration['height'] - 1:
            return None
        points = [_ground_point((b[0] + b[2]) * 0.5 + du, b[3] + dv, self.calibration, height_shift=dh, pitch_shift=dp) for du, dv, dh, dp in itertools.product((-3.0, 3.0), (-3.0, 3.0), (-0.15, 0.15), (-2.0, 2.0))]
        if any((p is None for p in points)):
            return None
        return [min((p[0] for p in points)), min((p[1] for p in points)), max((p[0] for p in points)), max((p[1] for p in points))]

    def _road_certificate(self, ledger, raw, observation, camera_stable):
        if not raw.get('perception_valid') or not camera_stable:
            return (False, 'invalid-or-unstable-RGB', None)
        if observation.ego_speed_mps > 0.2:
            return (False, 'ego-not-stationary-or-low-speed', None)
        if json.dumps(self.calibration, sort_keys=True) != self.fixed_calibration_serialized:
            return (False, 'nominal-camera-contract-changed', None)
        x, y = ledger['nominal_ground']
        x -= self.travel - ledger['travel']
        uv = project((x, y, 0.0), self.calibration)
        if uv is None:
            return (False, 'contact-patch-not-projectable', None)
        half = max(6.0, min(24.0, ledger['bbox_width_px'] * 0.5 + 4.0))
        roi = [uv[0] - half, uv[1] - 5.0, uv[0] + half, uv[1] + 5.0]
        details = {'current_contact_roi': roi, 'nominal_ground_after_ego_travel_m': [x, y], 'scope': 'tentative-existence-belief;not-full-reachable-person-area', 'universal_safety_guarantee_claimed': False}
        if roi[0] <= 1 or roi[1] <= 1 or roi[2] >= observation.width - 1 or (roi[3] >= observation.height - 1):
            return (False, 'contact-patch-outside-camera', details)
        if any((intersects(roi, d['bbox_xyxy']) for d in raw['detections'])):
            return (False, 'current-contact-patch-occluded-or-object-detected', details)
        road = 0
        total = 0
        x0, y0, x1, y1 = [int(v) for v in roi]
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                at = (py * observation.width + px) * 3
                _, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in observation.rgb[at:at + 3]))
                road += s <= 0.2 and 0.15 <= v <= 0.65
                total += 1
        details.update(road_HSV_fraction=road / max(1, total), road_sample_count=total)
        if total < 30 or road / max(1, total) < 0.95:
            return (False, 'contact-road-surface-not-unambiguous', details)
        return (True, 'repeated-valid-current-RGB-negative-detection-and-visible-road-contact-patch', details)

    def _memory_update(self, raw, prediction, observation):
        rgb = getattr(observation, 'rgb', None)
        valid_rgb = type(rgb) is bytes and len(rgb) == observation.width * observation.height * 3
        anchor = None
        stable = False
        if valid_rgb:
            anchor = bytes((rgb[(y * observation.width + x) * 3 + c] for y in range(observation.height - 45, observation.height - 10, 5) for x in range(observation.width // 2 - 40, observation.width // 2 + 40, 5) for c in range(3)))
            stable = self.previous_anchor is not None and sum((abs(a - b) for a, b in zip(anchor, self.previous_anchor))) / len(anchor) <= 3.0
            if observation.ego_speed_mps > 0.2:
                self.low_speed_reference_anchor = None
            elif self.low_speed_reference_anchor is None:
                self.low_speed_reference_anchor = anchor
            elif self.tentative_people:
                difference = sum((abs(a - b) for a, b in zip(anchor, self.low_speed_reference_anchor))) / len(anchor)
                if difference > 8.0:
                    self.low_speed_camera_jump = True
            stable = stable and (not self.low_speed_camera_jump)
        self.previous_anchor = anchor
        current = {o['track_id']: o for o in raw['objects']}
        for item in current.values():
            if item.get('kind') != 'pedestrian' or item.get('confidence', 0.0) < 0.5:
                continue
            key = item['track_id']
            ledger = self.tentative_people.get(key)
            footprint = self._footprint(item)
            if ledger is None:
                ledger = {'seen': 0, 'strong': False, 'footprint': footprint, 'last': observation.timestamp, 'travel': self.travel, 'support': 0}
                self.tentative_people[key] = ledger
            ledger['seen'] += 1
            ledger['strong'] |= ledger['seen'] >= 3 or item['confidence'] >= 0.85 or footprint is None or (footprint[0] - self.front <= 8.0)
            ledger.update(footprint=footprint, last=observation.timestamp, travel=self.travel, support=0, nominal_ground=(item['distance_m'] + self.front, item['lateral_y_m']) if footprint else None, bbox_width_px=item['bbox_xyxy'][2] - item['bbox_xyxy'][0])
        events = []
        diagnostics = []
        for key, ledger in self.tentative_people.items():
            if key in current or ledger['strong'] or key not in self.tracks:
                continue
            if not valid_rgb:
                supported, reason, details = (False, 'no-current-RGB-bytes', None)
            else:
                supported, reason, details = self._road_certificate(ledger, raw, observation, stable)
            ledger['support'] = ledger['support'] + 1 if supported else 0
            diagnostics.append({'track_id': key, 'reason': reason, 'positive_support_frames': ledger['support'], 'details': details})
            if ledger['support'] >= 4:
                del self.tracks[key]
                events.append({'track_id': key, 'source': reason, 'positive_support_frames': ledger['support'], 'details': details})
        result, notes = super()._memory_update(raw, prediction, observation)
        notes['person_clearance'] = {'version': 'complete-reachable-ground-RGB-clearance-prototype-v1', 'GT_inputs': False, 'current_persons_filtered': False, 'retired_tentative_hypotheses': events, 'certificate_checks': diagnostics, 'low_speed_camera_jump_latched': self.low_speed_camera_jump, 'assumptions': 'fixed nominal camera;level ground;person speed<=2mps;grey HSV road witness;valid current detector', 'safety_guarantee_claimed': False}
        notes['person_clearance'].update(version='tentative-far-person-contact-road-lifecycle-v1', inference_kind='tentative-existence-belief-update;not-proven-physical-clearance', assumptions='fixed nominal camera;level road;weak far person never confirmed;low-speed current road visibility;valid current detector;stationary-image anchor', residual_failure='A real far person seen only briefly can move away from its old contact patch; no universal occlusion or detection guarantee.')
        return (result, notes)
