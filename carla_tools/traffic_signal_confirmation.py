"""Optional RGB signal confirmation that keeps detector and colour evidence separate.

Every visible uncertain stop colour requests an immediate conservative stop.
Permanent stop memory requires repeated evidence, a previously persistent head,
or strong independent RGB housing/lamp support. Confirmed stop memory is never
released by time alone; the base adapter requires associated green. Projected
lane direction is diagnostic and cannot establish legal signal applicability.
Vertical signal housings and the fixed-camera, 20 Hz low-speed contract are
assumptions. Weak briefly visible real heads remain an ambiguity.
"""
import copy
import colorsys
import math

from .traffic_perception import validate_nominal_calibration
from .traffic_signal import SignalEvidenceAdapter as BaseSignalEvidenceAdapter, overlap as head_overlap

CONFIG = {'version': 'RGB-detector-colour-and-head-confirmation-v2', 'minimum_detector_score': 0.5, 'minimum_colour_confidence': 0.55, 'minimum_coloured_pixels': 3, 'minimum_head_width_px': 3.0, 'minimum_head_height_px': 8.0, 'minimum_vertical_head_aspect': 1.5, 'consecutive_observations_required': 3, 'minimum_observation_span_seconds': 0.1, 'minimum_joint_evidence_seconds': 0.075, 'frame_interval_seconds': 0.05, 'strong_single_observation_detector_score': 0.9, 'strong_RGB_head_minimum_width_px': 4.0, 'strong_RGB_head_minimum_height_px': 12.0, 'strong_RGB_head_minimum_aspect': 1.8, 'strong_lamp_expected_position_fraction': 0.7, 'persistent_head_observations_required': 3, 'head_association_iou': 0.1, 'unconfirmed_head_expiry_seconds': 0.5, 'unconfirmed_red_or_yellow_immediately_requests_stop': True, 'confirmed_red_timeout_release': False, 'own_approach_projection_is_diagnostic_only': True, 'vertical_head_design_assumption': 'current CARLA DEV heads; other forms remain uncertain and request stop'}

class SignalConfirmationAdapter(BaseSignalEvidenceAdapter):

    def __init__(self, nominal_calibration=None):
        self.calibration = validate_nominal_calibration(nominal_calibration) if nominal_calibration is not None else None
        super().__init__()

    def reset(self):
        super().reset()
        self.tracks = []
        self.confirmation_time = None

    @staticmethod
    def detector_score(signal, raw):
        bbox = signal['bbox_xyxy']
        matches = [d for d in raw.get('detections', []) if d.get('label_id') == 10 and len(d.get('bbox_xyxy', [])) == 4 and (max((abs(a - b) for a, b in zip(d['bbox_xyxy'], bbox))) <= 0.0001)]
        if len(matches) != 1:
            return None
        score = matches[0].get('confidence')
        return score if type(score) in (int, float) and math.isfinite(score) and (0 <= score <= 1) else None

    @staticmethod
    def RGB_head_shape(signal, observation):
        """Independent pixels: compact lit lamp in a dark vertical housing.

        The current CARLA DEV heads have red/top, yellow/middle, green/bottom.
        No detector output or colour count can manufacture this pixel support.
        """
        rgb = getattr(observation, 'rgb', None)
        if type(rgb) is not bytes or len(rgb) != observation.width * observation.height * 3:
            return {'supported': False, 'reason': 'RGB-bytes-unavailable'}
        box = signal['bbox_xyxy']
        width, height = (box[2] - box[0], box[3] - box[1])
        if width < 4.0 or height < 12.0 or height / max(width, 1e-09) < 1.8:
            return {'supported': False, 'reason': 'insufficient-vertical-housing-pixel-support'}
        x0, y0, x1, y1 = [int(v) for v in box]
        coloured = []
        dark = 0
        total = 0
        color = signal.get('color')
        h = max(1, y1 - y0)
        for y in range(max(0, y0), min(observation.height, y1)):
            for x in range(max(0, x0), min(observation.width, x1)):
                at = (y * observation.width + x) * 3
                r, g, b = [v / 255.0 for v in rgb[at:at + 3]]
                hue, sat, value = colorsys.rgb_to_hsv(r, g, b)
                total += 1
                dark += value <= 0.4
                degree = hue * 360.0
                same = color == 'red' and (degree <= 20.0 or degree >= 340.0) or (color == 'yellow' and 35.0 <= degree <= 75.0) or (color == 'green' and 80.0 <= degree <= 165.0)
                if same and sat >= 0.5 and (value >= 0.45):
                    coloured.append((x, y))
        if len(coloured) < 3:
            return {'supported': False, 'reason': 'insufficient-independent-lamp-pixels'}
        xs, ys = zip(*coloured)
        aspect = (max(xs) - min(xs) + 1) / max(1, max(ys) - min(ys) + 1)
        positions = [(y - y0) / h for y in ys]
        expected = sum((p <= 0.45 if color == 'red' else 0.2 <= p <= 0.8 if color == 'yellow' else p >= 0.55 for p in positions)) / len(positions)
        dark_fraction = dark / max(1, total)
        supported = expected >= 0.7 and dark_fraction >= 0.5 and (0.5 <= aspect <= 2.0)
        return {'supported': supported, 'reason': None if supported else 'lamp-position-or-dark-housing-not-supported', 'independent_lamp_pixels': len(coloured), 'lamp_blob_aspect': aspect, 'lamp_expected_vertical_position_fraction': expected, 'dark_housing_fraction': dark_fraction}

    def projection_note(self, box, prediction):
        if not self.calibration or not prediction or prediction.get('status') != 'PASS':
            return {'status': 'UNAVAILABLE', 'used_to_clear_a_signal': False, 'legal_applicability_inferred': False}
        points = prediction.get('waypoints_ego_m', [])
        if len(points) < 2:
            return {'status': 'UNAVAILABLE', 'used_to_clear_a_signal': False, 'legal_applicability_inferred': False}
        dx, dy = (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
        matrix = self.calibration['camera_to_actor_matrix']
        direction = [dx, dy, 0.0]
        ray = [sum((matrix[j][i] * direction[j] for j in range(3))) for i in range(3)]
        if ray[0] <= 0:
            return {'status': 'UNAVAILABLE', 'used_to_clear_a_signal': False, 'legal_applicability_inferred': False}
        k = self.calibration['intrinsics']
        vp = [k['cx'] + k['fx'] * ray[1] / ray[0], k['cy'] - k['fy'] * ray[2] / ray[0]]
        return {'status': 'RGB_PATH_TANGENT_PROJECTION_ONLY', 'vanishing_direction_uv': vp, 'head_horizontal_bearing_from_predicted_path_deg': math.degrees(math.atan(((box[0] + box[2]) / 2 - vp[0]) / k['fx'])), 'used_to_clear_a_signal': False, 'legal_applicability_inferred': False, 'reason': 'One camera bearing cannot establish which approach a visible head controls'}

    def update(self, raw, observation, prediction=None):
        if raw.get('GT_inputs') is not False or raw['frame'] != observation.frame or abs(raw['timestamp'] - observation.timestamp) > 1e-07:
            raise ValueError('signal-confirmation-current-RGB-evidence-required')
        now = observation.timestamp
        if self.confirmation_time is not None and abs(now - self.confirmation_time - 0.05) > 1e-06:
            raise ValueError('signal-confirmation-consecutive-time-required')
        self.confirmation_time = now
        self.tracks = [t for t in self.tracks if t['confirmed'] or now - t['last'] <= 0.5 + 1e-09]
        filtered = copy.deepcopy(raw)
        evidence = []
        used = set()
        for index, signal in enumerate(filtered.get('signals', [])):
            box = signal['bbox_xyxy']
            score = self.detector_score(signal, raw)
            color = signal.get('color')
            colour = signal.get('confidence', 0.0)
            count = signal.get('counts', {}).get(color, 0)
            width, height = (box[2] - box[0], box[3] - box[1])
            head = width >= 3.0 and height >= 8.0 and (height / max(width, 1e-09) >= 1.5)
            readable = color in ('red', 'yellow', 'green') and colour >= 0.55 and (count >= 3)
            reliable = readable and score is not None and (score >= 0.5) and head
            joint = 0.0 if score is None else score * colour
            appearance = self.RGB_head_shape(signal, observation) if readable else {'supported': False, 'reason': 'colour-not-readable'}
            track = None
            prior_head = False
            if score is not None and score >= 0.5 and head:
                matches = [(i, head_overlap(box, t['box'])) for i, t in enumerate(self.tracks) if i not in used]
                identifier, overlap = max(matches, key=lambda x: x[1], default=(None, 0.0))
                if overlap < 0.1:
                    track = {'box': list(box), 'last': now, 'first': now, 'head_first': now, 'head_count': 1, 'count': 0, 'joint_seconds': 0.0, 'confirmed': False, 'basis': None}
                    self.tracks.append(track)
                    identifier = len(self.tracks) - 1
                else:
                    track = self.tracks[identifier]
                    consecutive = abs(now - track['last'] - 0.05) <= 1e-06
                    prior_head = consecutive and track['head_count'] >= 3 and (now - track['head_first'] >= 0.1 - 1e-08)
                    if not consecutive and (not track['confirmed']):
                        track.update(first=now, count=0, joint_seconds=0.0, head_first=now, head_count=0)
                    track.update(box=list(box), last=now, head_count=track['head_count'] + 1)
                used.add(identifier)
                if color in ('red', 'yellow') and reliable:
                    if track['count'] == 0:
                        track['first'] = now
                    track['count'] += 1
                    track['joint_seconds'] += joint * 0.05
                    temporal = track['count'] >= 3 and now - track['first'] >= 0.1 - 1e-08 and (track['joint_seconds'] >= 0.075 - 1e-09)
                    strong = score >= 0.9 and appearance['supported']
                    if temporal or prior_head or strong:
                        track['confirmed'] = True
                        track['basis'] = 'strong-NN-and-independent-RGB-head' if strong else 'persistent-head-before-stop-colour' if prior_head else 'consecutive-stop-colour'
                elif not track['confirmed']:
                    track.update(count=0, joint_seconds=0.0, first=now)
            confirmed = bool(track and track['confirmed'])
            if color in ('red', 'yellow') and (not confirmed) or (color == 'green' and (not reliable)):
                signal['color'] = 'unknown'
                signal['confidence'] = 0.0
            evidence.append({'bbox_xyxy': list(box), 'original_colour': color, 'original_colour_confidence': colour, 'detector_score_recovered_from_original_detection': score, 'joint_detector_colour_score': joint, 'vertical_head_geometry_supported': head, 'reliable_coloured_head': reliable, 'independent_RGB_head_shape': appearance, 'previous_persistent_RGB_head_supported': prior_head, 'confirmed_stop_basis': track.get('basis') if track else None, 'temporal_confirmed_stop': confirmed, 'consecutive_stop_observations': track['count'] if track else 0, 'joint_evidence_seconds': track['joint_seconds'] if track else 0.0, 'projected_RGB_path_evidence': self.projection_note(box, prediction)})
        selected = next((s for color in ('red', 'yellow', 'unknown', 'green') for s in filtered.get('signals', []) if s['color'] == color), None)
        if selected:
            filtered['signal'] = selected['color']
            filtered['signal_confidence'] = selected['confidence']
        elif raw.get('signal') in ('red', 'yellow', 'green'):
            filtered.update(signal='unknown', signal_confidence=0.0)
        elif raw.get('signal') != 'unknown':
            filtered.update(signal='none', signal_confidence=0.0)
        result, notes = super().update(filtered, observation)
        notes.update(version=CONFIG['version'], unmodified_raw_signal=raw.get('signal'), confirmation_evidence=evidence, detector_and_colour_confidence_kept_separate=True, own_approach_association_implemented=False)
        return (result, notes)
