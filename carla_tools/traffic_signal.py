"""Causal RGB signal evidence, separating a suspect box from a red latch.

This does not infer which lane a signal legally controls. Confirmed red/yellow
retains the original supervisor's conservative latch until associated green.
An unreadable box requests a stop while visible and for a short absence hold;
it is never promoted into a permanently confirmed red signal by a timeout.
"""
from dataclasses import dataclass, asdict
import copy
import math


@dataclass(frozen=True)
class SignalEvidenceConfig:
    unknown_absence_hold_seconds: float = .5
    minimum_color_confidence: float = .55
    minimum_color_pixels: int = 3
    minimum_head_iou: float = .1
    associated_green_hold_seconds: float = .3

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid-signal-evidence-config')
        if self.minimum_color_pixels < 3 or not 0 < self.minimum_color_confidence <= 1 or not 0 < self.minimum_head_iou <= 1:
            raise ValueError('invalid-signal-evidence-threshold')


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


class SignalEvidenceAdapter:
    def __init__(self, config=None):
        self.config = config or SignalEvidenceConfig()
        self.reset()

    def reset(self):
        self.time = None
        self.frame = None
        self.last_unknown = None
        self.confirmed_heads = []
        self.green_since = None

    def _colored(self, signal, colors):
        return (signal.get('color') in colors
                and signal.get('confidence', 0.) >= self.config.minimum_color_confidence
                and signal.get('counts', {}).get(signal.get('color'), 0) >= self.config.minimum_color_pixels)

    def update(self, raw, observation):
        if (raw.get('GT_inputs') is not False or raw['frame'] != observation.frame
                or abs(raw['timestamp']-observation.timestamp) > 1e-7):
            raise ValueError('signal-evidence-RGB-provenance-required')
        now = observation.timestamp
        if self.time is not None and (observation.frame <= self.frame or abs(now-self.time-.05) > 1e-6):
            raise ValueError('signal-evidence-consecutive-RGB-required')
        self.time = now
        self.frame = observation.frame
        result = copy.deepcopy(raw)
        signals = raw.get('signals', [])
        for signal in signals:
            box = signal.get('bbox_xyxy')
            if (not isinstance(box, list) or len(box) != 4
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= observation.width
                    or not 0 <= box[1] < box[3] <= observation.height):
                raise ValueError('invalid-RGB-signal-bbox')
        stops = [s for s in signals if self._colored(s, ('red', 'yellow'))]
        greens = [s for s in signals if self._colored(s, ('green',))]
        unreadable = [s for s in signals if not self._colored(s, ('red', 'yellow', 'green'))]
        # A selected unknown without individual box evidence remains uncertain.
        suspect = bool(unreadable or raw.get('signal') == 'unknown'
                       or raw.get('signal') == 'green' and raw.get('signal_confidence', 0.) < self.config.minimum_color_confidence)
        if suspect:
            self.last_unknown = now
        hold = (self.last_unknown is not None
                and now-self.last_unknown <= self.config.unknown_absence_hold_seconds+1e-9)
        associated_green = False
        if stops:
            self.green_since = None
            for signal in stops:
                box = signal['bbox_xyxy']
                matches = [(i, overlap(box, old)) for i, old in enumerate(self.confirmed_heads)]
                index, score = max(matches, key=lambda x: x[1], default=(None, 0.))
                if score >= self.config.minimum_head_iou:
                    self.confirmed_heads[index] = list(box)
                else:
                    self.confirmed_heads.append(list(box))
            selected = next((s for color in ('red', 'yellow') for s in stops if s['color'] == color), stops[0])
            result['signal'] = selected['color']
            result['signal_confidence'] = selected['confidence']
        elif self.confirmed_heads:
            # Every previously confirmed head must be positively matched. A
            # green elsewhere in the image cannot release a confirmed red.
            associated_green = bool(greens) and all(any(
                overlap(box, green['bbox_xyxy']) >= self.config.minimum_head_iou for green in greens)
                for box in self.confirmed_heads)
            if associated_green and not hold:
                result['signal'] = 'green'
                result['signal_confidence'] = min(g['confidence'] for g in greens)
                if self.green_since is None:
                    self.green_since = now
                if now-self.green_since >= self.config.associated_green_hold_seconds-1e-9:
                    self.confirmed_heads = []
                    self.green_since = None
            else:
                # Preserve confirmed stop evidence even if an upstream
                # invalid frame prevented the original supervisor seeing it.
                result['signal'] = 'red'
                result['signal_confidence'] = 1.
                self.green_since = None
        elif greens and not hold:
            result['signal'] = 'green'
            result['signal_confidence'] = min(g['confidence'] for g in greens)
        else:
            result['signal'] = 'none'
            result['signal_confidence'] = 0.
            self.green_since = None
        if hold:
            # Preserve any upstream invalidity. Unknown never becomes a
            # permanent confirmed-red event, but visible uncertainty never
            # expires into permission to drive.
            result['perception_valid'] = False
        notes = {
            'version': 'causal-RGB-signal-evidence-v1',
            'GT_inputs': False,
            'raw_modified': False,
            'raw_signal': raw.get('signal'),
            'effective_signal': result['signal'],
            'unknown_currently_visible': suspect,
            'unknown_absence_hold_active': hold,
            'unknown_absence_hold_seconds': self.config.unknown_absence_hold_seconds,
            'last_unknown_timestamp': self.last_unknown,
            'confirmed_RGB_head_count': len(self.confirmed_heads),
            'associated_green_observed': associated_green,
            'confirmed_red_timeout_release': False,
            'own_approach_association_implemented': False,
            'suppressing_signal_handling_globally': False,
        }
        return result, notes
