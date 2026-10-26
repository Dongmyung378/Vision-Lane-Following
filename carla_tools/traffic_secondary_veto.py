"""Optional positive-person memory veto for bounded simulation experiments.

Use SecondaryHypothesisTrafficBridge explicitly through TrafficPolicy(bridge=).
Existing factories and default policies are unaffected. Without a scheduler,
the bridge equals the existing experimental hypothesis strategy, which is
distinct from the default traffic evidence strategy.

A source-bound positive agreement can only retain an existing weak hypothesis.
Both absence deletion paths are guarded; the parent's current same-ID positive
lateral-clearance exit stays active. False/None never clear an earlier veto.
This does not establish collision avoidance or public-road readiness.
"""
import copy
import hashlib
import math
import re

from .traffic_hypothesis import HypothesisMemoryBridge
from .traffic_experimental import HypothesisTrafficBridge
from .traffic_signal_confirmation import SignalConfirmationAdapter
from .traffic_secondary_confirmation import implementation_source_sha256

PRIMARY_SHA256 = "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533"
SECONDARY_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
RECEIPT_VERSION = "primary-ROI-positive-secondary-person-confirmation-v1"
CONFIG = dict(version="same-frame-secondary-person-receipt-veto-v1",
    primary_score_minimum=.5, primary_score_maximum_exclusive=.85,
    secondary_person_score_minimum=.5, minimum_person_box_IoU=.3,
    maximum_initial_observations=2, minimum_heuristic_one_sided_gap_m=8.,
    True_veto_sticky_until="original-current-same-ID-positive-clearance-or-reset",
    False_None_policy="parent-policy;does-not-undo-earlier-True",
    original_current_lateral_clearance_seconds=.2,
    new_retirement_permission=False, default_profile_promoted=False)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def validate_box(box, width, height):
    require(type(box) is list and len(box) == 4 and all(finite(value) for value in box), "secondary-veto-invalid-box")
    require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
        "secondary-veto-box-outside-original-RGB")


class SecondaryPersonVetoBridge(HypothesisMemoryBridge):
    """Guard existing absence retirement with source-bound positive receipts.

    Inject a trusted scheduler. The optional implementation hash override is an
    explicit replay/source pin, not permission to accept untrusted producers.
    This adapter does not execute a vehicle fail-safe when receipt validation
    raises: the surrounding application must stop on policy exceptions.
    """
    def __init__(self, *, secondary_scheduler=None, secondary_backend_source_sha256=None, **geometry):
        if secondary_backend_source_sha256 is not None:
            require(type(secondary_backend_source_sha256) is str
                and re.fullmatch("[a-f0-9]{64}", secondary_backend_source_sha256) is not None,
                "secondary-veto-explicit-implementation-SHA")
        self.secondary_scheduler = secondary_scheduler
        self.secondary_backend_source_sha256 = secondary_backend_source_sha256
        super().__init__(**geometry)

    def reset(self):
        super().reset()
        self.secondary_veto_ids = {}
        if self.secondary_scheduler is not None:
            require(callable(getattr(self.secondary_scheduler, "reset", None)), "secondary-veto-scheduler-reset-required")
            self.secondary_scheduler.reset()

    def _backend_implementation_sha256(self):
        if self.secondary_backend_source_sha256 is not None:
            return self.secondary_backend_source_sha256
        return implementation_source_sha256()

    def _verified_true_confirmations(self, packet, raw, observation):
        """Read only explicit matched-person confirmations as decision evidence."""
        if packet is None:
            return {}, []
        require(type(packet) is dict, "secondary-veto-receipt-dictionary-required")
        confirmations = packet.get("confirmations", [])
        require(type(confirmations) is list and len(confirmations) <= len(raw["objects"]),
            "secondary-veto-bounded-confirmation-list")
        true_items, decisions, seen = {}, [], set()
        current = {item["track_id"]: item for item in raw["objects"]}
        require(len(current) == len(raw["objects"]), "secondary-veto-unique-primary-track-IDs")
        for confirmation in confirmations:
            require(type(confirmation) is dict, "secondary-veto-confirmation-dictionary-required")
            value = confirmation.get("secondary_person_confirmation")
            require(value is None or type(value) is bool, "secondary-veto-explicit-boolean-or-None")
            key = confirmation.get("primary_track_id")
            require(type(key) is str and key not in seen, "secondary-veto-unique-primary-confirmation")
            seen.add(key)
            decisions.append(dict(primary_track_id=key, secondary_person_confirmation=value))
            if value is not True:
                continue
            require(packet.get("version") == RECEIPT_VERSION and packet.get("GT_inputs") is False,
                "secondary-veto-receipt-version-or-GT-contract")
            require(type(packet.get("frame")) is int and packet["frame"] == observation.frame
                and finite(packet.get("timestamp")) and abs(packet["timestamp"] - observation.timestamp) <= 1e-7,
                "secondary-veto-receipt-frame-or-age-mismatch")
            require(type(observation.rgb) is bytes and len(observation.rgb) == observation.width * observation.height * 3
                and packet.get("RGB_sha256") == hashlib.sha256(observation.rgb).hexdigest(),
                "secondary-veto-receipt-original-RGB-SHA-mismatch")
            primary_identity, secondary_identity = packet.get("primary_source_identity"), packet.get("secondary_source_identity")
            require(type(primary_identity) is dict and primary_identity.get("architecture") == "fasterrcnn_mobilenet"
                and primary_identity.get("weights_sha256") == PRIMARY_SHA256, "secondary-veto-primary-source-binding")
            require(type(secondary_identity) is dict
                and secondary_identity.get("architecture") == "fasterrcnn_resnet50_fpn_v2"
                and secondary_identity.get("weights_sha256") == SECONDARY_SHA256
                and secondary_identity.get("implementation_source_sha256") == self._backend_implementation_sha256(),
                "secondary-veto-secondary-source-binding")
            item = current.get(key)
            require(item is not None and item.get("kind") == "pedestrian"
                and item.get("label") == "person" and finite(item.get("confidence")) and item["confidence"] >= .5,
                "secondary-veto-only-current-primary-person")
            primary_box = confirmation.get("primary_bbox_xyxy")
            validate_box(primary_box, observation.width, observation.height)
            require(all(abs(a - b) <= 1e-6 for a, b in zip(primary_box, item["bbox_xyxy"]))
                and finite(confirmation.get("primary_score")) and abs(confirmation["primary_score"] - item["confidence"]) <= 1e-7,
                "secondary-veto-primary-box-or-score-mismatch")
            ledger = self.tentative_people.get(key)
            expected_count = 1 if ledger is None else ledger["seen"] + 1
            require(type(confirmation.get("observation_count")) is int and confirmation["observation_count"] == expected_count,
                "secondary-veto-primary-observation-count-mismatch")
            matches = confirmation.get("matched_secondary_persons")
            require(type(matches) is list and 1 <= len(matches) <= 1000, "secondary-veto-positive-person-match-required")
            recomputed = []
            for match in matches:
                require(type(match) is dict and match.get("label_id") == 1 and type(match.get("label_id")) is int
                    and finite(match.get("score")) and .5 <= match["score"] <= 1., "secondary-veto-matched-person-score-or-class")
                validate_box(match.get("bbox_xyxy"), observation.width, observation.height)
                overlap = box_iou(primary_box, match["bbox_xyxy"])
                require(overlap >= .3 and finite(match.get("IoU")) and abs(overlap - match["IoU"]) <= 1e-6,
                    "secondary-veto-independent-person-IoU")
                recomputed.append(dict(bbox_xyxy=copy.deepcopy(match["bbox_xyxy"]), score=match["score"], label_id=1, IoU=overlap))
            true_items[key] = dict(source_frame=observation.frame, source_timestamp=observation.timestamp,
                source_RGB_sha256=packet["RGB_sha256"], primary_track_id=key, primary_bbox_xyxy=copy.deepcopy(primary_box),
                primary_score=item["confidence"], observation_count=expected_count,
                matched_secondary_persons=recomputed, secondary_source_identity=copy.deepcopy(secondary_identity))
        return true_items, decisions

    def _weak_current_before_original(self, item):
        key = item["track_id"]
        if key in self.secondary_veto_ids or item.get("kind") != "pedestrian":
            return False
        footprint = self._footprint(item)
        ledger = self.tentative_people.get(key)
        return bool(.5 <= item.get("confidence", 0.) < .85 and item.get("in_path") is True
            and footprint is not None and footprint[0] - self.front > 8.
            and (ledger is None or not ledger["strong"] and ledger["seen"] + 1 < 3))

    def _force_existing_veto(self, key):
        ledger, memory = self.tentative_people.get(key), self.tracks.get(key)
        if ledger is None or memory is None or memory["item"].get("kind") != "pedestrian":
            return False
        ledger["strong"] = True
        memory["threat_relevant"] = True
        return True

    def _memory_update(self, raw, prediction, observation):
        # The secondary producer cannot mutate primary driving observations.
        packet = self.secondary_scheduler.update(copy.deepcopy(raw), observation) if self.secondary_scheduler is not None else None
        true_items, decisions = self._verified_true_confirmations(packet, raw, observation)
        pending = {item["track_id"]: true_items[item["track_id"]] for item in raw["objects"]
            if item["track_id"] in true_items and self._weak_current_before_original(item)}
        forced_before = [key for key in sorted(self.secondary_veto_ids) if self._force_existing_veto(key)]
        result, notes = super()._memory_update(raw, prediction, observation)
        newly_armed = []
        for key, receipt in pending.items():
            if self._force_existing_veto(key):
                self.secondary_veto_ids[key] = receipt
                newly_armed.append(key)
        # The original positive same-ID 0.2s lateral clearance remains the exit.
        # Never recreate an original-deleted memory or refresh last_seen.
        cleared = [key for key in sorted(self.secondary_veto_ids) if key not in self.tracks]
        for key in cleared:
            del self.secondary_veto_ids[key]
        forced_after = [key for key in sorted(self.secondary_veto_ids) if self._force_existing_veto(key)]
        notes["secondary_person_veto"] = dict(version=CONFIG["version"], GT_inputs=False,
            config=CONFIG, inference_receipt=copy.deepcopy(packet), decisions=decisions,
            newly_armed=sorted(newly_armed), forced_before_original=forced_before, forced_after_original=forced_after,
            original_current_clearance_deletions_preserved=cleared,
            active_veto_receipts=copy.deepcopy(self.secondary_veto_ids),
            causal_flag_source="secondary person retirement veto;not natural primary confirmation or clearance evidence",
            original_raw_objects_scores_boxes_and_ranges_unchanged=True,
            secondary_unmatched_objects_or_signals_used=False, new_retirement_authorization=False,
            False_None_exact_identity_scope="from reset without earlier True", default_profile_promoted=False)
        return result, notes


class SecondaryHypothesisTrafficBridge(HypothesisTrafficBridge):
    """Explicit composition of optional person veto and unchanged signal logic."""
    def __init__(self, *, nominal_calibration, front_bumper_x_m=2.4251085,
                 ego_half_width_m=1., secondary_scheduler=None,
                 secondary_backend_source_sha256=None):
        self.range = SecondaryPersonVetoBridge(
            nominal_calibration=nominal_calibration,
            front_bumper_x_m=front_bumper_x_m,
            ego_half_width_m=ego_half_width_m,
            secondary_scheduler=secondary_scheduler,
            secondary_backend_source_sha256=secondary_backend_source_sha256)
        self.signal = SignalConfirmationAdapter(nominal_calibration=nominal_calibration)
        self.reset()
