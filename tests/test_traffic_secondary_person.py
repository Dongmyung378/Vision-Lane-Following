"""Synthetic safety and opt-in contracts; no models or simulator are started."""
import copy
import hashlib
import inspect
import math
import subprocess
import sys
import unittest

from carla_tools.traffic_behavior import TrafficSupervisor
from carla_tools.traffic_hypothesis import HypothesisMemoryBridge
from carla_tools.traffic_perception import RGBObservation, to_traffic_observation
from carla_tools.traffic_secondary_confirmation import (
    PRIMARY_SHA, SECONDARY_SHA, RecordedSecondaryBackend, ResNet50V2Backend,
    SecondaryPersonConfirmationScheduler, implementation_source_sha256,
)
from carla_tools.traffic_secondary_veto import (
    SecondaryHypothesisTrafficBridge, SecondaryPersonVetoBridge,
)


def calibration():
    c, s = math.cos(math.radians(12.)), math.sin(math.radians(12.))
    return dict(width=640, height=360,
        intrinsics=dict(fx=320., fy=320., cx=320., cy=180.),
        camera_to_actor_matrix=[[c, 0., s, 1.5], [0., 1., 0., 0.],
            [-s, 0., c, 2.4], [0., 0., 0., 1.]])


def person(key="person", **changes):
    value = dict(track_id=key, label="person", kind="pedestrian", confidence=.65,
        bbox_xyxy=[288., 136., 306., 169.], distance_m=12.67,
        lateral_y_m=-.96, distance_uncertainty_m=5.84,
        lateral_uncertainty_m=.6, ground_valid=True,
        closing_speed_mps=None, in_path=True)
    value.update(changes)
    return value


def observation(index, speed=0.):
    return RGBObservation(bytes([75]) * 640 * 360 * 3,
        640, 360, index, index * .05, speed)


def raw(index, items=()):
    return dict(frame=index, timestamp=index * .05, objects=copy.deepcopy(list(items)),
        detections=[dict(label=p["label"], confidence=p["confidence"],
            bbox_xyxy=copy.deepcopy(p["bbox_xyxy"])) for p in items],
        GT_inputs=False, perception_valid=True, signal="none", signal_confidence=0.,
        stop_required=False, stop_line_distance_m=None, stop_line_confidence=0., stop_line={})


def step(bridge, index, items=(), speed=0.):
    return bridge.update(raw(index, items),
        dict(status="PASS", waypoints_ego_m=[[3., 0.], [40., 0.]]),
        observation(index, speed))


PRIMARY = dict(architecture="fasterrcnn_mobilenet", weights_sha256=PRIMARY_SHA)


def secondary_identity():
    return dict(architecture="fasterrcnn_resnet50_fpn_v2", weights_sha256=SECONDARY_SHA,
        implementation_source_sha256=implementation_source_sha256())


class RecordedFixtureBackend:
    recorded_only = True

    def __init__(self, boxes=()):
        self.identity = secondary_identity()
        self.boxes, self.calls = copy.deepcopy(list(boxes)), 0

    def predict(self, obs):
        self.calls += 1
        return copy.deepcopy(self.boxes)


def make_pair(boxes=()):
    backend = RecordedFixtureBackend(boxes)
    scheduler = SecondaryPersonConfirmationScheduler(backend, calibration(),
        primary_source_identity=PRIMARY)
    bridge = SecondaryPersonVetoBridge(nominal_calibration=calibration(),
        secondary_scheduler=scheduler)
    return bridge, scheduler, backend


def matching_person():
    return dict(bbox_xyxy=person()["bbox_xyxy"], label_id=1, score=.9)


class DecisionScheduler:
    """A trusted, source-labelled synthetic producer for validation tests."""
    def __init__(self, value, mutate=None):
        self.value, self.mutate = value, mutate
        self.reset()

    def reset(self):
        self.seen = {}

    def update(self, source, obs):
        confirmations = []
        for p in source["objects"]:
            key = p["track_id"]
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.seen[key] > 2:
                continue
            confirmations.append(dict(primary_track_id=key,
                primary_bbox_xyxy=copy.deepcopy(p["bbox_xyxy"]), primary_score=p["confidence"],
                observation_count=self.seen[key], secondary_person_confirmation=self.value,
                matched_secondary_persons=[dict(bbox_xyxy=copy.deepcopy(p["bbox_xyxy"]),
                    score=.9, label_id=1, IoU=1.)] if self.value is True else []))
        packet = dict(version="primary-ROI-positive-secondary-person-confirmation-v1",
            frame=obs.frame, timestamp=obs.timestamp, RGB_sha256=hashlib.sha256(obs.rgb).hexdigest(),
            GT_inputs=False, primary_source_identity=copy.deepcopy(PRIMARY),
            secondary_source_identity=secondary_identity(), confirmations=confirmations)
        if self.mutate:
            self.mutate(packet, source)
        return packet


class SecondaryPersonTests(unittest.TestCase):
    def test_import_has_no_torch_and_backend_device_is_required(self):
        subprocess.run([sys.executable, "-c",
            "import sys; import carla_tools.traffic_secondary_veto; "
            "assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"],
            check=True, capture_output=True)
        self.assertIs(inspect.signature(ResNet50V2Backend).parameters["device"].default,
            inspect.Parameter.empty)
        with self.assertRaisesRegex(ValueError, "explicit-valid-detection-device"):
            ResNet50V2Backend("unused", expected_sha256=SECONDARY_SHA, device="automatic")

    def test_max_two_calls_and_one_shared_call_per_frame(self):
        _, scheduler, backend = make_pair()
        for i in range(4):
            packet = scheduler.update(raw(i, [person(), person("other")]), observation(i))
        self.assertEqual(backend.calls, 2)
        self.assertEqual(scheduler.tracks["person"]["seen"], 4)
        self.assertEqual(packet["confirmations"], [])

    def test_first_positive_stops_additional_calls(self):
        _, scheduler, backend = make_pair([matching_person()])
        first = scheduler.update(raw(0, [person()]), observation(0))
        second = scheduler.update(raw(1, [person()]), observation(1))
        self.assertTrue(first["confirmations"][0]["secondary_person_confirmation"])
        self.assertEqual(second["confirmations"], [])
        self.assertEqual(backend.calls, 1)

    def test_unrelated_objects_never_enter_primary_or_confirm(self):
        boxes = [dict(bbox_xyxy=[0., 0., 5., 5.], score=.99, label_id=1)]
        boxes += [dict(bbox_xyxy=person()["bbox_xyxy"], score=.99, label_id=n) for n in (3, 10)]
        new, scheduler, _ = make_pair(boxes)
        old = HypothesisMemoryBridge(nominal_calibration=calibration())
        original = raw(0, [person()])
        frozen = copy.deepcopy(original)
        packet = scheduler.update(original, observation(0))
        self.assertEqual(original, frozen)
        self.assertFalse(packet["confirmations"][0]["secondary_person_confirmation"])
        self.assertEqual(len(packet["inference_receipt"]["raw_detections"]), 3)
        new.reset()
        self.assertEqual(step(old, 0, [person()])[0], step(new, 0, [person()])[0])

    def test_out_of_path_sightings_count_and_strong_near_unknown_are_excluded(self):
        _, scheduler, backend = make_pair()
        for i in range(2):
            scheduler.update(raw(i, [person(in_path=False)]), observation(i))
        scheduler.update(raw(2, [person()]), observation(2))
        self.assertEqual(backend.calls, 0)
        for p in (person(confidence=.9), person(ground_valid=False),
                  person(bbox_xyxy=[280., 170., 330., 250.])):
            with self.subTest(person=p):
                _, scheduler, backend = make_pair()
                scheduler.update(raw(0, [p]), observation(0))
                scheduler.update(raw(1, [person()]), observation(1))
                self.assertEqual(backend.calls, 0)

    def test_False_None_and_disabled_preserve_commands_state_and_acknowledgement(self):
        for decision in (False, None, "disabled"):
            old = HypothesisMemoryBridge(nominal_calibration=calibration())
            new = SecondaryPersonVetoBridge(nominal_calibration=calibration(),
                secondary_scheduler=None if decision == "disabled" else DecisionScheduler(decision))
            supervisors = [TrafficSupervisor(), TrafficSupervisor()]
            for i in range(18):
                results = [step(b, i, [person()] if i < 2 else []) for b in (old, new)]
                new_notes = copy.deepcopy(results[1][1])
                new_notes.pop("secondary_person_veto")
                self.assertEqual(results[0], (results[1][0], new_notes))
                commands = [s.step(to_traffic_observation(result[0]), 0.,
                    dict(throttle=.5, brake=0., steer=0., hand_brake=False, reverse=False,
                        diagnostics=dict(effective_target_speed_mps=3.)), timestamp=i * .05)
                    for s, result in zip(supervisors, results)]
                self.assertEqual(*commands)
                for b, command in zip((old, new), commands):
                    b.acknowledge(command)
                self.assertEqual(old.tracks, new.tracks)
                self.assertEqual(old.tentative_people, new.tentative_people)

    def test_positive_dropout_veto_covers_both_absence_paths_without_refresh(self):
        old = HypothesisMemoryBridge(nominal_calibration=calibration())
        new, _, backend = make_pair([matching_person()])
        for i in range(16):
            a, _ = step(old, i, [person()] if i == 0 else [])
            b, notes = step(new, i, [person()] if i == 0 else [])
        self.assertNotIn("person", old.tracks)
        self.assertIn("person", new.secondary_veto_ids)
        self.assertTrue(new.tentative_people["person"]["strong"])
        self.assertTrue(new.tracks["person"]["threat_relevant"])
        self.assertEqual(new.tracks["person"]["last_seen"], 0.)
        self.assertFalse(b["perception_valid"])
        self.assertEqual(backend.calls, 1)
        self.assertFalse(notes["secondary_person_veto"]["new_retirement_authorization"])

    def test_current_same_ID_positive_clearance_still_requires_point_two_seconds(self):
        new, _, _ = make_pair([matching_person()])
        step(new, 0, [person()])
        outside = person(bbox_xyxy=[500., 136., 518., 169.], lateral_y_m=8., in_path=False)
        step(new, 1, [outside])
        self.assertIn("person", new.tracks)
        for i in range(2, 8):
            result, _ = step(new, i, [outside])
        self.assertNotIn("person", new.tracks)
        self.assertNotIn("person", new.secondary_veto_ids)
        self.assertTrue(result["perception_valid"])

    def test_false_positive_agreement_can_hold_forever_and_fail_completion(self):
        # Both detectors can mistake the same background for a person. This
        # valid positive receipt must not be sold as a performance guarantee.
        new, _, _ = make_pair([matching_person()])
        for i in range(40):
            result, _ = step(new, i, [person()] if i == 0 else [])
        command = TrafficSupervisor().step(to_traffic_observation(result), 0.,
            dict(throttle=.5, brake=0., steer=0., hand_brake=False, reverse=False,
                diagnostics=dict(effective_target_speed_mps=3.)), timestamp=1.95)
        self.assertEqual(command["brake"], 1.)
        self.assertIn("person", new.tracks)

    def test_invalid_positive_provenance_or_match_is_rejected(self):
        mutations = [
            lambda p, _: p.update(frame=99),
            lambda p, _: p.update(timestamp=.1),
            lambda p, _: p.update(RGB_sha256="0" * 64),
            lambda p, _: p.update(GT_inputs=True),
            lambda p, _: p["secondary_source_identity"].update(implementation_source_sha256="0" * 64),
            lambda p, _: p["primary_source_identity"].update(weights_sha256="0" * 64),
            lambda p, _: p["confirmations"][0].update(observation_count=2),
            lambda p, _: p["confirmations"][0].update(secondary_person_confirmation=1),
            lambda p, _: p["confirmations"][0]["matched_secondary_persons"][0].update(score=.49),
            lambda p, _: p["confirmations"][0]["matched_secondary_persons"][0].update(IoU=.5),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                new = SecondaryPersonVetoBridge(nominal_calibration=calibration(),
                    secondary_scheduler=DecisionScheduler(True, mutate))
                with self.assertRaises(ValueError):
                    step(new, 0, [person()])

    def test_recorded_RGB_source_binding_and_missing_frame(self):
        obs = observation(0)
        source = dict(frame=0, timestamp=0., RGB_sha256=hashlib.sha256(obs.rgb).hexdigest(),
            weights_sha256=SECONDARY_SHA, detections=[matching_person()])
        backend = RecordedSecondaryBackend({0: source}, secondary_identity())
        scheduler = SecondaryPersonConfirmationScheduler(backend, calibration(), primary_source_identity=PRIMARY)
        packet = scheduler.update(raw(0, [person()]), obs)
        self.assertEqual(packet["new_NN_calls"], 0)
        self.assertTrue(packet["confirmations"][0]["secondary_person_confirmation"])
        self.assertEqual(packet["inference_receipt"]["recorded_source_receipt_sha256"].__len__(), 64)
        with self.assertRaisesRegex(ValueError, "NOOBS"):
            backend.predict(observation(1))
        bad = copy.deepcopy(source)
        bad["RGB_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "same-frame-RGB"):
            RecordedSecondaryBackend({0: bad}, secondary_identity()).predict(obs)
        native = copy.deepcopy(packet["inference_receipt"])
        native["secondary_source_identity"]["implementation_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "implementation-identity"):
            RecordedSecondaryBackend({0: native}, secondary_identity()).predict(obs)

    def test_reset_clears_all_epochs_and_composition_is_explicit(self):
        new, scheduler, backend = make_pair([matching_person()])
        step(new, 0, [person()])
        new.reset()
        self.assertEqual(new.secondary_veto_ids, {})
        self.assertEqual(scheduler.tracks, {})
        self.assertIsNone(scheduler.previous)
        combined = SecondaryHypothesisTrafficBridge(nominal_calibration=calibration())
        self.assertIsNone(combined.range.secondary_scheduler)


if __name__ == "__main__":
    unittest.main()
