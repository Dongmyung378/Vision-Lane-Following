"""Closed-loop outcomes, failure-inclusive pooling, time windows, and publication."""
import copy
import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from carla_tools.evaluation import (EvaluationConfig, EvaluationError, EvaluationTracker,
                                    aggregate_metrics, deduplicate_events, evaluate_episode,
                                    evaluation_config_from_dict, load_evaluation_config, write_metric_csv)
from scripts.evaluate_episodes import main


def row(index, *, reference=None, speed=2., lateral=.1, heading=1., margin=.2, steer=0., x=None):
    return {"frame": 100 + index, "timestamp": 5. + .05 * index,
            "tracking": {"reference_arc_m": .1 * index if reference is None else reference,
                         "lateral_error_m": lateral, "heading_error_deg": heading},
            "vehicle": {"transform": {"location": {"x": .1 * index if x is None else x, "y": 0., "z": 0.}},
                        "applied_control": {"steer": steer}},
            "speed_mps": speed, "body_lane_margin_m": margin,
            "latency_ms": {"inference_ms": float(index + 1), "pipeline_ms": float(index + 3)}}


def collision(frame=100, actor=7):
    return {"kind": "collision", "frame": frame, "other_actor_id": actor,
            "impulse": {"x": 1., "y": 0., "z": 0.}}


def marking(kind="Solid", color="White"):
    return {"type": kind, "color": color, "lane_change": "NONE", "width_m": .1}


def metadata(reason=None):
    result = {"episode_id": "synthetic", "route_id": "route", "method": "B0", "condition": "normal",
              "evaluation_route_length_m": 2., "evaluation_start_reference_m": 0.}
    if reason is not None:
        result["termination_reason"] = reason
    return result


class EvaluationConfigurationTests(unittest.TestCase):
    def test_checked_in_defaults_hash_and_exact_fields(self):
        config = load_evaluation_config()
        self.assertEqual(config, EvaluationConfig())
        self.assertEqual(len(config.sha256), 64)
        self.assertNotEqual(config.sha256, replace(config, timeout_seconds=10).sha256)
        for mutation in (lambda v: v.pop("timeout_seconds"), lambda v: v.update(extra=1)):
            value = config.to_dict()
            mutation(value)
            with self.assertRaises(EvaluationError):
                evaluation_config_from_dict(value)

    def test_invalid_numeric_values_boolean_version_and_windows_rejected(self):
        for name, value in (("timeout_seconds", True), ("timeout_seconds", math.nan),
                            ("minimum_future_horizon_m", -1), ("fixed_delta_seconds", 0),
                            ("schema_version", True), ("schema_version", 2),
                            ("recovery_hold_seconds", 6), ("timestamp_tolerance_seconds", .05)):
            raw = EvaluationConfig().to_dict()
            raw[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(EvaluationError):
                evaluation_config_from_dict(raw)

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for value in ('{"schema_version":1,"schema_version":1}', '{"timeout_seconds":NaN}'):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(EvaluationError):
                    load_evaluation_config(path)


class EvaluationTerminationTests(unittest.TestCase):
    def tracker(self, **kwargs):
        return EvaluationTracker(EvaluationConfig(), 100., **kwargs)

    def test_route_tolerance_clipped_nondecreasing_progress_and_actual_distance(self):
        tracker = EvaluationTracker(EvaluationConfig(), 2., start_reference_m=10.)
        tracker.update(row(0, reference=10., x=0.))
        tracker.update(row(1, reference=11., x=1.))
        tracker.update(row(2, reference=10.5, x=.5))
        outcome = tracker.update(row(3, reference=11.5, x=1.5))
        self.assertEqual(outcome["reason"], "route_completed")
        self.assertEqual(outcome["progress_m"], 1.5)
        self.assertEqual(outcome["completion_pct"], 75.)
        self.assertEqual(outcome["actual_distance_m"], 2.5)

    def test_precedence_retains_all_simultaneous_flags(self):
        tracker = EvaluationTracker(EvaluationConfig(), 2.)
        value = tracker.update(row(0, reference=2), events=[collision()], intervention=True,
                               technical_failure=True, remaining_horizon_m=14)
        self.assertEqual(value["reason"], "technical_failure")
        self.assertEqual(value["flags"], ["technical_failure", "collision", "intervention", "route_completed", "route_horizon_exhausted"])
        metric = tracker.finalize(metadata("technical_failure"))
        self.assertTrue(metric["integrity_valid"])
        self.assertFalse(metric["driving_success"])

    def test_footprint_departure_requires_continuous_half_second_and_resets(self):
        tracker = self.tracker()
        for index in range(10):
            self.assertFalse(tracker.update(row(index, margin=-.01))["terminated"])
        tracker.update(row(10, margin=0.))
        for index in range(11, 21):
            self.assertFalse(tracker.update(row(index, margin=-.01))["terminated"])
        self.assertEqual(tracker.update(row(21, margin=-.01))["reason"], "lane_departure")

    def test_center_inside_does_not_override_footprint(self):
        tracker = self.tracker()
        for index in range(11):
            value = tracker.update(row(index, lateral=0., margin=-.2))
        self.assertEqual(value["reason"], "lane_departure")

    def test_stall_requires_low_speed_and_little_progress_full_window(self):
        tracker = self.tracker()
        for index in range(101):
            value = tracker.update(row(index, reference=0., speed=.09, x=0.))
        self.assertEqual(value["reason"], "stalled")
        moving = self.tracker()
        for index in range(121):
            self.assertFalse(moving.update(row(index, reference=.01 * index, speed=.09))["terminated"])

    def test_timeout_and_horizon_boundary(self):
        config = replace(EvaluationConfig(), timeout_seconds=.1)
        tracker = EvaluationTracker(config, 100.)
        self.assertFalse(tracker.update(row(0), remaining_horizon_m=15)["terminated"])
        self.assertFalse(tracker.update(row(1), remaining_horizon_m=15)["terminated"])
        self.assertEqual(tracker.update(row(2), remaining_horizon_m=15)["reason"], "timeout")
        tracker = self.tracker()
        self.assertEqual(tracker.update(row(0), remaining_horizon_m=14.99)["reason"], "route_horizon_exhausted")

    def test_invalid_observation_preserves_last_valid_distance_and_samples(self):
        tracker = self.tracker()
        tracker.update(row(0))
        tracker.update(row(1))
        bad = row(2)
        bad["tracking"]["lateral_error_m"] = math.nan
        value = tracker.update(bad)
        self.assertEqual(value["reason"], "technical_failure")
        metric = tracker.finalize(metadata())
        self.assertEqual(metric["sample_count"], 2)
        self.assertAlmostEqual(metric["actual_distance_m"], .1)
        self.assertFalse(metric["integrity_valid"])

    def test_stale_skipped_nonmonotonic_and_delayed_frames_fail(self):
        for mutation in (lambda v: v.update(frame=100), lambda v: v.update(frame=102),
                         lambda v: v.update(timestamp=5.), lambda v: v.update(timestamp=5.1)):
            tracker = self.tracker()
            tracker.update(row(0))
            invalid = row(1)
            mutation(invalid)
            with self.subTest(mutation=mutation):
                self.assertEqual(tracker.update(invalid)["reason"], "technical_failure")
                self.assertEqual(len(tracker.rows), 1)

    def test_flat_observation_aliases(self):
        original = row(0)
        value = {"frame": original["frame"], "time_seconds": original["timestamp"], "reference_m": 0.,
                 "lateral_error_m": .1, "heading_error_deg": 1., "speed_mps": 2., "body_lane_margin_m": .2,
                 "actor_transform": original["vehicle"]["transform"], "applied_control": {"steer": .1}}
        self.assertFalse(self.tracker().update(value)["terminated"])

    def test_observation_after_termination_is_integrity_failure(self):
        tracker = self.tracker()
        tracker.update(row(0), technical_failure=True)
        self.assertEqual(tracker.update(row(1))["reason"], "technical_failure")
        self.assertIn("observation-after-termination", tracker.integrity_errors)


class EvaluationEventsAndRecoveryTests(unittest.TestCase):
    def test_collision_dedup_exact_identity_only_and_source_untouched(self):
        events = [collision(), collision(), collision(actor=8), collision(frame=101)]
        original = copy.deepcopy(events)
        selected = deduplicate_events(events)
        self.assertEqual(len(selected), 3)
        selected[0]["impulse"]["x"] = 8.
        self.assertEqual(events, original)

    def test_lane_marking_set_order_dedup_but_no_cross_frame_window(self):
        a, b = marking(), marking("Broken", "Yellow")
        events = [{"kind": "lane", "frame": 100, "markings": [a, b]},
                  {"kind": "lane", "frame": 100, "markings": [b, a]},
                  {"kind": "lane", "frame": 101, "markings": [a, b]},
                  {"kind": "lane", "frame": 100, "markings": [a]}]
        self.assertEqual(len(deduplicate_events(events)), 3)

    def test_warmup_and_post_stop_events_excluded_from_drive_counts(self):
        tracker = EvaluationTracker(EvaluationConfig(), 2.)
        tracker.update(row(0, reference=0.))
        tracker.update(row(1, reference=1.5))
        metric = tracker.finalize(metadata(), events=[collision(frame=99), collision(frame=102)])
        self.assertTrue(metric["driving_success"])
        self.assertEqual(metric["collision_count"], 0)
        self.assertEqual(len(metric["raw_events"]), 2)

    def test_late_collision_overrides_completion_preserves_samples(self):
        rows = [row(0, reference=0.), row(1, reference=.5), row(2, reference=1.), row(3, reference=1.5)]
        metric = evaluate_episode(metadata("collision"), rows, [collision(frame=101)])
        self.assertEqual(metric["termination_reason"], "collision")
        self.assertEqual(metric["termination_flags"], ["collision", "route_completed"])
        self.assertEqual(metric["sample_count"], 4)
        self.assertAlmostEqual(metric["actual_distance_m"], .3)

    def test_recovery_continuous_hold_not_first_in_tolerance(self):
        tracker = EvaluationTracker(EvaluationConfig(), 100., condition="left_offset")
        for index in range(20):
            tracker.update(row(index))
            self.assertIsNone(tracker.recovery_time_seconds)
        tracker.update(row(20))
        self.assertAlmostEqual(tracker.recovery_time_seconds, 1.)

    def test_recovery_reset_and_late_hold_rejected(self):
        tracker = EvaluationTracker(EvaluationConfig(), 100.)
        for index in range(101):
            tracker.update(row(index, lateral=.3 if index < 85 else .1))
        self.assertIsNone(tracker.recovery_time_seconds)
        for index in range(101, 110):
            tracker.update(row(index))
        self.assertIsNone(tracker.recovery_time_seconds)
        reset = EvaluationTracker(EvaluationConfig(), 100.)
        for index in range(31):
            reset.update(row(index, lateral=.3 if index == 10 else .1))
        self.assertIsNone(reset.recovery_time_seconds)
        reset.update(row(31))
        self.assertAlmostEqual(reset.recovery_time_seconds, 1.55)

    def test_intervention_free_distance_stops_at_intervention(self):
        tracker = EvaluationTracker(EvaluationConfig(), 100.)
        tracker.update(row(0))
        tracker.update(row(1))
        tracker.update(row(2), intervention=True)
        metric = tracker.finalize(metadata("intervention"))
        self.assertAlmostEqual(metric["intervention_free_distance_m"], .2)

    def test_invalid_event_is_technical_failure(self):
        tracker = EvaluationTracker(EvaluationConfig(), 100.)
        tracker.update(row(0))
        value = tracker.update(row(1), events=[{"kind": "collision", "frame": 101, "other_actor_id": True}])
        self.assertEqual(value["reason"], "technical_failure")
        self.assertEqual(len(tracker.rows), 1)

    def test_malformed_event_does_not_erase_other_valid_collisions(self):
        tracker = EvaluationTracker(EvaluationConfig(), 100.)
        tracker.update(row(0))
        malformed = collision()
        malformed["timestamp"] = math.nan
        metric = tracker.finalize(metadata(), events=[collision(), malformed])
        self.assertEqual(metric["termination_reason"], "technical_failure")
        self.assertEqual(metric["collision_count"], 1)
        self.assertEqual(metric["invalid_event_indices"], [1])
        json.dumps(metric, allow_nan=False)


class EvaluationMetricsAndPublicationTests(unittest.TestCase):
    def test_steering_rates_acceleration_latency_and_missing_sample_semantics(self):
        rows = [row(0, reference=0., steer=0.), row(1, reference=.5, steer=.1), row(2, reference=1.5, steer=.4)]
        metric = evaluate_episode(metadata("route_completed"), rows, [])
        self.assertAlmostEqual(metric["steering_rate_abs_per_second"]["mean"], 4.)
        self.assertAlmostEqual(metric["steering_acceleration_abs_per_second2"]["max"], 80.)
        self.assertEqual(metric["latency_ms"]["inference_ms"]["median"], 2.)
        self.assertAlmostEqual(metric["latency_ms"]["inference_ms"]["p95"], 2.9)
        self.assertNotIn("tick_to_rgb_ms", metric["latency_ms"])

    def test_failed_episodes_pool_samples_not_episode_means_and_distances(self):
        successful = evaluate_episode(metadata(), [row(i, reference=.1*i, lateral=.1) for i in range(16)], [])
        failed_meta = metadata("collision")
        failed_meta["episode_id"] = "failed"
        failed = evaluate_episode(failed_meta, [row(0, lateral=1.), row(1, lateral=1.)], [collision(frame=101)])
        aggregate = aggregate_metrics([successful, failed])["groups"][0]
        self.assertEqual(aggregate["episode_count"], 2)
        self.assertEqual(aggregate["successful_episodes"], 1)
        self.assertEqual(aggregate["failed_episodes"], 1)
        self.assertEqual(aggregate["sample_count"], 18)
        self.assertAlmostEqual(aggregate["lateral_abs_m"]["mean"], 3.6/18)
        self.assertAlmostEqual(aggregate["collision_per_km"], 1/(1.6/1000))

    def test_zero_distance_rates_are_na_even_collision_episode(self):
        metric = evaluate_episode(metadata("collision"), [row(0)], [collision()])
        self.assertEqual(metric["collision_count"], 1)
        self.assertIsNone(metric["collision_per_km"])
        self.assertIsNone(aggregate_metrics([metric])["groups"][0]["collision_per_km"])

    def test_empty_and_unterminated_episode_remain_technical_rows(self):
        for rows in ([], [row(0), row(1)]):
            metric = evaluate_episode(metadata(), rows, [])
            self.assertEqual(metric["termination_reason"], "technical_failure")
            self.assertFalse(metric["integrity_valid"])

    def test_missing_invalid_route_metadata_still_produces_failure_rows(self):
        for meta in ({}, {"evaluation_route_length_m": None}, {"evaluation_route_length_m": math.nan}, None):
            metric = evaluate_episode(meta, [], [])
            self.assertEqual(metric["termination_reason"], "technical_failure")
            self.assertFalse(metric["integrity_valid"])
            self.assertIsNone(metric["completion_pct"])
            self.assertEqual(aggregate_metrics([metric])["groups"][0]["episode_count"], 1)

    def test_embedded_configuration_replays_short_timeout_and_verifies_hash(self):
        config = replace(EvaluationConfig(), timeout_seconds=.1)
        meta = metadata("timeout")
        meta.update(evaluation_config=config.to_dict(), evaluation_config_sha256=config.sha256)
        metric = evaluate_episode(meta, [row(0), row(1), row(2)], [])
        self.assertEqual(metric["termination_reason"], "timeout")
        self.assertTrue(metric["integrity_valid"])
        self.assertEqual(metric["evaluation_config_sha256"], config.sha256)
        mismatched = evaluate_episode(meta, [row(0), row(1), row(2)], [], config=EvaluationConfig())
        self.assertFalse(mismatched["integrity_valid"])
        self.assertEqual(mismatched["termination_reason"], "technical_failure")
        self.assertIn("evaluation-config-hash-mismatch", mismatched["integrity_errors"])

    def test_perception_failure_is_retained_and_integrity_is_separate(self):
        rows = [row(0), row(1)]
        rows[-1].update(technical_failure=True, command=None)
        metric = evaluate_episode(metadata("technical_failure"), rows, [])
        self.assertFalse(metric["driving_success"])
        self.assertTrue(metric["integrity_valid"])
        self.assertEqual(metric["sample_count"], 2)

    def test_unsubstantiated_completion_is_integrity_failure(self):
        metric = evaluate_episode(metadata("route_completed"), [row(0)], [])
        self.assertEqual(metric["termination_reason"], "technical_failure")
        self.assertIn("unsubstantiated-evaluation-termination", metric["integrity_errors"])

    def test_grouping_preserves_method_condition_and_config(self):
        first = evaluate_episode(metadata("collision"), [row(0)], [collision()])
        second = copy.deepcopy(first)
        second["method"] = "M1"
        third = copy.deepcopy(first)
        third["condition"] = "rain"
        fourth = copy.deepcopy(first)
        fourth["evaluation_config_sha256"] = "0" * 64
        self.assertEqual(len(aggregate_metrics([first, second, third, fourth])["groups"]), 4)

    def test_csv_exclusive_complete_rows_and_na(self):
        metric = evaluate_episode(metadata("collision"), [row(0)], [collision()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            aggregate = write_metric_csv([metric], path)
            saved = path.read_bytes()
            data = list(csv.DictReader(io.StringIO(saved.decode())))[0]
            self.assertEqual(data["collision_per_km"], "N/A")
            self.assertEqual(data["termination_reason"], "collision")
            self.assertEqual(aggregate["episode_count"], 1)
            with self.assertRaises(EvaluationError):
                write_metric_csv([metric], path)
            self.assertEqual(path.read_bytes(), saved)

    def test_publication_failure_does_not_create_partial_file(self):
        metric = evaluate_episode(metadata("collision"), [row(0)], [collision()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with patch("carla_tools.evaluation.os.link", side_effect=OSError("synthetic")), self.assertRaises(EvaluationError):
                write_metric_csv([metric], path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_replay_preserves_sources_including_truncated_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            episode.mkdir()
            meta_path, rows_path = episode / "metadata.json", episode / "steps.jsonl"
            meta_path.write_text(json.dumps(metadata()), encoding="utf-8")
            rows_path.write_text(json.dumps(row(0)) + "\n{truncated", encoding="utf-8")
            originals = (meta_path.read_bytes(), rows_path.read_bytes())
            output = Path(directory) / "metrics.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--episodes", str(episode), "--output", str(output)]), 0)
            with output.open(encoding="utf-8") as handle:
                data = list(csv.DictReader(handle))[0]
            self.assertEqual(data["termination_reason"], "technical_failure")
            self.assertEqual(data["sample_count"], "1")
            self.assertEqual((meta_path.read_bytes(), rows_path.read_bytes()), originals)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--episodes", str(episode), "--output", str(rows_path)]), 1)
            self.assertEqual((meta_path.read_bytes(), rows_path.read_bytes()), originals)

    def test_cli_native_episode_file_uses_frozen_config_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "native-episode"
            episode.mkdir()
            config = replace(EvaluationConfig(), timeout_seconds=10.)
            meta = metadata("route_completed")
            meta.update(evaluation_config=config.to_dict(), evaluation_config_sha256=config.sha256,
                        route_id="synthetic-native-route", method="B0")
            path = episode / "episode.json"
            path.write_text(json.dumps(meta), encoding="utf-8")
            (episode / "steps.jsonl").write_text("\n".join(json.dumps(row(index, reference=index * .5))
                                                          for index in range(4)), encoding="utf-8")
            source = path.read_bytes()
            output = Path(directory) / "metrics.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--episodes", str(episode), "--output", str(output)]), 0)
            with output.open(encoding="utf-8") as handle:
                metric = list(csv.DictReader(handle))[0]
            self.assertEqual(metric["route_id"], "synthetic-native-route")
            self.assertEqual(metric["method"], "B0")
            self.assertEqual(metric["termination_reason"], "route_completed")
            self.assertEqual(metric["evaluation_config_sha256"], config.sha256)
            self.assertEqual(metric["completion_pct"], "75.0")
            self.assertEqual(path.read_bytes(), source)

    def test_cli_refuses_ambiguous_metadata_without_source_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            episode.mkdir()
            native, generic = episode / "episode.json", episode / "metadata.json"
            native.write_text(json.dumps(metadata()), encoding="utf-8")
            generic.write_text(json.dumps(metadata("technical_failure")), encoding="utf-8")
            original = (native.read_bytes(), generic.read_bytes())
            output = Path(directory) / "metrics.csv"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["--episodes", str(episode), "--output", str(output)]), 1)
            self.assertEqual(json.loads(stdout.getvalue())["code"], "ambiguous-episode-metadata")
            self.assertFalse(output.exists())
            self.assertEqual((native.read_bytes(), generic.read_bytes()), original)


if __name__ == "__main__":
    unittest.main()
