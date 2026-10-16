"""CPU fixtures exercise synchronization boundaries and failure-inclusive pooling."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from carla_tools.pipeline_latency import (
    PipelineFrameTimer, PipelineLatencyError, STAGES, aggregate_pipeline_latency,
    validate_pipeline_timing,
)


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, milliseconds):
        self.value += milliseconds / 1000


def timing(milliseconds, *, failed=False):
    clock = Clock()
    timer = PipelineFrameTimer(clock=clock)
    timer.measure("bgra_conversion_ms", lambda: clock.advance(1))
    timer.measure("learned_preprocessing_ms", lambda: clock.advance(1))
    def forward():
        clock.advance(milliseconds - 3)
        if failed:
            raise RuntimeError("fixture-inference-failed")
    try:
        timer.measure("forward_decode_ms", forward)
    except RuntimeError:
        pass
    timer.measure("controller_ms", lambda: clock.advance(1))
    return timer.finish()


def episode(identity, values, *, method="hybrid", case="right_curve", success=True):
    return {"episode_id": identity, "method": method, "case": case,
            "driving_success": success, "expected_frame_count": len(values),
            "rows": [{"frame": index + 10, "latency": timing(value)} for index, value in enumerate(values)]}


class PipelineFrameTimerTests(unittest.TestCase):
    def test_cpu_wall_boundary_retains_overhead_and_skipped_correction(self):
        clock = Clock()
        timer = PipelineFrameTimer(clock=clock)
        answer = timer.measure("bgra_conversion_ms", lambda: (clock.advance(3), b"rgb")[1])
        self.assertEqual(answer, b"rgb")
        clock.advance(2)  # Input validation and orchestration remain in total.
        timer.measure("forward_decode_ms", lambda: clock.advance(5))
        timer.measure("controller_ms", lambda: clock.advance(1))
        record = validate_pipeline_timing(timer.finish())
        self.assertAlmostEqual(record["total_processing_ms"], 11)
        self.assertAlmostEqual(record["pipeline_overhead_ms"], 2)
        self.assertEqual(record["stages_ms"]["near_correction_ms"], 0)
        self.assertNotIn("near_correction_ms", record["executed_stages"])
        self.assertEqual(record["synchronized_stages"], [])

    def test_cuda_callback_brackets_operations_and_prior_work_is_excluded(self):
        clock, order = Clock(), []
        def synchronize():
            order.append("sync")
            clock.advance(2)
        timer = PipelineFrameTimer(device="cuda", synchronize=synchronize, clock=clock)
        def operation():
            order.append("forward")
            clock.advance(5)
            return [[3., 0.]]
        self.assertEqual(timer.measure("forward_decode_ms", operation, gpu=True), [[3., 0.]])
        record = validate_pipeline_timing(timer.finish())
        self.assertEqual(order, ["sync", "sync", "forward", "sync"])
        self.assertAlmostEqual(record["stages_ms"]["forward_decode_ms"], 7)
        self.assertAlmostEqual(record["total_processing_ms"], 9)
        self.assertAlmostEqual(record["pipeline_overhead_ms"], 2)
        self.assertEqual(record["synchronized_stages"], ["forward_decode_ms"])

    def test_failed_cuda_operation_still_waits_and_records_elapsed_time(self):
        clock, calls = Clock(), []
        def synchronize():
            calls.append("sync")
            clock.advance(1)
        timer = PipelineFrameTimer(device="cuda", synchronize=synchronize, clock=clock)
        def fail():
            clock.advance(8)
            raise RuntimeError("fixture-async-failure")
        with self.assertRaisesRegex(RuntimeError, "fixture-async-failure"):
            timer.measure("forward_decode_ms", fail, gpu=True)
        record = validate_pipeline_timing(timer.finish())
        self.assertEqual(len(calls), 3)
        self.assertEqual(record["failed_stages"], ["forward_decode_ms"])
        self.assertAlmostEqual(record["stages_ms"]["forward_decode_ms"], 9)

    def test_cuda_sync_failure_is_retained_as_failed_stage(self):
        clock, calls = Clock(), []
        def synchronize():
            calls.append("sync")
            clock.advance(1)
            if len(calls) == 3:
                raise RuntimeError("fixture-sync-failure")
        timer = PipelineFrameTimer(device="cuda", synchronize=synchronize, clock=clock)
        with self.assertRaisesRegex(RuntimeError, "fixture-sync-failure"):
            timer.measure("forward_decode_ms", lambda: clock.advance(2), gpu=True)
        record = timer.finish()
        self.assertEqual(record["failed_stages"], ["forward_decode_ms"])
        self.assertAlmostEqual(record["stages_ms"]["forward_decode_ms"], 3)

    def test_cuda_forward_cannot_be_timed_without_explicit_sync(self):
        with self.assertRaisesRegex(PipelineLatencyError, "synchronizer-required"):
            PipelineFrameTimer(device="cuda")
        timer = PipelineFrameTimer(device="cuda", synchronize=lambda: None)
        with self.assertRaisesRegex(PipelineLatencyError, "forward-requires-synchronization"):
            timer.measure("forward_decode_ms", lambda: None)
        with self.assertRaisesRegex(PipelineLatencyError, "requires-cuda"):
            PipelineFrameTimer().measure("forward_decode_ms", lambda: None, gpu=True)

    def test_nested_repeated_and_after_finish_stages_are_rejected(self):
        timer = PipelineFrameTimer()
        def nested():
            with self.assertRaisesRegex(PipelineLatencyError, "nested"):
                timer.measure("controller_ms", lambda: None)
            with self.assertRaisesRegex(PipelineLatencyError, "active-stage"):
                timer.finish()
        timer.measure("bgra_conversion_ms", nested)
        with self.assertRaisesRegex(PipelineLatencyError, "repeated"):
            timer.measure("bgra_conversion_ms", lambda: None)
        timer.finish()
        with self.assertRaisesRegex(PipelineLatencyError, "finished"):
            timer.measure("controller_ms", lambda: None)

    def test_finish_and_validation_return_detached_copies(self):
        timer = PipelineFrameTimer()
        timer.measure("controller_ms", lambda: None)
        record = timer.finish()
        record["stages_ms"]["controller_ms"] = -1
        record["executed_stages"].clear()
        original = timer.finish()
        checked = validate_pipeline_timing(original)
        checked["executed_stages"].clear()
        self.assertEqual(timer.finish()["executed_stages"], ["controller_ms"])

    def test_invalid_or_backward_clock_is_rejected(self):
        for value in (True, float("nan"), float("inf"), "1"):
            with self.subTest(value=value), self.assertRaisesRegex(PipelineLatencyError, "clock-invalid"):
                PipelineFrameTimer(clock=lambda: value)
        clock = Clock()
        timer = PipelineFrameTimer(clock=clock)
        clock.value = -1
        with self.assertRaisesRegex(PipelineLatencyError, "backwards"):
            timer.measure("controller_ms", lambda: None)


class PipelineTimingRecordTests(unittest.TestCase):
    def test_boolean_schema_version_and_unexpected_fields_are_rejected(self):
        for change in ({"schema_version": True}, {"claimed_realtime": True}):
            bad = timing(10)
            bad.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(PipelineLatencyError, "schema-invalid"):
                validate_pipeline_timing(bad)

    def test_missing_nan_negative_or_boolean_duration_is_rejected(self):
        base = timing(10)
        for value in (float("nan"), float("inf"), -1, True):
            bad = copy.deepcopy(base)
            bad["stages_ms"]["controller_ms"] = value
            with self.subTest(value=value), self.assertRaises(PipelineLatencyError):
                validate_pipeline_timing(bad)
        for field in ("controller_ms", "forward_decode_ms"):
            bad = copy.deepcopy(base)
            del bad["stages_ms"][field]
            with self.assertRaisesRegex(PipelineLatencyError, "stage-fields"):
                validate_pipeline_timing(bad)

    def test_false_component_sum_and_unexecuted_time_are_rejected(self):
        bad = timing(10)
        bad["total_processing_ms"] += 3
        with self.assertRaisesRegex(PipelineLatencyError, "sum-mismatch"):
            validate_pipeline_timing(bad)
        bad = timing(10)
        bad["stages_ms"]["near_correction_ms"] = 1
        with self.assertRaisesRegex(PipelineLatencyError, "unexecuted"):
            validate_pipeline_timing(bad)

    def test_cuda_sync_claim_cannot_omit_forward_or_name_missing_stage(self):
        bad = timing(10)
        bad.update(device="cuda", synchronization="explicit-pre-and-post-GPU-stage")
        with self.assertRaisesRegex(PipelineLatencyError, "synchronization-or-failure"):
            validate_pipeline_timing(bad)
        bad["synchronized_stages"] = ["forward_decode_ms", "near_correction_ms"]
        with self.assertRaisesRegex(PipelineLatencyError, "synchronization-or-failure"):
            validate_pipeline_timing(bad)


class PipelineAggregationTests(unittest.TestCase):
    def test_warmup_is_per_episode_and_failed_episodes_remain_in_statistics(self):
        episodes = [episode("clean", [1000, 10, 20]), episode("failed", [2000, 100, 100], success=False)]
        report = aggregate_pipeline_latency(episodes, warmup_frames=1, budget_ms=50)
        group = next(group for group in report["groups"] if group["case"] == "right_curve")
        steady = group["steady_state"]
        self.assertEqual(report["frame_count"], 6)
        self.assertEqual(report["failed_episode_count"], 1)
        self.assertEqual(len(report["frame_records"]), 6)
        self.assertEqual(group["warmup_frame_count"], 2)
        self.assertEqual(group["all_frames"]["failed_episode_frame_count"], 3)
        self.assertEqual(steady["failed_episode_frame_count"], 2)
        self.assertAlmostEqual(steady["total_processing_ms"]["median_ms"], 60)
        self.assertAlmostEqual(steady["total_processing_ms"]["p95_ms"], 100)
        self.assertEqual(steady["total_processing_ms"]["over_budget_count"], 2)
        self.assertFalse(report["processing_budget_passed"])
        self.assertFalse(report["realtime_driving_demonstrated"])

    def test_groups_are_pooled_by_frame_and_small_case_cannot_be_hidden(self):
        episodes = [episode("straight", [10] * 30, case="straight"), episode("curve", [100], case="left_curve")]
        report = aggregate_pipeline_latency(episodes, warmup_frames=0, budget_ms=50)
        overall = next(group for group in report["groups"] if group["case"] == "all")
        curve = next(group for group in report["groups"] if group["case"] == "left_curve")
        self.assertTrue(overall["steady_state"]["total_processing_ms"]["p95_budget_passed"])
        self.assertFalse(curve["steady_state"]["total_processing_ms"]["p95_budget_passed"])
        self.assertFalse(report["processing_budget_passed"])
        self.assertAlmostEqual(overall["all_frames"]["total_processing_ms"]["mean_ms"], 400 / 31)

    def test_p95_interpolation_and_budget_equality_are_explicit(self):
        report = aggregate_pipeline_latency([episode("ep", [10, 20, 30, 50])], warmup_frames=0, budget_ms=50)
        statistics = report["groups"][0]["steady_state"]["total_processing_ms"]
        self.assertAlmostEqual(statistics["p95_ms"], 47)
        self.assertAlmostEqual(statistics["median_ms"], 25)
        self.assertEqual(statistics["over_budget_count"], 0)
        self.assertTrue(report["processing_budget_passed"])

    def test_all_warmup_has_no_steady_estimate_or_false_zero_budget_pass(self):
        report = aggregate_pipeline_latency([episode("short", [10, 20])], warmup_frames=10)
        for group in report["groups"]:
            self.assertEqual(group["steady_state"]["frame_count"], 0)
            self.assertIsNone(group["steady_state"]["total_processing_ms"]["p95_ms"])
            self.assertIsNone(group["steady_state"]["total_processing_ms"]["over_budget_fraction"])
            self.assertEqual(group["all_frames"]["frame_count"], 2)
        self.assertFalse(report["processing_budget_passed"])

    def test_inference_failure_duration_and_outcome_survive_json_roundtrip(self):
        source = episode("failure", [10, 50], success=False)
        source["rows"][-1]["latency"] = timing(50, failed=True)
        report = aggregate_pipeline_latency([source], warmup_frames=0)
        workspace = Path(__file__).resolve().parents[1] / "results" / "local"
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            target = Path(directory) / "fixture.json"
            target.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
            loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded["frame_records"][-1]["latency"]["failed_stages"], ["forward_decode_ms"])
        self.assertEqual(loaded["groups"][0]["all_frames"]["failed_stage_frame_count"], 1)
        self.assertEqual(loaded["groups"][0]["all_frames"]["failed_episode_frame_count"], 2)

    def test_missing_failed_frame_or_incomplete_trace_is_rejected(self):
        source = episode("failure", [10, 20], success=False)
        source["rows"].pop()
        with self.assertRaisesRegex(PipelineLatencyError, "coverage-mismatch"):
            aggregate_pipeline_latency([source])
        source = episode("failure", [10, 20], success=False)
        del source["rows"][-1]["latency"]
        with self.assertRaisesRegex(PipelineLatencyError, "schema-invalid"):
            aggregate_pipeline_latency([source])

    def test_duplicates_unordered_frames_and_ambiguous_outcomes_are_rejected(self):
        source = episode("ep", [10, 20])
        with self.assertRaisesRegex(PipelineLatencyError, "duplicate"):
            aggregate_pipeline_latency([source, source])
        for frame in (True, 10, -1):
            changed = copy.deepcopy(source)
            changed["rows"][-1]["frame"] = frame
            with self.subTest(frame=frame), self.assertRaisesRegex(PipelineLatencyError, "frame-identity"):
                aggregate_pipeline_latency([changed])
        source["driving_success"] = "False"
        with self.assertRaisesRegex(PipelineLatencyError, "outcome-invalid"):
            aggregate_pipeline_latency([source])

    def test_invalid_policy_never_silently_excludes_samples(self):
        for warmup in (True, -1, 1.5):
            with self.subTest(warmup=warmup), self.assertRaises(PipelineLatencyError):
                aggregate_pipeline_latency([episode("ep", [10])], warmup_frames=warmup)
        for budget in (True, 0, -1, float("nan")):
            with self.subTest(budget=budget), self.assertRaises(PipelineLatencyError):
                aggregate_pipeline_latency([episode("ep", [10])], budget_ms=budget)

    def test_inputs_and_output_records_are_independent(self):
        source = episode("ep", [10, 20])
        before = copy.deepcopy(source)
        report = aggregate_pipeline_latency([source], warmup_frames=0)
        report["frame_records"][0]["latency"]["stages_ms"]["controller_ms"] = 123
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
