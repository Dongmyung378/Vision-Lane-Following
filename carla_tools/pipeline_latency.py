"""Measure processing stages without equating synchronous simulation to realtime.

The caller places the timer after sensor delivery and ends it before logging or
simulator advancement. CUDA stages require an explicit synchronization callback;
this module never imports Torch, selects a GPU or runs simulator operations.
"""
from __future__ import annotations

import copy
import math
import statistics
import time

SCHEMA_VERSION = 1
STAGES = ("bgra_conversion_ms", "learned_preprocessing_ms", "forward_decode_ms",
          "near_correction_ms", "controller_ms")
TIMING_SCOPE = ("sensor-delivered BGRA conversion, learned preprocessing/device transfer, "
                "forward/decode/device transfer, optional RGB correction and controller; "
                "excludes sensor/simulator wait, simulator advancement, disk writes and video encoding")


class PipelineLatencyError(ValueError):
    """A timing record is incomplete, ambiguous or internally inconsistent."""


def _number(value, message, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise PipelineLatencyError(message)
    return float(value)


class PipelineFrameTimer:
    """Time serial operations, retaining timings when an operation raises.

    Initial CUDA work is drained before the total interval starts. Each GPU
    operation is additionally synchronized immediately before and after its own
    interval. Synchronization after an operation is inside that stage's timing.
    Other Python validation and orchestration inside the boundary are reported
    as overhead instead of disappearing from the total processing duration.
    """

    def __init__(self, *, device="cpu", synchronize=None, clock=time.perf_counter):
        if device not in ("cpu", "cuda"):
            raise PipelineLatencyError("pipeline-device-invalid")
        if not callable(clock) or (device == "cuda" and not callable(synchronize)):
            raise PipelineLatencyError("pipeline-clock-or-cuda-synchronizer-required")
        if synchronize is not None and not callable(synchronize):
            raise PipelineLatencyError("pipeline-synchronizer-invalid")
        self.device, self._synchronize, self._clock = device, synchronize, clock
        self._values = {stage: 0. for stage in STAGES}
        self._executed, self._synchronized, self._failed = [], [], []
        self._active, self._finished = False, None
        if device == "cuda":
            self._synchronize()
        self._started = self._read_clock()
        self._last = self._started

    def _read_clock(self):
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise PipelineLatencyError("pipeline-clock-invalid")
        if hasattr(self, "_last") and value < self._last:
            raise PipelineLatencyError("pipeline-clock-moved-backwards")
        self._last = float(value)
        return self._last

    def measure(self, stage, operation, *, gpu=False):
        """Return the operation's result; propagate its error after timing it."""
        if self._finished is not None or self._active:
            raise PipelineLatencyError("pipeline-timer-finished-or-nested")
        if stage not in STAGES or stage in self._executed or not callable(operation) or type(gpu) is not bool:
            raise PipelineLatencyError("pipeline-stage-invalid-or-repeated")
        if gpu and self.device != "cuda":
            raise PipelineLatencyError("pipeline-gpu-stage-requires-cuda")
        if self.device == "cuda" and stage == "forward_decode_ms" and not gpu:
            raise PipelineLatencyError("pipeline-cuda-forward-requires-synchronization")
        if gpu:
            self._synchronize()
        started = self._read_clock()
        self._active = True
        self._executed.append(stage)
        if gpu:
            self._synchronized.append(stage)
        try:
            return operation()
        except BaseException:
            self._failed.append(stage)
            raise
        finally:
            try:
                if gpu:
                    self._synchronize()
            except BaseException:
                if stage not in self._failed:
                    self._failed.append(stage)
                raise
            finally:
                self._values[stage] = (self._read_clock() - started) * 1000
                self._active = False

    def finish(self):
        """Return an independent record; repeated reads cannot alter the timer."""
        if self._active:
            raise PipelineLatencyError("pipeline-timer-has-active-stage")
        if self._finished is None:
            elapsed = (self._read_clock() - self._started) * 1000
            component_sum = math.fsum(self._values.values())
            if component_sum > elapsed + 1e-6:
                raise PipelineLatencyError("pipeline-stage-time-exceeds-total")
            self._finished = {
                "schema_version": SCHEMA_VERSION, "device": self.device,
                "synchronization": "explicit-pre-and-post-GPU-stage" if self.device == "cuda" else "none-cpu",
                "executed_stages": list(self._executed), "synchronized_stages": list(self._synchronized),
                "failed_stages": list(self._failed), "stages_ms": dict(self._values),
                "pipeline_overhead_ms": max(0., elapsed - component_sum),
                "total_processing_ms": elapsed, "timing_scope": TIMING_SCOPE,
            }
        return copy.deepcopy(self._finished)


def validate_pipeline_timing(record):
    """Reject missing stages, nonsynchronized CUDA inference and false totals."""
    keys = {"schema_version", "device", "synchronization", "executed_stages", "synchronized_stages",
            "failed_stages", "stages_ms", "pipeline_overhead_ms", "total_processing_ms", "timing_scope"}
    if (type(record) is not dict or set(record) != keys or type(record.get("schema_version")) is not int
            or record.get("schema_version") != SCHEMA_VERSION):
        raise PipelineLatencyError("pipeline-timing-schema-invalid")
    if record["device"] not in ("cpu", "cuda") or record["timing_scope"] != TIMING_SCOPE:
        raise PipelineLatencyError("pipeline-timing-scope-or-device-invalid")
    expected_sync = "explicit-pre-and-post-GPU-stage" if record["device"] == "cuda" else "none-cpu"
    if record["synchronization"] != expected_sync:
        raise PipelineLatencyError("pipeline-synchronization-description-invalid")
    for key in ("executed_stages", "synchronized_stages", "failed_stages"):
        values = record[key]
        if (type(values) is not list or any(type(value) is not str or value not in STAGES for value in values)
                or len(set(values)) != len(values)):
            raise PipelineLatencyError("pipeline-stage-coverage-invalid")
    executed = set(record["executed_stages"])
    synchronized = set(record["synchronized_stages"])
    if (not synchronized <= executed or not set(record["failed_stages"]) <= executed
            or (record["device"] == "cpu" and synchronized)
            or (record["device"] == "cuda" and "forward_decode_ms" in executed and "forward_decode_ms" not in synchronized)):
        raise PipelineLatencyError("pipeline-stage-synchronization-or-failure-invalid")
    stages = record["stages_ms"]
    if type(stages) is not dict or set(stages) != set(STAGES):
        raise PipelineLatencyError("pipeline-stage-fields-invalid")
    values = {key: _number(value, "pipeline-stage-duration-invalid") for key, value in stages.items()}
    if any(values[key] != 0 for key in set(STAGES) - executed):
        raise PipelineLatencyError("pipeline-unexecuted-stage-has-duration")
    total = _number(record["total_processing_ms"], "pipeline-total-duration-invalid")
    overhead = _number(record["pipeline_overhead_ms"], "pipeline-overhead-duration-invalid")
    if not math.isclose(total, math.fsum(values.values()) + overhead, rel_tol=1e-9, abs_tol=1e-6):
        raise PipelineLatencyError("pipeline-duration-sum-mismatch")
    return copy.deepcopy(record)


def _percentile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower, upper = math.floor(location), math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def _distribution(values, budget_ms=None):
    result = {"count": len(values), "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    if values:
        result.update(mean_ms=statistics.fmean(values), median_ms=statistics.median(values),
                      p95_ms=_percentile(values, .95), max_ms=max(values))
    if budget_ms is not None:
        count = sum(value > budget_ms for value in values)
        result.update(budget_ms=budget_ms, over_budget_count=count,
                      over_budget_fraction=count / len(values) if values else None,
                      p95_budget_passed=result["p95_ms"] is not None and result["p95_ms"] <= budget_ms)
    return result


def _summarize(records, budget_ms):
    return {
        "frame_count": len(records),
        "failed_episode_frame_count": sum(not record["driving_success"] for record in records),
        "failed_stage_frame_count": sum(bool(record["latency"]["failed_stages"]) for record in records),
        "stages_ms": {stage: _distribution([record["latency"]["stages_ms"][stage] for record in records]) for stage in STAGES},
        "pipeline_overhead_ms": _distribution([record["latency"]["pipeline_overhead_ms"] for record in records]),
        "total_processing_ms": _distribution([record["latency"]["total_processing_ms"] for record in records], budget_ms),
    }


def aggregate_pipeline_latency(episodes, *, warmup_frames=10, budget_ms=50.):
    """Pool every recorded frame by method and case, including failed episodes.

    Each episode supplies episode_id, method, case, driving_success and rows.
    A row supplies an increasing integer frame and a complete latency record.
    Optional expected_frame_count binds the number of rows to its source trace.
    Warmup marks the first N frames of *each* episode; those frames stay in the
    all-frame summaries and frame_records. Budget decisions use steady-state
    p95 values and report over-budget counts separately, never realtime claims.
    """
    if type(warmup_frames) is not int or warmup_frames < 0:
        raise PipelineLatencyError("pipeline-warmup-policy-invalid")
    budget_ms = _number(budget_ms, "pipeline-budget-invalid", positive=True)
    if type(episodes) not in (list, tuple) or not episodes:
        raise PipelineLatencyError("pipeline-episodes-empty-or-invalid")
    records, identities = [], set()
    for episode in episodes:
        if type(episode) is not dict:
            raise PipelineLatencyError("pipeline-episode-invalid")
        identity, method, case = (episode.get(key) for key in ("episode_id", "method", "case"))
        if any(type(value) is not str or not value.strip() for value in (identity, method, case)) or identity in identities:
            raise PipelineLatencyError("pipeline-episode-identity-invalid-or-duplicate")
        if case == "all" or type(episode.get("driving_success")) is not bool:
            raise PipelineLatencyError("pipeline-episode-case-or-outcome-invalid")
        identities.add(identity)
        rows = episode.get("rows")
        if type(rows) not in (list, tuple) or not rows:
            raise PipelineLatencyError("pipeline-episode-rows-empty-or-invalid")
        if "expected_frame_count" in episode and (type(episode["expected_frame_count"]) is not int
                or episode["expected_frame_count"] != len(rows)):
            raise PipelineLatencyError("pipeline-frame-coverage-mismatch")
        previous_frame = -1
        for index, row in enumerate(rows):
            frame = row.get("frame") if type(row) is dict else None
            if type(frame) is not int or frame <= previous_frame:
                raise PipelineLatencyError("pipeline-frame-identity-invalid-or-unordered")
            previous_frame = frame
            timing = validate_pipeline_timing(row.get("latency"))
            records.append({"episode_id": identity, "method": method, "case": case,
                            "driving_success": episode["driving_success"], "frame": frame,
                            "warmup": index < warmup_frames, "latency": timing})
    groups = []
    for method in sorted({record["method"] for record in records}):
        method_records = [record for record in records if record["method"] == method]
        for case in ("all", *sorted({record["case"] for record in method_records})):
            selected = [record for record in method_records if case == "all" or record["case"] == case]
            steady = [record for record in selected if not record["warmup"]]
            groups.append({"method": method, "case": case,
                           "episode_count": len({record["episode_id"] for record in selected}),
                           "failed_episode_count": len({record["episode_id"] for record in selected if not record["driving_success"]}),
                           "warmup_frame_count": len(selected) - len(steady),
                           "all_frames": _summarize(selected, budget_ms),
                           "steady_state": _summarize(steady, budget_ms)})
    return {"schema_version": SCHEMA_VERSION, "status": "PASS", "timing_scope": TIMING_SCOPE,
            "warmup_policy": {"frames_per_episode": warmup_frames, "excluded_from": "steady-state-only",
                              "all_driving_and_failed_frames_retained": True},
            "budget_policy": {"processing_budget_ms": budget_ms, "decision": "each-method-and-case-steady-state-p95",
                              "over_budget_counts_reported_separately": True},
            "episode_count": len(episodes), "frame_count": len(records),
            "failed_episode_count": sum(not episode["driving_success"] for episode in episodes),
            "processing_budget_passed": all(group["steady_state"]["total_processing_ms"]["p95_budget_passed"] for group in groups),
            "realtime_driving_demonstrated": False, "groups": groups, "frame_records": records}
