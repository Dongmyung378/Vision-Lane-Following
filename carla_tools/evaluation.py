"""Source-preserving, simulator-independent closed-loop episode evaluation.

Perception never consumes this module's privileged route observations. All
driving failures remain metric rows; malformed evidence is a technical failure.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_sha256, read_strict_json

DEFAULT_EVALUATION_PATH = Path(__file__).resolve().parents[1] / "configs/evaluation.json"
METRIC_VERSION = "vlf-closed-loop-v1"
TERMINATION_PRECEDENCE = ("technical_failure", "collision", "lane_departure", "intervention",
                          "route_completed", "stalled", "timeout", "route_horizon_exhausted")


class EvaluationError(ValueError):
    """A stable evaluation configuration, integrity, or publication error."""


def _finite(value, code, *, minimum=None):
    if type(value) not in (int, float):
        raise EvaluationError(code)
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise EvaluationError(code) from None
    if not math.isfinite(result) or minimum is not None and result < minimum:
        raise EvaluationError(code)
    return result


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int = 1
    metric_version: str = METRIC_VERSION
    route_completion_tolerance_m: float = 0.5
    lane_departure_hold_seconds: float = 0.5
    stalled_speed_mps: float = 0.1
    stalled_progress_m: float = 0.1
    stalled_hold_seconds: float = 5.0
    timeout_seconds: float = 120.0
    minimum_future_horizon_m: float = 15.0
    recovery_lateral_tolerance_m: float = 0.2
    recovery_heading_tolerance_deg: float = 3.0
    recovery_hold_seconds: float = 1.0
    recovery_deadline_seconds: float = 5.0
    fixed_delta_seconds: float = 0.05
    timestamp_tolerance_seconds: float = 0.001

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.metric_version != METRIC_VERSION:
            raise EvaluationError("invalid-evaluation-version")
        for item in fields(self):
            if item.name in ("schema_version", "metric_version"):
                continue
            number = _finite(getattr(self, item.name), "invalid-evaluation-" + item.name, minimum=0)
            if number == 0:
                raise EvaluationError("invalid-evaluation-" + item.name)
            object.__setattr__(self, item.name, number)
        if self.recovery_hold_seconds > self.recovery_deadline_seconds:
            raise EvaluationError("invalid-recovery-deadline")
        if self.timestamp_tolerance_seconds >= self.fixed_delta_seconds:
            raise EvaluationError("invalid-timestamp-tolerance")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def evaluation_config_from_dict(value):
    if type(value) is not dict or set(value) != {item.name for item in fields(EvaluationConfig)}:
        raise EvaluationError("invalid-evaluation-config-fields")
    return EvaluationConfig(**value)


_evaluation_config_from_dict = evaluation_config_from_dict


def load_evaluation_config(path=DEFAULT_EVALUATION_PATH):
    try:
        return evaluation_config_from_dict(read_strict_json(DEFAULT_EVALUATION_PATH if path is None else path))
    except ContractError as error:
        raise EvaluationError("evaluation-config-" + error.code) from error


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values):
    return {"count": len(values), "mean": math.fsum(values) / len(values) if values else None,
            "median": _quantile(values, .5), "p95": _quantile(values, .95),
            "max": max(values) if values else None}


def _condition_name(condition):
    if type(condition) is dict:
        return json.dumps(condition, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if type(condition) is str and condition:
        return condition
    raise EvaluationError("invalid-evaluation-condition")


def _observation(row):
    if type(row) is not dict or type(row.get("frame")) is not int or row["frame"] < 0:
        raise EvaluationError("invalid-evaluation-frame")
    for flag in ("technical_failure", "intervention"):
        if flag in row and type(row[flag]) is not bool:
            raise EvaluationError("invalid-evaluation-flag")
    tracking = row.get("tracking", row)
    vehicle = row.get("vehicle", {})
    pose = row.get("actor_transform", vehicle.get("transform", {}))
    location = pose.get("location", {})
    control = row.get("applied_control", vehicle.get("applied_control", {}))
    reference = row.get("reference_m", tracking.get("reference_arc_m"))
    timestamp = row.get("time_seconds", row.get("timestamp"))
    result = {"frame": row["frame"], "timestamp": _finite(timestamp, "invalid-evaluation-time"),
              "reference_m": _finite(reference, "invalid-evaluation-reference"),
              "lateral_error_m": _finite(tracking.get("lateral_error_m"), "invalid-evaluation-lateral"),
              "heading_error_deg": _finite(tracking.get("heading_error_deg"), "invalid-evaluation-heading"),
              "speed_mps": _finite(row.get("speed_mps"), "invalid-evaluation-speed", minimum=0),
              "body_lane_margin_m": _finite(row.get("body_lane_margin_m"), "invalid-evaluation-footprint"),
              "x": _finite(location.get("x"), "invalid-evaluation-position"),
              "y": _finite(location.get("y"), "invalid-evaluation-position"),
              "steer": _finite(control.get("steer"), "invalid-evaluation-steer")}
    if abs(result["steer"]) > 1 or abs(result["heading_error_deg"]) > 180:
        raise EvaluationError("invalid-evaluation-control-or-heading")
    latency = row.get("latency_ms", {})
    if type(latency) is not dict:
        raise EvaluationError("invalid-evaluation-latency")
    result["latency_ms"] = {str(key): _finite(value, "invalid-evaluation-latency", minimum=0)
                            for key, value in latency.items()}
    return result


def deduplicate_events(events, *, start_frame=None, end_frame=None):
    """Exact frame/actor or frame/full marking-set duplicates; raw rows untouched."""
    if type(events) not in (list, tuple):
        raise EvaluationError("invalid-evaluation-events")
    selected, seen = [], set()
    for event in events:
        if type(event) is not dict or type(event.get("frame")) is not int or event["frame"] < 0:
            raise EvaluationError("invalid-evaluation-event-frame")
        try:
            canonical_sha256(event)
        except ContractError as error:
            raise EvaluationError("invalid-evaluation-event-payload") from error
        if start_frame is not None and event["frame"] < start_frame or end_frame is not None and event["frame"] > end_frame:
            continue
        kind = event.get("kind")
        if kind == "collision":
            actor = event.get("other_actor_id")
            if type(actor) is not int or actor < 0:
                raise EvaluationError("invalid-evaluation-collision-actor")
            impulse = event.get("impulse")
            if impulse is not None:
                if type(impulse) is not dict or set(impulse) != {"x", "y", "z"}:
                    raise EvaluationError("invalid-evaluation-collision-impulse")
                for value in impulse.values():
                    _finite(value, "invalid-evaluation-collision-impulse")
            key = ("collision", event["frame"], actor)
        elif kind in ("lane", "lane_invasion"):
            markings = event.get("markings")
            if type(markings) is not list or not markings:
                raise EvaluationError("invalid-evaluation-lane-markings")
            canonical = set()
            for marking in markings:
                if type(marking) is not dict or set(marking) != {"type", "color", "lane_change", "width_m"}:
                    raise EvaluationError("invalid-evaluation-lane-marking")
                if any(type(marking[key]) is not str or not marking[key] for key in ("type", "color", "lane_change")):
                    raise EvaluationError("invalid-evaluation-lane-marking")
                width = _finite(marking["width_m"], "invalid-evaluation-lane-marking", minimum=0)
                canonical.add(canonical_sha256({**marking, "width_m": width}))
            key = ("lane", event["frame"], tuple(sorted(canonical)))
        else:
            raise EvaluationError("invalid-evaluation-event-kind")
        if key not in seen:
            selected.append(copy.deepcopy(event))
            seen.add(key)
    return selected


class EvaluationTracker:
    """Incremental evaluator of an original-route observation stream.

    ``route_length_m`` is the valid evaluation length, excluding the additional
    label corridor. The first retained frame starts all time windows and distance.
    Invalid observations terminate with technical_failure without advancing state.
    """

    def __init__(self, config, route_length_m, start_reference_m=0.0, condition="normal"):
        if not isinstance(config, EvaluationConfig):
            raise EvaluationError("invalid-evaluation-config")
        self.config = config
        self.route_length_m = _finite(route_length_m, "invalid-evaluation-route-length", minimum=0)
        if self.route_length_m <= config.route_completion_tolerance_m:
            raise EvaluationError("invalid-evaluation-route-length")
        self.start_reference_m = _finite(start_reference_m, "invalid-evaluation-start-reference", minimum=0)
        self.condition = _condition_name(condition)
        self._condition_value = copy.deepcopy(condition)
        self.rows, self.raw_events, self.integrity_errors = [], [], []
        self.progress_m = self.actual_distance_m = 0.0
        self.intervention_free_distance_m = None
        self._departure_start = self._stall_start = self._recovery_start = None
        self._stall_progress = 0.0
        self.recovery_time_seconds = None
        self.flags, self.reason = [], None

    def _failure(self, code):
        if code not in self.integrity_errors:
            self.integrity_errors.append(code)
        self.flags = [reason for reason in TERMINATION_PRECEDENCE if reason in set(self.flags) | {"technical_failure"}]
        self.reason = "technical_failure"
        return self._decision()

    def _decision(self):
        return {"terminated": self.reason is not None, "reason": self.reason, "flags": list(self.flags),
                "progress_m": self.progress_m, "completion_pct": 100 * self.progress_m / self.route_length_m,
                "actual_distance_m": self.actual_distance_m,
                "recovery_time_seconds": self.recovery_time_seconds,
                "integrity_errors": list(self.integrity_errors)}

    def update(self, observation, *, events=(), technical_failure=False, intervention=False,
               remaining_horizon_m=None):
        if type(technical_failure) is not bool or type(intervention) is not bool:
            return self._failure("invalid-evaluation-flag")
        if self.reason is not None:
            return self._failure("observation-after-termination")
        try:
            row = _observation(observation)
            if self.rows:
                previous = self.rows[-1]
                delta = row["timestamp"] - previous["timestamp"]
                if row["frame"] != previous["frame"] + 1 or delta <= 0:
                    raise EvaluationError("noncontiguous-evaluation-observation")
                if abs(delta - self.config.fixed_delta_seconds) > self.config.timestamp_tolerance_seconds:
                    raise EvaluationError("evaluation-timestep-mismatch")
            horizon = None if remaining_horizon_m is None else _finite(remaining_horizon_m, "invalid-evaluation-horizon", minimum=0)
            # Validate every supplied event before committing observation state.
            if type(events) not in (list, tuple):
                raise EvaluationError("invalid-evaluation-events")
            candidates = list(events)
            deduplicate_events(candidates)
        except (EvaluationError, TypeError, AttributeError, ContractError) as error:
            return self._failure(str(error) if isinstance(error, EvaluationError) else "invalid-evaluation-observation")
        if self.rows:
            self.actual_distance_m += math.hypot(row["x"] - self.rows[-1]["x"], row["y"] - self.rows[-1]["y"])
        self.rows.append(row)
        self.raw_events.extend(copy.deepcopy(candidates))
        self.progress_m = min(self.route_length_m, max(self.progress_m, row["reference_m"] - self.start_reference_m, 0.0))
        elapsed = row["timestamp"] - self.rows[0]["timestamp"]
        flags = set()
        if technical_failure:
            flags.add("technical_failure")
        effective_events = deduplicate_events(self.raw_events, start_frame=self.rows[0]["frame"], end_frame=row["frame"])
        if any(event["kind"] == "collision" for event in effective_events):
            flags.add("collision")
        if row["body_lane_margin_m"] < 0:
            if self._departure_start is None:
                self._departure_start = row["timestamp"]
            if row["timestamp"] - self._departure_start >= self.config.lane_departure_hold_seconds - 1e-9:
                flags.add("lane_departure")
        else:
            self._departure_start = None
        if intervention or observation.get("intervention") is True:
            self.intervention_free_distance_m = self.actual_distance_m
            flags.add("intervention")
        if self.progress_m >= self.route_length_m - self.config.route_completion_tolerance_m:
            flags.add("route_completed")
        if row["speed_mps"] < self.config.stalled_speed_mps:
            if self._stall_start is None or self.progress_m - self._stall_progress >= self.config.stalled_progress_m:
                self._stall_start, self._stall_progress = row["timestamp"], self.progress_m
            if row["timestamp"] - self._stall_start >= self.config.stalled_hold_seconds - 1e-9:
                flags.add("stalled")
        else:
            self._stall_start = None
        if elapsed >= self.config.timeout_seconds - 1e-9:
            flags.add("timeout")
        if horizon is not None and horizon < self.config.minimum_future_horizon_m - 1e-9:
            flags.add("route_horizon_exhausted")
        in_recovery = (abs(row["lateral_error_m"]) <= self.config.recovery_lateral_tolerance_m and
                       abs(row["heading_error_deg"]) <= self.config.recovery_heading_tolerance_deg)
        if in_recovery:
            if self._recovery_start is None:
                self._recovery_start = row["timestamp"]
            if (self.recovery_time_seconds is None and
                    row["timestamp"] - self._recovery_start >= self.config.recovery_hold_seconds - 1e-9 and
                    elapsed <= self.config.recovery_deadline_seconds + 1e-9):
                self.recovery_time_seconds = elapsed
        else:
            self._recovery_start = None
        self.flags = [reason for reason in TERMINATION_PRECEDENCE if reason in flags]
        self.reason = self.flags[0] if self.flags else None
        return self._decision()

    def finalize(self, metadata=None, *, termination_reason=None, events=()):
        """Return a metric row, preserving all valid frames and original events."""
        metadata = metadata or {}
        if type(metadata) is not dict:
            raise EvaluationError("invalid-evaluation-metadata")
        # Keep valid event records even when another callback payload is corrupt.
        # The corrupt source remains untouched on disk and its index is reported.
        supplied = events if events else self.raw_events
        raw_events, invalid_event_indices = [], []
        if type(events) not in (list, tuple):
            self._failure("invalid-evaluation-events")
            supplied = []
        if type(supplied) not in (list, tuple):
            self._failure("invalid-evaluation-events")
            supplied = []
        for index, event in enumerate(supplied):
            try:
                deduplicate_events([event])
                raw_events.append(copy.deepcopy(event))
            except (EvaluationError, ContractError) as error:
                self._failure(str(error))
                invalid_event_indices.append(index)
        effective_events = deduplicate_events(raw_events, start_frame=self.rows[0]["frame"] if self.rows else None,
                                               end_frame=self.rows[-1]["frame"] if self.rows else None)
        flags = set(self.flags)
        if any(event["kind"] == "collision" for event in effective_events):
            flags.add("collision")
        if termination_reason is not None:
            if termination_reason not in TERMINATION_PRECEDENCE:
                self._failure("unknown-evaluation-termination")
                flags.add("technical_failure")
            elif termination_reason in ("technical_failure", "intervention"):
                flags.add(termination_reason)
            elif termination_reason not in flags:
                self._failure("unsubstantiated-evaluation-termination")
                flags.add("technical_failure")
        if not self.rows:
            self._failure("empty-evaluation-episode")
            flags.add("technical_failure")
        if not flags:
            self._failure("unterminated-evaluation-episode")
            flags.add("technical_failure")
        flags |= set(self.flags)
        self.flags = [reason for reason in TERMINATION_PRECEDENCE if reason in flags]
        self.reason = self.flags[0]
        lateral = [abs(row["lateral_error_m"]) for row in self.rows]
        heading = [abs(row["heading_error_deg"]) for row in self.rows]
        rates, accelerations = [], []
        previous_delta = previous_rate = None
        for first, second in zip(self.rows, self.rows[1:]):
            delta = second["timestamp"] - first["timestamp"]
            rate = (second["steer"] - first["steer"]) / delta
            rates.append(abs(rate))
            if previous_rate is not None:
                accelerations.append(abs(2 * (rate - previous_rate) / (delta + previous_delta)))
            previous_delta, previous_rate = delta, rate
        latency_keys = {key for row in self.rows for key in row["latency_ms"]}
        latency = {key: _summary([row["latency_ms"][key] for row in self.rows if key in row["latency_ms"]])
                   for key in sorted(latency_keys)}
        collisions = sum(event["kind"] == "collision" for event in effective_events)
        lanes = len(effective_events) - collisions
        distance_km = self.actual_distance_m / 1000
        duration = self.rows[-1]["timestamp"] - self.rows[0]["timestamp"] if self.rows else 0.0
        route_metadata = metadata.get("route_contract") or metadata.get("route") or {}
        route_id = metadata.get("route_id", route_metadata.get("route_id", "unknown") if type(route_metadata) is dict else "unknown")
        condition_name = self._condition_value.get("name", "unknown") if type(self._condition_value) is dict else self._condition_value
        recovery_required = condition_name not in ("normal", "center", "unknown")
        result = {"metric_version": self.config.metric_version, "evaluation_config_sha256": self.config.sha256,
                  "episode_id": str(metadata.get("episode_id", "unknown")),
                  "route_id": str(route_id),
                  "method": str(metadata.get("method", metadata.get("prediction_method", "unknown"))),
                  "condition": self.condition, "termination_reason": self.reason, "termination_flags": list(self.flags),
                  "driving_success": self.reason == "route_completed", "integrity_valid": not self.integrity_errors,
                  "integrity_errors": list(self.integrity_errors), "sample_count": len(self.rows),
                  "duration_seconds": duration, "evaluation_length_m": self.route_length_m,
                  "progress_m": self.progress_m, "completion_pct": 100 * self.progress_m / self.route_length_m,
                  "actual_distance_m": self.actual_distance_m, "collision_count": collisions,
                  "lane_invasion_count": lanes, "collision_per_km": collisions / distance_km if distance_km else None,
                  "lane_invasion_per_km": lanes / distance_km if distance_km else None,
                  "intervention_free_distance_m": (self.actual_distance_m if self.intervention_free_distance_m is None
                                                   else self.intervention_free_distance_m),
                  "recovery_required": recovery_required, "recovery_success": self.recovery_time_seconds is not None,
                  "recovery_time_seconds": self.recovery_time_seconds,
                  "lateral_abs_m": _summary(lateral), "heading_abs_deg": _summary(heading),
                  "terminal_lateral_abs_m": lateral[-1] if lateral else None,
                  "terminal_heading_abs_deg": heading[-1] if heading else None,
                  "minimum_body_lane_margin_m": min((row["body_lane_margin_m"] for row in self.rows), default=None),
                  "steering_rate_abs_per_second": _summary(rates),
                  "steering_acceleration_abs_per_second2": _summary(accelerations),
                  "latency_ms": latency, "raw_events": raw_events, "invalid_event_indices": invalid_event_indices,
                  "deduplicated_events": effective_events,
                  "pooled_samples": {"lateral_abs_m": lateral, "heading_abs_deg": heading,
                                     "steering_rate_abs_per_second": rates,
                                     "steering_acceleration_abs_per_second2": accelerations,
                                     "latency_ms": {key: [row["latency_ms"][key] for row in self.rows
                                                          if key in row["latency_ms"]] for key in sorted(latency_keys)}}}
        # Last safeguard: evidence publication cannot produce nonfinite JSON.
        canonical_sha256(result)
        return result


def evaluate_episode(metadata, rows, events, *, config=None):
    """Replay raw records independently; invalid/truncated episodes still count."""
    invalid_metadata = type(metadata) is not dict
    if invalid_metadata:
        metadata = {}
    config_error = None
    embedded = metadata.get("evaluation_config")
    if config is None:
        try:
            config = evaluation_config_from_dict(embedded) if embedded is not None else load_evaluation_config()
        except EvaluationError:
            config = load_evaluation_config()
            config_error = "invalid-embedded-evaluation-config"
    if embedded is not None:
        try:
            if evaluation_config_from_dict(embedded).sha256 != config.sha256:
                config_error = "evaluation-config-mismatch"
        except EvaluationError:
            config_error = "invalid-embedded-evaluation-config"
    if metadata.get("evaluation_config_sha256", config.sha256) != config.sha256:
        config_error = "evaluation-config-hash-mismatch"
    length = metadata.get("evaluation_route_length_m", metadata.get("evaluation_length_m", metadata.get("route_length_m")))
    start = metadata.get("evaluation_start_reference_m", metadata.get("start_reference_m", 0.0))
    initialization_error = None
    try:
        tracker = EvaluationTracker(config, length, start, metadata.get("condition", "normal"))
    except (EvaluationError, ContractError, ValueError, TypeError) as error:
        initialization_error = str(error) if isinstance(error, EvaluationError) else "invalid-evaluation-metadata"
        # A failed bootstrap still produces a row. These fallback bounds exist
        # only to retain valid sample distances; completion is reported unknown.
        tracker = EvaluationTracker(config, max(19., config.route_completion_tolerance_m + 1.), 0., "unknown")
    if type(rows) not in (list, tuple):
        tracker._failure("invalid-evaluation-row-stream")
        rows = []
    for row in rows:
        remaining = row.get("remaining_horizon_m") if type(row) is dict else None
        if tracker.reason is not None:
            tracker._failure("observation-after-termination")
            break
        tracker.update(row, events=(), remaining_horizon_m=remaining,
                       technical_failure=type(row) is dict and row.get("technical_failure") is True,
                       intervention=type(row) is dict and row.get("intervention") is True)
    if initialization_error is not None or invalid_metadata:
        tracker._failure(initialization_error or "invalid-evaluation-metadata")
    if config_error is not None:
        tracker._failure(config_error)
    result = tracker.finalize(metadata, termination_reason=metadata.get("termination_reason"), events=events)
    if initialization_error is not None or invalid_metadata:
        result.update(evaluation_length_m=None, progress_m=None, completion_pct=None)
    return result


def aggregate_metrics(episodes):
    """Condition/method pooled metrics include every failed episode's samples."""
    groups = {}
    for episode in episodes:
        if type(episode) is not dict or episode.get("metric_version") != METRIC_VERSION:
            raise EvaluationError("invalid-episode-metrics")
        key = (episode["method"], episode["condition"], episode["evaluation_config_sha256"])
        groups.setdefault(key, []).append(episode)
    result = []
    for (method, condition, config_hash), rows in sorted(groups.items()):
        distance = math.fsum(row["actual_distance_m"] for row in rows)
        collisions = sum(row["collision_count"] for row in rows)
        lanes = sum(row["lane_invasion_count"] for row in rows)
        pooled = {}
        for field in ("lateral_abs_m", "heading_abs_deg", "steering_rate_abs_per_second", "steering_acceleration_abs_per_second2"):
            pooled[field] = _summary([value for row in rows for value in row["pooled_samples"][field]])
        latency_keys = {key for row in rows for key in row["pooled_samples"]["latency_ms"]}
        latency = {key: _summary([value for row in rows for value in row["pooled_samples"]["latency_ms"].get(key, [])])
                   for key in sorted(latency_keys)}
        result.append({"method": method, "condition": condition, "evaluation_config_sha256": config_hash,
                       "episode_count": len(rows), "successful_episodes": sum(row["driving_success"] for row in rows),
                       "failed_episodes": sum(not row["driving_success"] for row in rows),
                       "technical_failure_episodes": sum(row["termination_reason"] == "technical_failure" for row in rows),
                       "integrity_failure_episodes": sum(not row["integrity_valid"] for row in rows),
                       "completion_pct": _summary([row["completion_pct"] for row in rows if row["completion_pct"] is not None]),
                       "sample_count": sum(row["sample_count"] for row in rows),
                       "actual_distance_m": distance, "collision_count": collisions, "lane_invasion_count": lanes,
                       "collision_per_km": collisions / (distance / 1000) if distance else None,
                       "lane_invasion_per_km": lanes / (distance / 1000) if distance else None,
                       "recovery_required_episodes": sum(row["recovery_required"] for row in rows),
                       "recovery_successful_episodes": sum(row["recovery_success"] and row["recovery_required"] for row in rows),
                       "recovery_time_seconds": _summary([row["recovery_time_seconds"] for row in rows
                                                          if row["recovery_required"] and row["recovery_time_seconds"] is not None]),
                       "latency_ms": latency, **pooled})
    return {"metric_version": METRIC_VERSION, "episode_count": len(episodes), "groups": result}


CSV_FIELDS = ("metric_version", "evaluation_config_sha256", "episode_id", "route_id", "method", "condition",
              "termination_reason", "termination_flags", "driving_success", "integrity_valid", "integrity_errors",
              "sample_count", "duration_seconds", "evaluation_length_m", "progress_m", "completion_pct",
              "actual_distance_m", "collision_count", "lane_invasion_count", "collision_per_km", "lane_invasion_per_km",
              "intervention_free_distance_m", "recovery_required", "recovery_success", "recovery_time_seconds", "terminal_lateral_abs_m",
              "terminal_heading_abs_deg", "minimum_body_lane_margin_m", "lateral_abs_m", "heading_abs_deg",
              "steering_rate_abs_per_second", "steering_acceleration_abs_per_second2", "latency_ms")


def write_metric_csv(episode_metrics, output_csv, *, config=None):
    """Exclusive atomic publication; never overwrites evidence or existing output."""
    episodes = list(episode_metrics)
    aggregate = aggregate_metrics(episodes)
    output = Path(output_csv)
    if output.exists():
        raise EvaluationError("metric-output-already-exists")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for episode in episodes:
        value = {}
        for field in CSV_FIELDS:
            item = episode[field]
            value[field] = "N/A" if item is None else (json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
                                                       if isinstance(item, (dict, list)) else item)
        writer.writerow(value)
    encoded = stream.getvalue().encode("utf-8")
    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".metrics-", suffix=".tmp", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation is atomic and exclusive; unlike replace it cannot
        # clobber a simultaneously published report. Temporary and final share FS.
        os.link(temporary, output)
    except FileExistsError as error:
        raise EvaluationError("metric-output-already-exists") from error
    except OSError as error:
        raise EvaluationError("metric-publication-failed") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return aggregate
