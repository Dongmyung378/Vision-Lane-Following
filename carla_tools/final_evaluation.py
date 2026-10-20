"""Declare held-out evaluation before observing any final RGB or outcomes.

This module plans and checks inputs on CPU. Static OpenDrive topology is
planning evidence, not a driving result. Methods share the same controller,
initial states and physical environment; only the privileged reference may
consume the pinned target route.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import PurePosixPath
import re

from .contracts import canonical_sha256, route_sha256, validate_route_contract
from .evaluation import TERMINATION_PRECEDENCE, load_evaluation_config

VERSION = "vlf-heldout-evaluation-v1"
ROUTE_SCOPE = "final-static-opendrive-topology"
METHODS = ("B0", "M1", "privileged")
TOWNS = ("Town01", "Town02")
INITIAL_STATES = (
    {"state_id": "center_s0", "seed": 42, "initial_reference_m": 0.,
     "lateral_m": 0., "yaw_deg": 0.},
    {"state_id": "right_s2", "seed": 43, "initial_reference_m": 2.,
     "lateral_m": .2, "yaw_deg": 2.},
    {"state_id": "left_s4", "seed": 44, "initial_reference_m": 4.,
     "lateral_m": -.2, "yaw_deg": -2.},
)
ENVIRONMENTS = (
    ("A", "Town01", "ClearNoon", 0.),
    ("B", "Town02", "ClearNoon", 0.),
    ("C", "Town01", "HardRainNoon", 0.),
    ("D", "Town01", "ClearSunset", 0.),
    ("E", "Town01", "ClearNoon", 2.),
)
REQUIRED_ARTIFACT_ROLES = frozenset({"dataset", "split", "selection", "topology", "budget"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class FinalEvaluationError(ValueError):
    """A plan, geometry, leakage or binding failure."""


def _require(condition, code):
    if not condition:
        raise FinalEvaluationError(code)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def portable_path(value):
    """Reject absolute paths, drive names, alternate streams and traversal."""
    _require(type(value) is str and bool(value) and "\\" not in value
             and ":" not in value and not PurePosixPath(value).is_absolute()
             and all(part not in ("", ".", "..") for part in value.split("/")),
             "final-unsafe-artifact-path")
    return value


def runtime_route_contract(route, spawn_point_index):
    """Lower frozen topology to the historical drive-loop input contract.

    The native caller supplies a measured, existing bootstrap spawn index.
    Its spawn pose is not the evaluation start: the drive loop resets to the
    frozen route anchor and explicit initial reference before measuring.
    """
    _require(type(spawn_point_index) is int and 0 <= spawn_point_index <= 100000,
             "final-bootstrap-index-invalid")
    value = copy.deepcopy(route)
    value["spawn_point_index"] = spawn_point_index
    value["selection_scope"] = "D5-topology-only-not-frame-labels"
    value.pop("route_sha256", None)
    value["route_sha256"] = route_sha256(value)
    validate_route_contract(value)
    return value


def validate_route(route):
    _require(type(route) is dict and route.get("town") in TOWNS
             and route.get("spawn_point_index", "missing") is None
             and route.get("selection_scope") == ROUTE_SCOPE,
             "final-static-topology-route-required")
    _require(route.get("route_sha256") == route_sha256(route),
             "final-topology-route-hash-mismatch")
    # Zero is used solely to exercise the old structural validator. It is
    # never stored in the topology catalog or used as a native spawn guess.
    structural = runtime_route_contract(route, 0)
    from .driving_route import FixedRouteTracker
    FixedRouteTracker(structural)
    for anchor in route["centerline_anchor_sequence"]:
        rotation = anchor.get("world_rotation")
        _require(type(rotation) is dict and set(rotation) == {"pitch", "yaw", "roll"}
                 and all(_number(value) for value in rotation.values()),
                 "final-topology-anchor-heading-required")
    return route


def _segment_distance(a, b, c, d):
    """Exact XY distance between two closed line segments."""
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def point(p, q, r):
        dx, dy = r[0] - q[0], r[1] - q[1]
        length = dx * dx + dy * dy
        t = 0. if length == 0 else min(1., max(0., ((p[0] - q[0]) * dx + (p[1] - q[1]) * dy) / length))
        return math.hypot(p[0] - q[0] - t * dx, p[1] - q[1] - t * dy)

    if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
        return 0.
    return min(point(a, c, d), point(b, c, d), point(c, a, b), point(d, a, b))


def corridor_separation(first, second):
    """Use whole XY segments, including opposite traffic directions."""
    if first["town"] != second["town"]:
        return math.inf
    def xy(route):
        return [(p["world_location"]["x"], p["world_location"]["y"])
                for p in route["centerline_anchor_sequence"]]
    aa, bb = xy(first), xy(second)
    return min(_segment_distance(a, b, c, d)
               for a, b in zip(aa, aa[1:]) for c, d in zip(bb, bb[1:]))


def validate_route_catalog(catalog):
    _require(type(catalog) is dict and catalog.get("schema_version") == 1
             and type(catalog.get("schema_version")) is int
             and catalog.get("split") == "final"
             and type(catalog.get("routes")) is list and len(catalog["routes"]) == 4,
             "final-four-topology-routes-required")
    routes = catalog["routes"]
    for route in routes:
        validate_route(route)
    _require(len({r["route_id"] for r in routes}) == 4
             and len({r["scene_group_id"] for r in routes}) == 4,
             "final-duplicate-route-or-scene")
    for town in TOWNS:
        pair = [route for route in routes if route["town"] == town]
        _require(len(pair) == 2 and len({r["opendrive_sha256"] for r in pair}) == 1,
                 "final-town-route-or-map-coverage")
        _require(corridor_separation(*pair) >= 75., "final-within-town-scenes-too-close")
    return catalog


def audit_route_exclusion(catalog, observed_routes, *, minimum_separation_m=75.):
    """Reject previously used IDs, scene groups and nearby physical corridors."""
    validate_route_catalog(catalog)
    _require(type(observed_routes) is list and bool(observed_routes)
             and _number(minimum_separation_m) and minimum_separation_m >= 75.,
             "final-development-route-inventory-required")
    observed = []
    for route in observed_routes:
        validate_route_contract(route)
        observed.append(route)
    comparisons = []
    for route in catalog["routes"]:
        for prior in observed:
            _require(route["route_id"] != prior["route_id"]
                     and route["scene_group_id"] != prior["scene_group_id"],
                     "final-seen-route-or-scene")
            if route["town"] != prior["town"]:
                continue
            _require(route["opendrive_sha256"] == prior["opendrive_sha256"],
                     "final-route-inventory-map-version-mismatch")
            distance = corridor_separation(route, prior)
            comparisons.append({"final_route_id": route["route_id"],
                                "observed_route_id": prior["route_id"],
                                "separation_m": distance})
            _require(distance >= minimum_separation_m, "final-route-near-observed-corridor")
    return {"status": "PASS", "scope": "static-topology-only-no-RGB-or-driving-outcomes",
            "catalog_sha256": canonical_sha256(catalog),
            "observed_routes_sha256": canonical_sha256(observed_routes),
            "minimum_separation_m": minimum_separation_m,
            "observed_route_count": len(observed_routes), "comparisons": comparisons,
            "inventory_completeness_requires_bound_dataset_and_development_evidence": True}


def metric_policy():
    return {
        "version": "heldout-paired-failure-inclusive-v1",
        "primary": "clean-completion-count-and-route-completion-by-group-and-method",
        "lateral_error": "pooled-all-driving-frame-absolute-mean-and-linear-interpolated-p95",
        "paired_comparison": "same-group-route-initial-state; preserve-all-three-methods",
        "episode_summary": "report-each-episode-mean-p95-completion-termination-and-integrity",
        "early_failures": "retain-in-denominator-and-report-with-observed-distance; low-error-is-not-success",
        "event_rate": "1000-times-total-deduplicated-events-over-total-actual-metres; zero-distance-null",
        "collision_dedup": "same-frame-and-other-actor",
        "lane_event_dedup": "same-frame-and-unordered-marking-set; no-cross-frame-window",
        "recovery": "only-nonzero-initial-lateral-or-yaw; frozen-tolerance-hold-and-deadline",
        "steering": "applied-normalized-steering-rate-and-second-derivative; not-physical-jerk",
        "latency": "batch1-same-T4; per-stage-and-total-median-p95; no-zero-fill",
        "latency_scope": "BGRA-conversion-RGB-perception-controller; excludes-render-sensor-wait-GT-disk-video",
        "reference_latency": "privileged-GT-construction-plus-controller-reported-separately",
        "processing_budget_ms": 50.,
        "steady_latency_excluded_initial_frames": 10,
        "quantile": "linear-interpolation-at-(n-1)*q",
        "aggregation": "group-and-method; pooled-frame-and-equal-episode-results-labelled-separately",
        "uncertainty": "two-route-clusters-per-group-and-three-initial-states; no-independent-training-seed-claim",
        "missing_trials": "never-drop; incomplete-matrix-prevents-completion-claim",
    }


def retry_policy():
    return {
        "maximum_attempts_per_trial": 2,
        "automatic_retries": False,
        "allowed": "documented-infrastructure-failure-only-with-identical-freeze-and-trial",
        "excluded": ["collision", "lane_departure", "stalled", "timeout", "prediction_failure", "controller_rejection"],
        "preserve": "every-attempt-and-original-failure; one-explicit-designated-attempt-per-trial",
        "unavailable_map_or_route": "record-blocked-trial; never-substitute-route-or-relax-threshold",
        "semantic_code_change": "invalidate-freeze-and-rerun-all-affected-methods; no-result-based-selection",
    }


def _validate_bindings(bindings):
    _require(type(bindings) is list and bool(bindings), "final-artifact-bindings-required")
    paths, roles = set(), set()
    for row in bindings:
        _require(type(row) is dict and set(row) == {"path", "sha256", "roles"},
                 "final-artifact-binding-fields")
        portable_path(row["path"])
        _require(row["path"].casefold() not in paths and type(row["sha256"]) is str
                 and _SHA.fullmatch(row["sha256"]) is not None,
                 "final-artifact-duplicate-or-hash")
        _require(type(row["roles"]) is list and bool(row["roles"])
                 and all(type(role) is str and role in REQUIRED_ARTIFACT_ROLES for role in row["roles"])
                 and len(row["roles"]) == len(set(row["roles"])), "final-artifact-roles")
        paths.add(row["path"].casefold())
        roles.update(row["roles"])
    _require(roles >= REQUIRED_ARTIFACT_ROLES, "final-missing-provenance-role")


def planned_trials(manifest):
    """Create exactly paired physical starts, with counterbalanced method order."""
    trials, paired = [], 0
    for group in manifest["groups"]:
        for route_id in group["route_ids"]:
            for state in manifest["initial_states"]:
                order = METHODS[paired % len(METHODS):] + METHODS[:paired % len(METHODS)]
                for method in order:
                    name = f"{group['group_id']}__{route_id}__{state['state_id']}__{method}"
                    trials.append({"trial_id": name, "group_id": group["group_id"],
                                   "route_id": route_id, "town": group["town"], "method": method,
                                   "initial_state": copy.deepcopy(state),
                                   "environment": copy.deepcopy(group["environment"]),
                                   "episode": "episodes/" + name})
                paired += 1
    return trials


def build_manifest(routes, selection, artifact_bindings):
    from .baseline import load_baseline_config
    from .config import load_collection_config
    from pathlib import Path
    validate_route_catalog(routes)
    _validate_bindings(artifact_bindings)
    collection = load_collection_config(Path(__file__).resolve().parents[1] / "configs/collection_d4.json")
    evaluation = replace(load_evaluation_config(), timeout_seconds=10.)
    groups = [{"group_id": group, "town": town,
               "route_ids": sorted(r["route_id"] for r in routes["routes"] if r["town"] == town),
               "environment": {"weather": weather, "camera_pitch_delta_deg": pitch}}
              for group, town, weather, pitch in ENVIRONMENTS]
    value = {
        "schema_version": 1, "protocol_version": VERSION, "split": "final",
        "status": "PLANNED_UNOBSERVED", "final_results_observed": False,
        "selected_pipeline_id": selection["pipeline_id"],
        "routes_sha256": canonical_sha256(routes),
        "controller_config": copy.deepcopy(selection["controller_config"]),
        "preprocessing": copy.deepcopy(selection["preprocessing"]),
        "collection_config": collection.to_dict(),
        "baseline_config": load_baseline_config().to_dict(),
        "evaluation_config": evaluation.to_dict(),
        "groups": groups, "initial_states": copy.deepcopy(list(INITIAL_STATES)),
        "methods": list(METHODS), "route_length_m": 15., "warmup_frames": 10,
        "expected_episodes": 90, "paired_states": 30, "episodes_per_method": 30,
        "seeds_are_distinct_physical_initial_states_not_independent_training_repeats": True,
        "camera_calibration_policy": "B0-and-M1-use-fixed-nominal-calibration; shifted-extrinsics-evaluation-only",
        "privileged_reference": "GT-waypoints-from-original-route-and-same-frame-pose; common-controller",
        "bootstrap_policy": "nearest-live-spawn-XY-to-frozen-start-then-lowest-index; reset-to-frozen-route-start",
        "termination_precedence": list(TERMINATION_PRECEDENCE),
        "metrics": metric_policy(), "retry_policy": retry_policy(),
        "artifact_bindings": copy.deepcopy(artifact_bindings),
        "execution_stage": "after-freeze-verification-only",
        "selection_after_results_allowed": False,
    }
    value["trials"] = planned_trials(value)
    validate_manifest(value, routes, selection)
    return value


def validate_manifest(manifest, routes, selection):
    """Validate plans without loading a model or producing final observations."""
    from .baseline import baseline_config_from_dict, load_baseline_config
    from .config import collection_config_from_dict, load_collection_config
    from .controller_study import study_controller_config_from_dict
    from .evaluation import evaluation_config_from_dict
    from .selected_pipeline import _validate_selection
    from pathlib import Path
    required = {
        "schema_version", "protocol_version", "split", "status", "final_results_observed",
        "selected_pipeline_id", "routes_sha256", "controller_config", "preprocessing",
        "collection_config", "baseline_config", "evaluation_config", "groups", "initial_states",
        "methods", "route_length_m", "warmup_frames", "expected_episodes", "paired_states",
        "episodes_per_method", "seeds_are_distinct_physical_initial_states_not_independent_training_repeats",
        "camera_calibration_policy", "privileged_reference", "bootstrap_policy",
        "termination_precedence", "metrics", "retry_policy", "artifact_bindings",
        "execution_stage", "selection_after_results_allowed", "trials"}
    _require(type(manifest) is dict and set(manifest) == required, "final-manifest-fields")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
             and manifest["protocol_version"] == VERSION and manifest["split"] == "final"
             and manifest["status"] == "PLANNED_UNOBSERVED"
             and manifest["final_results_observed"] is False
             and manifest["selection_after_results_allowed"] is False,
             "final-manifest-version-or-observation-policy")
    validate_route_catalog(routes)
    _validate_selection(selection)
    _require(manifest["selected_pipeline_id"] == selection["pipeline_id"]
             and manifest["routes_sha256"] == canonical_sha256(routes),
             "final-route-or-pipeline-identity-mismatch")
    _require(manifest["controller_config"] == selection["controller_config"]
             and manifest["preprocessing"] == selection["preprocessing"],
             "final-selected-controller-or-preprocessing-changed")
    controller = study_controller_config_from_dict(manifest["controller_config"])
    _require(controller.base.target_speed_kmh == 25. and controller.fixed_delta_seconds == .05,
             "final-speed-or-period-mismatch")
    collection = collection_config_from_dict(manifest["collection_config"])
    nominal = load_collection_config(Path(__file__).resolve().parents[1] / "configs/collection_d4.json")
    _require(collection == nominal, "final-nominal-camera-or-collection-changed")
    baseline_config_from_dict(manifest["baseline_config"])
    _require(manifest["baseline_config"] == load_baseline_config().to_dict(),
             "final-B0-calibration-or-parameters-changed")
    evaluation_config_from_dict(manifest["evaluation_config"])
    _require(manifest["evaluation_config"] == replace(load_evaluation_config(), timeout_seconds=10.).to_dict(),
             "final-evaluation-thresholds-changed")
    expected_groups = [{"group_id": group, "town": town,
                        "route_ids": sorted(r["route_id"] for r in routes["routes"] if r["town"] == town),
                        "environment": {"weather": weather, "camera_pitch_delta_deg": pitch}}
                       for group, town, weather, pitch in ENVIRONMENTS]
    _require(canonical_sha256(manifest["groups"]) == canonical_sha256(expected_groups)
             and canonical_sha256(manifest["initial_states"]) == canonical_sha256(list(INITIAL_STATES))
             and manifest["methods"] == list(METHODS), "final-groups-states-or-methods-changed")
    _require(type(manifest["route_length_m"]) in (float, int) and manifest["route_length_m"] == 15.
             and type(manifest["warmup_frames"]) is int and manifest["warmup_frames"] == 10,
             "final-drive-length-or-warmup-changed")
    _require(all(route["route_length_m"] >= 4. + manifest["route_length_m"] + 15.
                 for route in routes["routes"]), "final-future-corridor-too-short")
    constants = {
        "expected_episodes": 90, "paired_states": 30, "episodes_per_method": 30,
        "seeds_are_distinct_physical_initial_states_not_independent_training_repeats": True,
        "camera_calibration_policy": "B0-and-M1-use-fixed-nominal-calibration; shifted-extrinsics-evaluation-only",
        "privileged_reference": "GT-waypoints-from-original-route-and-same-frame-pose; common-controller",
        "bootstrap_policy": "nearest-live-spawn-XY-to-frozen-start-then-lowest-index; reset-to-frozen-route-start",
        "execution_stage": "after-freeze-verification-only",
    }
    _require(all(type(manifest[key]) is type(expected) and manifest[key] == expected
                 for key, expected in constants.items()), "final-budget-or-execution-policy-changed")
    _require(manifest["termination_precedence"] == list(TERMINATION_PRECEDENCE)
             and canonical_sha256(manifest["metrics"]) == canonical_sha256(metric_policy())
             and canonical_sha256(manifest["retry_policy"]) == canonical_sha256(retry_policy()),
             "final-metric-or-retry-policy-changed")
    _validate_bindings(manifest["artifact_bindings"])
    _require(canonical_sha256(manifest["trials"]) == canonical_sha256(planned_trials(manifest))
             and len(manifest["trials"]) == 90, "final-paired-trial-list-mismatch")
    return manifest
