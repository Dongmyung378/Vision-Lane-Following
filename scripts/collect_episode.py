"""Collect a D4 stationary episode on a verified D5 target route."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.config import ConfigError, collection_config_from_dict, load_collection_config
from carla_tools.routes import RouteError, load_route_catalog, verify_route_in_world
from scripts.verify_collection import write_report


def identifier(value: str) -> str:
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) or value.upper() in reserved:
        raise argparse.ArgumentTypeError("Use a portable identifier of 1-64 letters, digits, underscores or hyphens.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "collection_d4.json")
    parser.add_argument("--routes", type=Path, default=ROOT / "configs" / "routes_d5.json")
    parser.add_argument("--route-id", type=identifier, required=True)
    parser.add_argument("--episode-id", type=identifier, required=True)
    parser.add_argument("--dataset-version", type=identifier, default="d4-smoke-v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "local" / "dataset-d4-smoke-v1")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    stage = "configuration"
    try:
        cfg = load_collection_config(args.config)
        catalog = load_route_catalog(args.routes)
        routes = [route for route in catalog["routes"] if route["route_id"] == args.route_id]
        if len(routes) != 1:
            raise RouteError("unknown-route-id")
        route = routes[0]
        raw = cfg.to_dict()
        raw["vehicle"]["spawn_point_index"] = route["spawn_point_index"]
        cfg = collection_config_from_dict(raw)
        metadata = {
            "dataset_version": args.dataset_version, "episode_id": args.episode_id,
            "route_id": route["route_id"], "scene_group_id": route["scene_group_id"],
            "town": cfg.map, "route_contract": route,
            "perturbation": {"lateral_m": 0.0, "yaw_deg": 0.0, "provenance": "stationary-no-perturbation"},
            "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6",
        }
        if args.validate_only:
            print(json.dumps({"status": "CONFIG_VALIDATED", "simulator_verified": False,
                              "configuration_sha256": cfg.sha256, "route_id": route["route_id"],
                              "episode_id": args.episode_id}, indent=2))
            return 0
        destination = args.output_root / args.episode_id
        if destination.exists():
            raise RouteError("episode-destination-already-exists")
        stage = "live-route-validation"
        carla = importlib.import_module("carla")
        client = carla.Client(cfg.connection.host, cfg.connection.port)
        client.set_timeout(cfg.connection.timeout_seconds)
        if client.get_client_version() != cfg.carla_version or client.get_server_version() != cfg.carla_version:
            raise RouteError("carla-version-mismatch")
        verify_route_in_world(route, client.get_world().get_map(), carla)
        stage = "episode-capture"
        from carla_tools.capture import capture_episode
        report = capture_episode(cfg, destination, metadata, carla_module=carla)
        report.update(episode_id=args.episode_id, dataset_version=args.dataset_version, route_id=route["route_id"])
        write_report(args.output_root / f"{args.episode_id}_capture_report.json", report)
        summary = {key: report[key] for key in ("status", "episode_id", "route_id", "frames_written", "metrics", "cleanup", "error") if key in report}
        summary["report"] = f"{args.episode_id}_capture_report.json"
        summary["validation"] = {key: report.get("validation", {}).get(key) for key in ("status", "frame_count", "complete", "errors")}
        print(json.dumps(summary, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        code = error.code if isinstance(error, RouteError) else "invalid-configuration" if isinstance(error, ConfigError) else "episode-cli-failed"
        print(json.dumps({"status": "FAIL", "error": {"stage": stage, "code": code, "type": type(error).__name__}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
