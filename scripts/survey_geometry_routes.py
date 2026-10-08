"""Read-only Town01 straight/left/right geometry survey for D6 native QA."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.geometry_survey import load_geometry_catalog, survey_geometry_catalog
from carla_tools.routes import RouteError
from scripts.verify_collection import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "local" / "geometry_routes_d6.json")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            catalog = load_geometry_catalog(args.output)
            print(json.dumps({"status": "CONFIG_VALIDATED", "simulator_verified": False,
                              "cases": [item["case"] for item in catalog["geometry_cases"]]}))
            return 0
        if args.output.exists() or args.output.is_symlink():
            raise RouteError("geometry-catalog-already-exists")
        if not 1 <= args.port <= 65534 or not 0 < args.timeout <= 120:
            raise RouteError("invalid-connection-parameters")
        carla = importlib.import_module("carla")
        client = carla.Client("127.0.0.1", args.port)
        client.set_timeout(args.timeout)
        if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
            raise RouteError("carla-version-mismatch")
        catalog = survey_geometry_catalog(client.get_world().get_map(), carla)
        write_report(args.output, catalog)
        print(json.dumps({"status": catalog["status"], "spawn_points_scanned": catalog["spawn_points_scanned"],
                          "candidate_counts": catalog["candidate_counts"], "missing_cases": catalog["missing_cases"],
                          "cases": [{"case": item["case"], "spawn_point_index": item["route"]["spawn_point_index"],
                                     "heading_change_deg": item["geometry"]["waypoint_net_heading_change_deg"]}
                                    for item in catalog["geometry_cases"]]}, indent=2))
        return 0 if catalog["status"] == "GEOMETRY_SURVEY_VERIFIED" else 1
    except Exception as error:
        print(json.dumps({"status": "FAIL", "code": error.code if isinstance(error, RouteError) else "geometry-survey-operation-failed",
                          "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
