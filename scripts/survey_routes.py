"""Read Town01 topology and pin two separated D5 corridors without ticking."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.routes import RouteError, load_route_catalog, select_route_catalog
from scripts.verify_collection import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "routes_d5.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            catalog = load_route_catalog(args.output)
            print(json.dumps({"status": "CONFIG_VALIDATED", "simulator_verified": False,
                              "routes": [r["route_id"] for r in catalog["routes"]]}))
            return 0
        if args.output.exists():
            raise RouteError("route-catalog-already-exists")
        carla = importlib.import_module("carla")
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(30)
        if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
            raise RouteError("carla-version-mismatch")
        world = client.get_world()
        catalog = select_route_catalog(world.get_map(), carla)
        write_report(args.output, catalog)
        print(json.dumps({"status": catalog["status"], "routes": [
            {"route_id": r["route_id"], "spawn_point_index": r["spawn_point_index"]} for r in catalog["routes"]],
            "corridor_separation_m": catalog["measured_corridor_separation_m"]}, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL", "code": error.code if isinstance(error, RouteError) else "survey-operation-failed",
                          "type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
