"""Data CLI boundaries: offline validation, safe destinations and fail-closed inputs."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools import capture as capture_module
from scripts import build_split_manifest as split_cli
from scripts import collect_episode as collect_cli
from scripts import survey_routes as survey_cli
from scripts import validate_episode as validate_cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "collection_d4.json"
ROUTES = ROOT / "configs" / "routes_d5.json"


def invoke(main, argv):
    """Inspect the JSON contract without coupling tests to its formatting."""
    output = io.StringIO()
    with redirect_stdout(output):
        code = main(argv)
    return code, json.loads(output.getvalue())


def transform(anchor):
    return SimpleNamespace(location=SimpleNamespace(**anchor["world_location"]),
                           rotation=SimpleNamespace(**anchor["world_rotation"]))


class RecordedWaypoint:
    """A read-only waypoint chain from the pinned real catalog, without CARLA."""

    def __init__(self, anchors, index=0):
        anchor = anchors[index]
        self.anchors, self.index = anchors, index
        self.road_id, self.section_id, self.lane_id = (
            anchor[key] for key in ("road_id", "section_id", "lane_id"))
        self.s = anchor["s"]
        self.transform = transform(anchor)
        self.lane_width = anchor["lane_width_m"]
        self.is_junction = anchor["is_junction"]
        self.lane_type = "Driving"

    def next(self, distance):
        if distance != 1.0:
            raise AssertionError("This fixture represents the one-metre pinned survey.")
        return [RecordedWaypoint(self.anchors, min(self.index + 1, len(self.anchors) - 1))]


class RecordedMap:
    def __init__(self, route):
        self.route = route
        self.name = route["map_package"]

    def get_spawn_points(self):
        return [transform(self.route["start"])]

    def get_waypoint(self, location, **kwargs):
        return RecordedWaypoint(self.route["centerline_anchor_sequence"])

    def to_opendrive(self):
        # The geometry can match while the live map source no longer matches.
        return "<OpenDRIVE>changed-map-source</OpenDRIVE>"


class DataCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(ROUTES.read_text(encoding="utf-8"))
        cls.route = cls.catalog["routes"][0]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "output"

    def collect_args(self, *, route_id=None, episode_id="episode_001", extra=()):
        return ["--config", str(CONFIG), "--routes", str(ROUTES),
                "--route-id", route_id or self.route["route_id"],
                "--episode-id", episode_id, "--output-root", str(self.output), *extra]

    def test_collect_validate_only_is_offline_and_does_not_create_output(self):
        with patch.object(collect_cli.importlib, "import_module") as importer, \
                patch.object(capture_module, "capture_episode") as capture:
            code, report = invoke(collect_cli.main, self.collect_args(extra=["--validate-only"]))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "CONFIG_VALIDATED")
        self.assertFalse(report["simulator_verified"])
        self.assertEqual(report["route_id"], self.route["route_id"])
        self.assertEqual(len(report["configuration_sha256"]), 64)
        self.assertEqual(self.route["map_package"], "Carla/Maps/Town01")
        importer.assert_not_called()
        capture.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unknown_route_is_rejected_before_carla_import(self):
        with patch.object(collect_cli.importlib, "import_module") as importer, \
                patch.object(capture_module, "capture_episode") as capture:
            code, report = invoke(collect_cli.main, self.collect_args(route_id="unknown_route"))
        self.assertEqual(code, 1)
        self.assertEqual(report["error"]["code"], "unknown-route-id")
        self.assertEqual(report["error"]["stage"], "configuration")
        importer.assert_not_called()
        capture.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_and_windows_reserved_ids_are_rejected_by_argument_parser(self):
        invalid = ("../escape", "a/b", "a\\b", "bad.name", "-bad", "x" * 65,
                   "CON", "con", "PRN", "AUX", "NUL", "COM1", "LPT9")
        with patch.object(collect_cli.importlib, "import_module") as importer:
            for option in ("--route-id", "--episode-id", "--dataset-version"):
                for value in invalid:
                    args = self.collect_args()
                    if option in args:
                        index = args.index(option)
                        del args[index:index + 2]
                    args.append(f"{option}={value}")
                    with self.subTest(option=option, value=value), \
                            redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        collect_cli.main(args)
                    self.assertEqual(raised.exception.code, 2)
        importer.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_destination_is_preserved_without_importing_carla(self):
        destination = self.output / "episode_001"
        destination.mkdir(parents=True)
        sentinel = destination / "episode.json"
        original = b'{"status":"COMPLETE","sentinel":"keep exactly"}\n'
        sentinel.write_bytes(original)
        with patch.object(collect_cli.importlib, "import_module") as importer, \
                patch.object(capture_module, "capture_episode") as capture:
            code, report = invoke(collect_cli.main, self.collect_args())
        self.assertEqual(code, 1)
        self.assertEqual(report["error"]["code"], "episode-destination-already-exists")
        importer.assert_not_called()
        capture.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [destination])
        self.assertEqual(list(destination.iterdir()), [sentinel])

    def test_existing_empty_destination_also_refuses_collection(self):
        destination = self.output / "episode_001"
        destination.mkdir(parents=True)
        with patch.object(collect_cli.importlib, "import_module") as importer:
            code, report = invoke(collect_cli.main, self.collect_args())
        self.assertEqual(code, 1)
        self.assertEqual(report["error"]["code"], "episode-destination-already-exists")
        importer.assert_not_called()
        self.assertEqual(list(destination.iterdir()), [])

    def test_changed_live_route_source_stops_before_capture(self):
        world_map = RecordedMap(copy.deepcopy(self.route))
        client = Mock()
        client.get_client_version.return_value = "0.9.16"
        client.get_server_version.return_value = "0.9.16"
        client.get_world.return_value = SimpleNamespace(get_map=lambda: world_map)
        carla = SimpleNamespace(Client=Mock(return_value=client),
                                LaneType=SimpleNamespace(Driving="Driving"))
        with patch.object(collect_cli.importlib, "import_module", return_value=carla) as importer, \
                patch.object(capture_module, "capture_episode") as capture:
            code, report = invoke(collect_cli.main, self.collect_args())
        self.assertEqual(code, 1)
        self.assertEqual(report["error"]["code"], "live-route-contract-mismatch")
        self.assertEqual(report["error"]["stage"], "live-route-validation")
        importer.assert_called_once_with("carla")
        carla.Client.assert_called_once_with("127.0.0.1", 2000)
        capture.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_survey_validate_only_accepts_native_map_prefix_without_mutation(self):
        original = ROUTES.read_bytes()
        with patch.object(survey_cli.importlib, "import_module") as importer:
            code, report = invoke(survey_cli.main, ["--output", str(ROUTES), "--validate-only"])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "CONFIG_VALIDATED")
        self.assertFalse(report["simulator_verified"])
        self.assertEqual(report["routes"], [route["route_id"] for route in self.catalog["routes"]])
        importer.assert_not_called()
        self.assertEqual(ROUTES.read_bytes(), original)

    def test_existing_survey_output_refuses_before_carla_import(self):
        target = self.directory / "routes.json"
        original = b"existing catalog bytes must survive\n"
        target.write_bytes(original)
        with patch.object(survey_cli.importlib, "import_module") as importer:
            code, report = invoke(survey_cli.main, ["--output", str(target)])
        self.assertEqual(code, 1)
        self.assertEqual(report["code"], "route-catalog-already-exists")
        importer.assert_not_called()
        self.assertEqual(target.read_bytes(), original)

    def test_missing_episode_validation_fails_offline_without_creating_files(self):
        target = self.directory / "missing-episode"
        with patch.object(collect_cli.importlib, "import_module") as importer:
            code, report = invoke(validate_cli.main, [str(target)])
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["errors"])
        self.assertNotIn("metadata", report)
        importer.assert_not_called()
        self.assertFalse(target.exists())

    def test_validation_stdout_omits_metadata_and_retains_failure_evidence(self):
        for status, code in (("PASS", 0), ("FAIL", 1)):
            stored = {"status": status, "frame_count": 60, "complete": status == "PASS",
                      "errors": [] if status == "PASS" else [{"code": "image-hash-mismatch"}],
                      "metadata": {"route_contract": {"large_private_field": "not stdout"}}}
            with self.subTest(status=status), \
                    patch.object(validate_cli, "validate_episode", return_value=stored):
                actual_code, report = invoke(validate_cli.main, [str(self.directory / "episode")])
            self.assertEqual(actual_code, code)
            self.assertNotIn("metadata", report)
            self.assertEqual(report["errors"], stored["errors"])
            self.assertEqual(report["frame_count"], 60)

    def test_malformed_duplicate_and_nonfinite_assignments_fail_before_builder(self):
        assignments = self.directory / "assignments.json"
        output = self.directory / "split.csv"
        cases = ("{", '{"episode_001":"train","episode_001":"dev"}',
                 '{"episode_001":NaN}', '{"episode_001":1e999}')
        for content in cases:
            assignments.write_text(content, encoding="utf-8")
            with self.subTest(content=content), \
                    patch.object(split_cli, "build_split_manifest") as builder:
                code, report = invoke(split_cli.main, ["--episodes", str(self.directory / "episode"),
                                                       "--assignments", str(assignments),
                                                       "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(report, {"status": "FAIL", "code": "invalid-split-assignments"})
            builder.assert_not_called()
            self.assertFalse(output.exists())

    def test_valid_assignments_are_forwarded_without_changing_group_ids(self):
        assignments = self.directory / "assignments.json"
        requested = {"episode_001": "train", "episode_002": "dev"}
        assignments.write_text(json.dumps(requested), encoding="utf-8")
        episodes = [self.directory / "episode_001", self.directory / "episode_002"]
        output = self.directory / "split.csv"
        with patch.object(split_cli, "build_split_manifest",
                          return_value={"status": "PASS", "sample_count": 120}) as builder:
            code, report = invoke(split_cli.main, ["--episodes", *(str(path) for path in episodes),
                                                   "--assignments", str(assignments),
                                                   "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(report["sample_count"], 120)
        builder.assert_called_once_with(episodes, output, requested,
                                        contract_path=split_cli.DEFAULT_CONTRACT_PATH)
        self.assertEqual(json.loads(assignments.read_text(encoding="utf-8")), requested)


if __name__ == "__main__":
    unittest.main()
