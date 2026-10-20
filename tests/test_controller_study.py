"""Analytic path geometry, control bounds and legacy compatibility checks."""

import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from carla_tools.controller import ControllerConfig, ControllerError, PurePursuitController, VehicleGeometry
from carla_tools.controller_study import (
    ControllerStudyConfig, StanleyController, StudyPurePursuitController,
    build_controller, load_study_controller_config, study_controller_config_from_dict,
)


def geometry(rear_y=0.0):
    return VehicleGeometry(2.8, -1.4, 1.4, 70.0, ((0.0, 1.0), (100.0, 1.0)),
                           front_track_m=1.6, rear_axle_y_m=rear_y, front_axle_y_m=rear_y)


def straight(lateral=0.0, heading=0.0):
    c, s = math.cos(heading), math.sin(heading)
    return [[c * x - s * lateral, s * x + c * lateral] for x in (3.0, 6.0, 9.0, 12.0, 15.0)]


def actor_circle(radius, sign=1.0, rear_y=0.0):
    # Independently construct actor positions by rotating its rigid offset about
    # a rear-axle ICR; this is not the controller's three-point circle solver.
    rear_radius = -rear_y + sign * math.sqrt(radius * radius - 1.4 * 1.4)
    result = []
    for arc in (3.0, 6.0, 9.0, 12.0, 15.0):
        theta = sign * arc / radius
        result.append([1.4 * (math.cos(theta) - 1.0) + (rear_radius + rear_y) * math.sin(theta),
                       1.4 * math.sin(theta) + (-rear_y - rear_radius) * (math.cos(theta) - 1.0)])
    return result, rear_radius


class StudyConfigurationTests(unittest.TestCase):
    def test_default_checked_in_wrapper_and_legacy_configuration_remain_distinct(self):
        cfg = load_study_controller_config()
        self.assertEqual(cfg, ControllerStudyConfig())
        self.assertEqual(cfg.base, ControllerConfig())
        self.assertEqual(cfg.target_speed_kmh, 25.0)
        self.assertEqual(cfg.fixed_delta_seconds, 0.05)
        with self.assertRaises(FrozenInstanceError):
            cfg.method = "stanley"
        with self.assertRaises(AttributeError):
            getattr(cfg, "invented_field")
        self.assertNotEqual(cfg.sha256, cfg.base.sha256)

    def test_wrapper_round_trip_hash_and_each_policy_parameter_bound(self):
        cfg = ControllerStudyConfig(method="stanley")
        self.assertEqual(study_controller_config_from_dict(cfg.to_dict()), cfg)
        for key, value in (("stanley_cross_track_gain", 2), ("stanley_softening_mps", 2),
                           ("stanley_heading_gain", 0.8), ("steer_smoothing_alpha", 0.5)):
            changed = replace(cfg, **{key: value})
            self.assertNotEqual(changed.sha256, cfg.sha256)
        for key, value in (("schema_version", True), ("schema_version", 2),
                           ("controller_version", "old"), ("method", "unknown"),
                           ("stanley_cross_track_gain", float("nan")),
                           ("stanley_softening_mps", 0), ("stanley_heading_gain", True),
                           ("steer_smoothing_alpha", float("inf")),
                           ("steer_smoothing_alpha", 0), ("steer_smoothing_alpha", 1.01),
                           ("base", {})):
            with self.subTest(key=key, value=value), self.assertRaises(ControllerError):
                replace(cfg, **{key: value})
        for mutation in (lambda raw: raw.pop("method"), lambda raw: raw.update(unknown=1),
                         lambda raw: raw["base"].update(speed_kp=True)):
            raw = cfg.to_dict()
            mutation(raw)
            with self.assertRaises(ControllerError):
                study_controller_config_from_dict(raw)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"method":"stanley","method":"pure_pursuit"}', encoding="utf-8")
            with self.assertRaisesRegex(ControllerError, "duplicate-json-key"):
                load_study_controller_config(path)

    def test_factory_dispatch_and_mismatched_constructors(self):
        measured = geometry()
        self.assertIs(type(build_controller(ControllerConfig(), measured)), PurePursuitController)
        self.assertIs(type(build_controller(ControllerStudyConfig(), measured)), StudyPurePursuitController)
        self.assertIs(type(build_controller(ControllerStudyConfig(method="stanley"), measured)), StanleyController)
        with self.assertRaises(ControllerError):
            build_controller({}, measured)
        with self.assertRaises(ControllerError):
            StanleyController(ControllerStudyConfig(), measured)
        with self.assertRaises(ControllerError):
            StudyPurePursuitController(ControllerStudyConfig(method="stanley"), measured)


class StudyPurePursuitTests(unittest.TestCase):
    def test_no_smoothing_has_exact_legacy_actuator_outputs_across_history(self):
        base = ControllerConfig()
        legacy = PurePursuitController(base, geometry())
        study = build_controller(ControllerStudyConfig(base=base), geometry())
        for index in range(100):
            points = straight(0.7 * math.sin(index / 8.0))
            speed = 4.0 + index / 50.0
            old, new = legacy.step(points, speed), study.step(points, speed)
            for key in ("throttle", "brake", "steer", "hand_brake", "reverse"):
                self.assertEqual(old[key], new[key])
            self.assertEqual(new["diagnostics"]["base_controller_configuration_sha256"], base.sha256)
            self.assertEqual(new["diagnostics"]["controller_method"], "pure_pursuit")

    def test_smoothing_precedes_slew_and_rejected_step_does_not_advance_state(self):
        cfg = ControllerStudyConfig(steer_smoothing_alpha=0.1)
        controller, replay = build_controller(cfg, geometry()), build_controller(cfg, geometry())
        first = controller.step(straight(0.3), 5.0)
        diag = first["diagnostics"]
        self.assertAlmostEqual(first["steer"], diag["steer_clipped"] * 0.1)
        with self.assertRaises(ControllerError):
            controller.step(straight(float("nan")), 5.0)
        second = controller.step(straight(-0.3), 5.0)
        self.assertEqual(first, replay.step(straight(0.3), 5.0))
        self.assertEqual(second, replay.step(straight(-0.3), 5.0))
        controller.reset()
        self.assertEqual(first, controller.step(straight(0.3), 5.0))


class StanleyGeometryTests(unittest.TestCase):
    def make(self, **kwargs):
        return build_controller(ControllerStudyConfig(method="stanley", **kwargs), geometry())

    def test_straight_zero_reference_and_actor_lateral_offset_have_expected_sign(self):
        for lateral in (-0.5, 0.0, 0.5):
            result = self.make().step(straight(lateral), 5.0)
            diag = result["diagnostics"]
            reference = diag["stanley_reference"]
            self.assertAlmostEqual(reference["cross_track_error_m"], lateral)
            self.assertEqual(reference["heading_error_rad"], 0.0)
            self.assertAlmostEqual(diag["stanley_cross_track_term_rad"], math.atan(lateral / 6.0))
            self.assertEqual(math.copysign(1, result["steer"]), math.copysign(1, lateral))
            if lateral == 0:
                self.assertEqual(result["steer"], 0.0)
                self.assertEqual(diag["effective_target_speed_mps"], 25.0 / 3.6)
            json.dumps(result, allow_nan=False)

    def test_heading_term_and_measured_front_projection_on_analytic_rotated_line(self):
        for heading in (-0.2, 0.2):
            cfg = ControllerStudyConfig(method="stanley")
            diag = build_controller(cfg, geometry()).step(straight(0.0, heading), 5.0)["diagnostics"]
            reference = diag["stanley_reference"]
            self.assertAlmostEqual(reference["heading_error_rad"], heading)
            # Front axle is right/left of the desired front path by the
            # perpendicular component of its measured actor-local x offset.
            self.assertAlmostEqual(reference["cross_track_error_m"], 1.4 * math.sin(heading))
            self.assertGreater(diag["steering_rad"] * heading, 0.0)

    def test_actor_circle_front_conversion_is_exact_for_both_turns_and_xy_offsets(self):
        for radius in (8.0, 13.6, 50.0):
            for sign in (-1.0, 1.0):
                for rear_y in (-0.1, 0.0, 0.1):
                    points, rear_radius = actor_circle(radius, sign, rear_y)
                    cfg = ControllerStudyConfig(method="stanley")
                    diag = build_controller(cfg, geometry(rear_y)).step(points, 4.0)["diagnostics"]
                    ref = diag["stanley_reference"]
                    with self.subTest(radius=radius, sign=sign, rear_y=rear_y):
                        self.assertAlmostEqual(ref["cross_track_error_m"], 0.0, places=10)
                        self.assertAlmostEqual(ref["target_front_axle_xy_m"][0], 1.4, places=10)
                        self.assertAlmostEqual(ref["target_front_axle_xy_m"][1], rear_y, places=10)
                        self.assertAlmostEqual(ref["rear_reference_curvature_per_m"], 1.0 / rear_radius)
                        self.assertAlmostEqual(ref["actor_reference_curvature_per_m"], sign / radius)
                        self.assertAlmostEqual(diag["steering_rad"], math.atan(2.8 / rear_radius))
                        self.assertAlmostEqual(diag["curvature_per_m"], 1.0 / rear_radius)
                        inner_radius = 2.8 / math.tan(abs(diag["native_inner_steering_rad"]))
                        self.assertAlmostEqual(inner_radius + 0.8, abs(rear_radius))

    def test_mirroring_reverses_steering_but_not_speed_or_preview(self):
        points, _ = actor_circle(13.6)
        right = self.make().step([[x, y + 0.2] for x, y in points], 5.0)
        left = self.make().step([[x, -y - 0.2] for x, y in points], 5.0)
        self.assertAlmostEqual(left["steer"], -right["steer"])
        self.assertAlmostEqual(left["diagnostics"]["curvature_per_m"], -right["diagnostics"]["curvature_per_m"])
        self.assertEqual(left["throttle"], right["throttle"])
        self.assertEqual(left["brake"], right["brake"])

    def test_softening_reduces_stopped_cross_track_term_without_division_by_zero(self):
        strong = self.make(stanley_softening_mps=0.5).step(straight(0.3), 0.0)
        soft = self.make(stanley_softening_mps=2.0).step(straight(0.3), 0.0)
        self.assertGreater(strong["diagnostics"]["stanley_cross_track_term_rad"],
                           soft["diagnostics"]["stanley_cross_track_term_rad"])
        self.assertTrue(math.isfinite(strong["steer"]))
        json.dumps(strong, allow_nan=False)

    def test_future_curve_preview_limits_speed_before_local_stanley_turn(self):
        result = self.make().step([[3., 0.], [6., 0.], [9., 0.], [11., 1.], [12., 3.]], 5.0)
        self.assertEqual(result["steer"], 0.0)
        self.assertEqual(result["diagnostics"]["curvature_per_m"], 0.0)
        self.assertTrue(result["diagnostics"]["curvature_speed_limited"])
        self.assertGreater(result["brake"], 0.0)

    def test_invalid_points_speed_timestep_and_unreachable_circle_preserve_all_state(self):
        cfg = ControllerStudyConfig(method="stanley", steer_smoothing_alpha=0.5)
        controller, replay = build_controller(cfg, geometry()), build_controller(cfg, geometry())
        first = controller.step(straight(0.2), 4.0)
        invalid = [(straight()[:4], 4., .05), (straight()[::-1], 4., .05),
                   (straight(float("nan")), 4., .05), (straight(), True, .05),
                   (straight(), float("inf"), .05), (straight(), -1., .05),
                   (straight(), 4., .1), (straight(), 4., True),
                   ([[3., 0.], [4., .1], [3., .2], [9., 1.], [12., 2.]], 4., .05),
                   ([[3., 0.], [4. - math.cos(.4), math.sin(.4)],
                     [4. - math.cos(.8), math.sin(.8)], [6., 1.], [9., 2.]], 4., .05)]
        for points, speed, timestep in invalid:
            with self.subTest(points=points, speed=speed, timestep=timestep), self.assertRaises(ControllerError):
                controller.step(points, speed, timestep)
        second = controller.step(straight(-0.2), 4.0)
        self.assertEqual(first, replay.step(straight(0.2), 4.0))
        self.assertEqual(second, replay.step(straight(-0.2), 4.0))
        controller.reset()
        self.assertEqual(first, controller.step(straight(0.2), 4.0))

    def test_bounded_smoothing_slew_pi_and_lateral_speed_policy_across_history(self):
        base = replace(ControllerConfig(), max_normalized_steer=0.4, steer_rate_per_second=0.7)
        controller = self.make(base=base, steer_smoothing_alpha=0.6)
        previous = 0.0
        for index in range(200):
            lateral = 2.0 if (index // 25) % 2 else -2.0
            result = controller.step(straight(lateral), 8.0 if index % 2 else 0.0)
            self.assertLessEqual(abs(result["steer"]), 0.4)
            self.assertLessEqual(abs(result["steer"] - previous), 0.035 + 1e-12)
            self.assertLessEqual(result["throttle"], base.throttle_max)
            self.assertLessEqual(result["brake"], base.brake_max)
            self.assertEqual(result["throttle"] * result["brake"], 0.0)
            diag = result["diagnostics"]
            self.assertLessEqual(diag["effective_target_speed_mps"] ** 2 * diag["speed_limit_curvature_per_m"],
                                 base.max_lateral_acceleration_mps2 + 1e-12)
            self.assertLessEqual(abs(diag["speed_integral_m"]), base.integral_limit_m)
            json.dumps(result, allow_nan=False)
            previous = result["steer"]

    def test_closed_loop_straight_recovery_in_independent_ackermann_plant(self):
        for lateral, heading in ((0.4, 0.04), (-0.4, -0.04)):
            controller = self.make()
            rear_x, rear_y = 0.0, lateral
            for _ in range(240):
                c, s = math.cos(heading), math.sin(heading)
                actor_x, actor_y = rear_x + 1.4 * c, rear_y + 1.4 * s
                targets = [[c * distance - s * actor_y, -s * distance - c * actor_y]
                           for distance in (3., 6., 9., 12., 15.)]
                command = controller.step(targets, 25.0 / 3.6)
                inner = abs(command["steer"]) * math.radians(70.0)
                curvature = (math.copysign(1.0 / (2.8 / math.tan(inner) + 0.8), command["steer"])
                             if inner else 0.0)
                rear_x += 25.0 / 3.6 * c * 0.05
                rear_y += 25.0 / 3.6 * s * 0.05
                heading += 25.0 / 3.6 * curvature * 0.05
            self.assertLess(abs(rear_y), 0.01)
            self.assertLess(abs(heading), 0.002)

    def test_actor_circle_recovers_with_measured_xy_offset_in_independent_ackermann_plant(self):
        for sign in (-1.0, 1.0):
            for rear_offset_y in (-0.1, 0.1):
                radius = 10.0
                rear_radius = -rear_offset_y + sign * math.sqrt(radius ** 2 - 1.4 ** 2)
                centre_x, centre_y = 12.0, 34.0 + rear_radius
                rear_x, rear_y, heading = 12.0, 34.2, 0.02 * sign
                controller = build_controller(ControllerStudyConfig(method="stanley"), geometry(rear_offset_y))
                errors = []
                for index in range(300):
                    c, s = math.cos(heading), math.sin(heading)
                    actor_x = rear_x + 1.4 * c + rear_offset_y * s
                    actor_y = rear_y + 1.4 * s - rear_offset_y * c
                    angle = math.atan2(actor_y - centre_y, actor_x - centre_x)
                    targets = []
                    for arc in (3., 6., 9., 12., 15.):
                        theta = angle + sign * arc / radius
                        dx = centre_x + radius * math.cos(theta) - actor_x
                        dy = centre_y + radius * math.sin(theta) - actor_y
                        targets.append([c * dx + s * dy, -s * dx + c * dy])
                    result = controller.step(targets, 3.0)
                    inner = abs(result["steer"]) * math.radians(70.)
                    curvature = (math.copysign(1. / (2.8 / math.tan(inner) + .8), result["steer"])
                                 if inner else 0.)
                    rear_x += 3. * c * .05
                    rear_y += 3. * s * .05
                    heading += 3. * curvature * .05
                    if index > 180:
                        errors.append(abs(math.hypot(actor_x - centre_x, actor_y - centre_y) - radius))
                with self.subTest(sign=sign, rear_offset_y=rear_offset_y):
                    self.assertLess(max(errors), 0.05)


if __name__ == "__main__":
    unittest.main()
