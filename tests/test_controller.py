"""D8 units, native wheel measurement, sign, bounds, and recovery checks."""

import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace as Obj

from carla_tools.controller import (
    ControllerConfig, ControllerError, PurePursuitController, VehicleGeometry,
    _actor_circle_curvature, controller_config_from_dict, load_controller_config,
)


def geometry(curve=None, limit=70.0):
    return VehicleGeometry(2.8, -1.4, 1.4, limit,
                           curve or ((0.0, 1.0), (100.0, 1.0)), front_track_m=1.6)


def points(lateral=0.0):
    return [[float(x), lateral] for x in (3, 6, 9, 12, 15)]


def native_physics(yaw=0.0, translation=(12.0, 34.0, 1.0)):
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty, tz = translation
    wheels = []
    for x, y in ((1.4, -0.8), (1.4, 0.8), (-1.4, -0.8), (-1.4, 0.8)):
        wheels.append(Obj(position=Obj(x=(tx + c*x - s*y)*100.0,
                                      y=(ty + s*x + c*y)*100.0,
                                      z=(tz - 0.3)*100.0),
                          max_steer_angle=70.0 if x > 0 else 0.0))
    physics = Obj(wheels=wheels, steering_curve=[Obj(x=0.0, y=1.0), Obj(x=100.0, y=1.0)])
    matrix = [[c, s, 0.0, -c*tx-s*ty], [-s, c, 0.0, s*tx-c*ty],
              [0.0, 0.0, 1.0, -tz], [0.0, 0.0, 0.0, 1.0]]
    return physics, Obj(get_inverse_matrix=lambda: matrix)


class ControllerConfigurationTests(unittest.TestCase):
    def test_checked_in_config_matches_immutable_defaults(self):
        self.assertEqual(load_controller_config(), ControllerConfig())
        with self.assertRaises(FrozenInstanceError):
            ControllerConfig().target_speed_kmh = 30

    def test_unknown_missing_duplicate_nonfinite_and_bool_config_rejected(self):
        for key, value in (("speed_kp", True), ("brake_max", -0.1), ("speed_ki", float("nan")),
                           ("lookahead_min_m", 10), ("target_speed_kmh", 35),
                           ("fixed_delta_seconds", 0.1), ("schema_version", True),
                           ("max_lateral_acceleration_mps2", 0),
                           ("max_lateral_acceleration_mps2", float("inf")),
                           ("max_lateral_acceleration_mps2", True)):
            raw = ControllerConfig().to_dict()
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(ControllerError):
                controller_config_from_dict(raw)
        for mutation in (lambda r: r.pop("brake_max"), lambda r: r.update(extra=1)):
            raw = ControllerConfig().to_dict()
            mutation(raw)
            with self.assertRaises(ControllerError):
                controller_config_from_dict(raw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ControllerError, "duplicate-json-key"):
                load_controller_config(path)

    def test_configuration_hash_normalizes_numbers_and_detects_gain_change(self):
        raw = ControllerConfig().to_dict()
        raw["target_speed_kmh"] = 25
        raw["lookahead_min_m"] = 4
        self.assertEqual(controller_config_from_dict(raw).sha256, ControllerConfig().sha256)
        self.assertNotEqual(replace(ControllerConfig(), speed_kp=0.5).sha256, ControllerConfig().sha256)

    def test_old_controller_schema_and_policy_are_rejected(self):
        for old in (1,2,3):
            raw = ControllerConfig().to_dict()
            raw["schema_version"] = old
            with self.subTest(old=old), self.assertRaisesRegex(ControllerError, "schema-version"):
                controller_config_from_dict(raw)
            raw = ControllerConfig().to_dict()
            raw["controller_version"] = f"vlf-pure-pursuit-d8-v{old}"
            with self.subTest(old=old), self.assertRaisesRegex(ControllerError, "controller-version"):
                controller_config_from_dict(raw)


class VehicleGeometryTests(unittest.TestCase):
    def test_world_centimetres_are_converted_then_rotated_to_actor_metres(self):
        for yaw in (0.0, math.pi/2, -2.7):
            physics, transform = native_physics(yaw=yaw)
            result = VehicleGeometry.from_physics_control(physics, transform)
            with self.subTest(yaw=yaw):
                self.assertAlmostEqual(result.wheelbase_m, 2.8)
                self.assertAlmostEqual(result.rear_axle_x_m, -1.4)
                self.assertAlmostEqual(result.front_axle_x_m, 1.4)
                self.assertEqual(result.max_steer_angle_deg, 70.0)
                self.assertAlmostEqual(result.front_track_m, 1.6)
                self.assertEqual(result.steering_curve, ((0.0, 1.0), (100.0, 1.0)))

    def test_geometry_requires_measurement_pose_and_supported_axles(self):
        physics, transform = native_physics()
        with self.assertRaisesRegex(ControllerError, "requires-vehicle-transform"):
            VehicleGeometry.from_physics_control(physics)
        physics.wheels.pop()
        with self.assertRaisesRegex(ControllerError, "unsupported-native-wheel-layout"):
            VehicleGeometry.from_physics_control(physics, transform)
        physics, transform = native_physics()
        physics.wheels[0].position.x = float("nan")
        with self.assertRaises(ControllerError):
            VehicleGeometry.from_physics_control(physics, transform)
        physics, transform = native_physics()
        physics.wheels[0], physics.wheels[2] = physics.wheels[2], physics.wheels[0]
        with self.assertRaises(ControllerError):
            VehicleGeometry.from_physics_control(physics, transform)

    def test_nonrigid_and_mirrored_measurement_transforms_rejected(self):
        for bad in ([[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                    [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], None):
            physics, _ = native_physics()
            with self.subTest(bad=bad), self.assertRaises(ControllerError):
                VehicleGeometry.from_physics_control(physics, Obj(get_inverse_matrix=lambda: bad))

    def test_invalid_wheelbase_limits_and_curve_rejected(self):
        for changes in ({"wheelbase_m": 3.0}, {"rear_axle_x_m": 1.4},
                        {"max_steer_angle_deg": 0}, {"front_track_m": 0},
                        {"front_track_m": True}, {"front_track_m": float("nan")},
                        {"steering_curve": [(0,1),(0,0.8)]},
                        {"steering_curve": [(20,1),(100,1)]},
                        {"steering_curve": [(0,1),(100,0)]}):
            raw = geometry().to_dict()
            raw.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ControllerError):
                VehicleGeometry(**raw)

    def test_curve_units_are_kmh_and_source_mutation_cannot_change_geometry(self):
        curve = [[0,1], [20,0.9], [60,0.8], [120,0.7]]
        measured = geometry(curve)
        curve[0][1] = 0.1
        self.assertAlmostEqual(measured.steering_factor(25/3.6), 0.8875)
        self.assertAlmostEqual(measured.steering_factor(200/3.6), 0.7)

    def test_native_front_track_is_required_and_steering_gain_matches_physx_maximum(self):
        with self.assertRaises(TypeError):
            VehicleGeometry(2.8,-1.4,1.4,70.,((0,1),(100,1)))
        physics, transform = native_physics()
        physics.wheels[0].max_steer_angle = 60.0
        measured = VehicleGeometry.from_physics_control(physics, transform)
        self.assertEqual(measured.max_steer_angle_deg, 70.0)
        self.assertAlmostEqual(measured.front_track_m, 1.6)


class PurePursuitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ControllerConfig()
        self.controller = PurePursuitController(self.cfg, geometry())

    def test_straight_targets_produce_zero_steer_and_positive_acceleration(self):
        result = self.controller.step(points(), 0.0)
        self.assertEqual(result["steer"], 0.0)
        self.assertEqual(result["throttle"], self.cfg.throttle_max)
        self.assertEqual(result["brake"], 0.0)
        self.assertFalse(result["hand_brake"])
        self.assertFalse(result["reverse"])

    def test_rear_axle_shift_and_circle_interpolation_use_measured_offset(self):
        result = self.controller.step(points(), 25/3.6)
        diag = result["diagnostics"]
        requested = self.cfg.lookahead_base_m + self.cfg.lookahead_gain_seconds * 25/3.6
        self.assertAlmostEqual(diag["lookahead_effective_m"], requested)
        self.assertAlmostEqual(diag["target_rear_axle_xy_m"][0], requested)
        self.assertAlmostEqual(diag["target_actor_xy_m"][0], requested-1.4)
        self.assertEqual(diag["lookahead_selection"], "circle-segment-interpolation")

    def test_low_speed_uses_first_available_target_without_extrapolation(self):
        diag = self.controller.step(points(), 0.0)["diagnostics"]
        self.assertEqual(diag["lookahead_selection"], "first-available-target")
        self.assertAlmostEqual(diag["lookahead_effective_m"], 4.4)
        self.assertAlmostEqual(diag["target_actor_xy_m"][0], 3.0)
        self.assertEqual(diag["target_actor_xy_m"][1], 0.0)

    def test_right_positive_left_negative_and_symmetric(self):
        right = self.controller.step(points(0.5), 25/3.6)
        self.controller.reset()
        left = self.controller.step(points(-0.5), 25/3.6)
        self.assertGreater(right["steer"], 0.0)
        self.assertLess(left["steer"], 0.0)
        self.assertAlmostEqual(left["steer"], -right["steer"])
        self.assertAlmostEqual(left["diagnostics"]["curvature_per_m"],
                               -right["diagnostics"]["curvature_per_m"])

    def test_circle_geometry_matches_analytic_pure_pursuit_curvature(self):
        radius = 25.0
        waypoints = circle_points(radius)
        ctrl = PurePursuitController(replace(self.cfg, lookahead_base_m=0.5,
                                          lookahead_min_m=0.5, lookahead_max_m=0.5), geometry())
        diag = ctrl.step(waypoints, 25/3.6)["diagnostics"]
        # The actor circle and the rear axle circle have different radii.
        rear_radius = math.sqrt(radius**2-1.4**2)
        self.assertAlmostEqual(diag["curvature_per_m"],1/rear_radius)
        self.assertAlmostEqual(diag["steering_rad"],math.atan(2.8/rear_radius))

    def test_steering_rate_limit_bounds_first_step_and_direction_reversal(self):
        ctrl = PurePursuitController(self.cfg, geometry(limit=10))
        right = [[float(x), float(y)] for x,y in ((0,4),(1,5),(2,6),(3,7),(4,8))]
        previous = 0.0
        for _ in range(20):
            result = ctrl.step(right, 25/3.6)
            self.assertLessEqual(abs(result["steer"]-previous), 0.10000000001)
            self.assertLessEqual(abs(result["steer"]), 1.0)
            previous = result["steer"]
        self.assertEqual(previous, 1.0)
        result = ctrl.step([[x,-y] for x,y in right], 25/3.6)
        self.assertAlmostEqual(result["steer"], 0.9)
        self.assertTrue(result["diagnostics"]["steer_rate_limited"])

    def test_speed_curve_factor_changes_steering_command_normalization(self):
        nominal = self.controller.step(points(0.2), 25/3.6)
        reduced = PurePursuitController(self.cfg, geometry(((0,0.5),(100,0.5))))
        limited = reduced.step(points(0.2), 25/3.6)
        self.assertAlmostEqual(limited["steer"], 2*nominal["steer"])

    def test_overspeed_brakes_without_throttle_and_low_speed_never_reverses(self):
        overspeed = self.controller.step(points(), 35/3.6)
        self.assertEqual(overspeed["throttle"], 0.0)
        self.assertEqual(overspeed["brake"], self.cfg.brake_max)
        slow = self.controller.step(points(), 0.0)
        self.assertGreater(slow["throttle"], 0.0)
        self.assertFalse(slow["reverse"])

    def test_actor_straight_path_stays_straight_with_nonzero_lateral_axle_offset(self):
        measured = replace(geometry(), rear_axle_y_m=0.1, front_axle_y_m=0.1)
        controller = PurePursuitController(self.cfg, measured)
        result = controller.step(points(0.0), 25/3.6)
        self.assertEqual(result["steer"], 0.0)
        self.assertEqual(result["diagnostics"]["target_rear_axle_xy_m"][1], -0.1)
        self.assertEqual(result["diagnostics"]["target_actor_xy_m"][1], 0.0)

    def test_saturation_antiwindup_prevents_stalled_vehicle_integral_buildup(self):
        for _ in range(300):
            result = self.controller.step(points(), 0.0)
        self.assertEqual(result["diagnostics"]["speed_integral_m"], 0.0)
        self.assertTrue(result["diagnostics"]["speed_pi_antiwindup"])
        overspeed = self.controller.step(points(), 30/3.6)
        self.assertGreater(overspeed["brake"], 0.0)
        self.assertEqual(overspeed["throttle"], 0.0)

    def test_integral_accumulates_only_unsaturated_error_and_reset_is_exact(self):
        initial = self.controller.step(points(0.2), 24/3.6)
        for _ in range(30):
            final = self.controller.step(points(0.2), 24/3.6)
        self.assertGreater(final["throttle"], initial["throttle"])
        self.controller.reset()
        self.assertEqual(self.controller.step(points(0.2), 24/3.6), initial)

    def test_malformed_nonfinite_reversed_and_wrong_timestep_inputs_are_rejected(self):
        bad = [points()[:4], [[1,2,3]]*5, points()[::-1], points(), points(),
               [[-1.4,0],[3,0],[6,0],[9,0],[12,0]]]
        bad[3][0][0] = float("nan")
        bad[4][0][1] = True
        for targets in bad:
            with self.subTest(targets=targets), self.assertRaises(ControllerError):
                self.controller.step(targets, 25/3.6)
        for speed, timestep in ((-1,0.05),(True,0.05),(float("inf"),0.05),(0,0),(0,0.1)):
            with self.subTest(speed=speed, timestep=timestep), self.assertRaises(ControllerError):
                self.controller.step(points(), speed, timestep)

    def test_rejected_step_does_not_mutate_pi_or_steering_state(self):
        first = self.controller.step(points(0.3), 24/3.6)
        with self.assertRaises(ControllerError):
            self.controller.step(points(float("nan")), 0.0)
        second = self.controller.step(points(0.3), 24/3.6)
        clean = PurePursuitController(self.cfg, geometry())
        self.assertEqual(clean.step(points(0.3),24/3.6), first)
        self.assertEqual(clean.step(points(0.3),24/3.6), second)

    def test_small_lateral_and_yaw_error_recovers_in_independent_bicycle_plant(self):
        for lateral, yaw in ((0.4,math.radians(3)), (-0.4,-math.radians(3))):
            controller = PurePursuitController(self.cfg, geometry())
            rear_x, rear_y, heading = 0.0, lateral, yaw
            for _ in range(180):
                actor_x = rear_x+1.4*math.cos(heading)
                actor_y = rear_y+1.4*math.sin(heading)
                # Ground-truth straight target lane remains fixed at world y=0;
                # it is not replaced with the vehicle's displaced lane.
                targets = []
                for future in (3,6,9,12,15):
                    dx, dy = actor_x+future-actor_x, -actor_y
                    targets.append([math.cos(heading)*dx+math.sin(heading)*dy,
                                    -math.sin(heading)*dx+math.cos(heading)*dy])
                command = controller.step(targets, 25/3.6)
                # Independent wheel-radius model: inner wheel radius L/tan(i),
                # centre radius is half a measured front track farther away.
                inner = command["steer"]*math.radians(70)
                curvature = (math.copysign(1.0 / (2.8/math.tan(abs(inner))+0.8),inner)
                             if inner != 0.0 else 0.0)
                rear_x += 25/3.6*math.cos(heading)*0.05
                rear_y += 25/3.6*math.sin(heading)*0.05
                heading += 25/3.6*curvature*0.05
            with self.subTest(lateral=lateral):
                self.assertLess(abs(rear_y), 0.02)
                self.assertLess(abs(heading), math.radians(0.5))

    def test_ackermann_inner_angle_recreates_desired_bicycle_curvature(self):
        for lateral in (-1.0,1.0):
            diag = self.controller.step(points(lateral), 4.0)["diagnostics"]
            inner = abs(diag["native_inner_steering_rad"])
            inner_radius = 2.8/math.tan(inner)
            outer_angle = math.atan(2.8/(inner_radius+1.6))
            center_radius = (2.8/math.tan(inner) + 2.8/math.tan(outer_angle))/2.0
            with self.subTest(lateral=lateral):
                self.assertAlmostEqual(1.0/center_radius, abs(diag["curvature_per_m"]))
                self.assertAlmostEqual(diag["steer_unclipped"]*math.radians(70),
                                       diag["native_inner_steering_rad"])
                self.assertGreater(inner, abs(diag["steering_rad"]))

    def test_measured_wider_front_track_requires_larger_inner_wheel_angle(self):
        narrow = PurePursuitController(self.cfg, replace(geometry(),front_track_m=1.0))
        wide = PurePursuitController(self.cfg, replace(geometry(),front_track_m=2.0))
        nd = narrow.step(points(1.0),4.0)["diagnostics"]
        wd = wide.step(points(1.0),4.0)["diagnostics"]
        self.assertEqual(nd["curvature_per_m"],wd["curvature_per_m"])
        self.assertEqual(nd["steering_rad"],wd["steering_rad"])
        self.assertGreater(wd["native_inner_steering_rad"],nd["native_inner_steering_rad"])

    def test_infeasible_tight_target_is_clipped_without_reversing_steer_sign(self):
        targets = [[0.01,0.1],[0.02,0.2],[0.03,0.3],[0.04,0.4],[0.05,0.5]]
        result = self.controller.step(targets,0.0)
        diag = result["diagnostics"]
        self.assertTrue(diag["ackermann_inner_radius_nonpositive"])
        self.assertGreater(diag["native_inner_steering_rad"],math.pi/2)
        self.assertEqual(diag["steer_clipped"],1.0)
        self.assertGreater(result["steer"],0.0)

    def test_nonforward_actor_circle_denominator_rejected_without_state_change(self):
        initial = self.controller.step(points(),5.0)
        bad = [[-1.38,0.1],[-1.37,0.2],[-1.36,0.3],[-1.35,0.4],[-1.34,0.5]]
        with self.assertRaisesRegex(ControllerError,"actor-circle-target-not-forward"):
            self.controller.step(bad,5.0)
        second = self.controller.step(points(),5.0)
        replay = PurePursuitController(self.cfg,geometry())
        self.assertEqual(replay.step(points(),5.0),initial)
        self.assertEqual(replay.step(points(),5.0),second)

    def test_zero_rear_offset_reduces_to_standard_pure_pursuit(self):
        for target in ([4.,1.],[4.,-1.],[6.,0.]):
            curvature, denominator = _actor_circle_curvature(target,[0.,0.])
            self.assertAlmostEqual(curvature,2*target[1]/sum(x*x for x in target))
            self.assertAlmostEqual(denominator,sum(x*x for x in target))

    def test_actor_circle_matches_measured_xy_offset_for_both_turn_directions(self):
        for rear_y in (-0.1,0.1):
            measured = replace(geometry(),rear_axle_y_m=rear_y,front_axle_y_m=rear_y)
            cfg = replace(self.cfg,lookahead_base_m=0.5,lookahead_min_m=0.5,lookahead_max_m=0.5)
            for sign in (-1.,1.):
                radius=10.
                rear_radius=-rear_y+sign*math.sqrt(radius**2-1.4**2)
                targets=circle_points(radius,sign,rear_y)
                diag=PurePursuitController(cfg,measured).step(targets,3.)["diagnostics"]
                with self.subTest(rear_y=rear_y,sign=sign):
                    self.assertAlmostEqual(diag["curvature_per_m"],1/rear_radius)
                    for x,y in targets:
                        self.assertAlmostEqual((x+1.4)**2+(y-rear_y-rear_radius)**2,radius**2)

    def test_actor_circle_is_followed_by_independent_translated_ackermann_plant(self):
        measured=replace(geometry(),rear_axle_y_m=0.1,front_axle_y_m=0.1)
        cfg=replace(self.cfg,lookahead_base_m=0.5,lookahead_min_m=0.5,lookahead_max_m=0.5)
        for sign in (-1.,1.):
            actor_radius=10.
            rear_radius=-0.1+sign*math.sqrt(actor_radius**2-1.4**2)
            origin_x,origin_y=12.,34.
            center_x,center_y=origin_x,origin_y+rear_radius
            rear_x,rear_y,heading=origin_x,origin_y,0.
            controller=PurePursuitController(cfg,measured)
            errors=[]
            for index in range(240):
                c,s=math.cos(heading),math.sin(heading)
                actor_x=rear_x+1.4*c+0.1*s
                actor_y=rear_y+1.4*s-0.1*c
                angle=math.atan2(actor_y-center_y,actor_x-center_x)
                targets=[]
                for arc in (3,6,9,12,15):
                    theta=angle+sign*arc/actor_radius
                    dx=center_x+actor_radius*math.cos(theta)-actor_x
                    dy=center_y+actor_radius*math.sin(theta)-actor_y
                    targets.append([c*dx+s*dy,-s*dx+c*dy])
                command=controller.step(targets,3.0)
                inner=abs(command["steer"])*math.radians(70)
                # Native inner-wheel radius, then an independently measured
                # half-track gives the rear axle's centre turning radius.
                curvature=(math.copysign(1/(2.8/math.tan(inner)+0.8),command["steer"])
                           if inner else 0.)
                rear_x+=3.*c*.05
                rear_y+=3.*s*.05
                heading+=3.*curvature*.05
                if index>120:
                    errors.append(abs(math.hypot(actor_x-center_x,actor_y-center_y)-actor_radius))
            with self.subTest(sign=sign):
                self.assertLess(max(errors),0.03)


def circle_points(radius, sign=1.0, rear_y=0.0):
    # Desired ACTOR trajectory: current actor at (a,b), rear ICR at (0,R).
    # Offset XY and its physical circle radius are independently constructed.
    a,b=1.4,-rear_y
    rear_radius=b+sign*math.sqrt(radius**2-a*a)
    result=[]
    for arc in (3,6,9,12,15):
        theta=sign*arc/radius
        result.append([a*(math.cos(theta)-1)+(rear_radius-b)*math.sin(theta),
                       a*math.sin(theta)+(b-rear_radius)*(math.cos(theta)-1)])
    return result


class CurvatureSpeedPolicyTests(unittest.TestCase):
    def make_controller(self):
        return PurePursuitController(ControllerConfig(), geometry())

    def test_straight_keeps_25_kmh_and_emits_finite_json_diagnostics(self):
        diag = self.make_controller().step(points(), 5.0)["diagnostics"]
        self.assertEqual(diag["preview_curvatures_per_m"], [0.0,0.0,0.0])
        self.assertEqual(diag["effective_target_speed_mps"], 25/3.6)
        self.assertFalse(diag["curvature_speed_limited"])
        self.assertIsNone(diag["curvature_speed_cap_mps"])
        json.dumps(diag, allow_nan=False)

    def test_three_circle_triples_match_analytic_curvature_and_speed_limit(self):
        for radius in (10.0, 13.6):
            # Select an actual circle point so sparse chord interpolation does
            # not add its own tighter commanded-curvature limit to this test.
            cfg = replace(ControllerConfig(), lookahead_base_m=0.5,
                          lookahead_min_m=0.5, lookahead_max_m=0.5)
            diag = PurePursuitController(cfg, geometry()).step(circle_points(radius), 4.0)["diagnostics"]
            with self.subTest(radius=radius):
                for curvature in diag["preview_curvatures_per_m"]:
                    self.assertAlmostEqual(curvature, 1.0/radius)
                self.assertAlmostEqual(diag["effective_target_speed_mps"],
                                       math.sqrt(2.0*math.sqrt(radius**2-1.4**2)))
                self.assertTrue(diag["curvature_speed_limited"])
                self.assertLessEqual(diag["effective_target_speed_kmh"], 25.0)

    def test_curvature_preview_is_invariant_to_ego_yaw_translation_and_mirroring(self):
        source = circle_points(13.6)
        expected = 1/13.6
        for yaw, translation, mirror in ((0.0,(0.0,0.0),1),
                                         (math.radians(5),(0.3,-0.3),1),
                                         (math.radians(-5),(-0.3,0.3),-1)):
            c, s = math.cos(yaw), math.sin(yaw)
            transformed = [[c*x-s*y+translation[0], mirror*(s*x+c*y+translation[1])]
                           for x,y in source]
            diag = self.make_controller().step(transformed, 4.0)["diagnostics"]
            with self.subTest(yaw=yaw, mirror=mirror):
                for curvature in diag["preview_curvatures_per_m"]:
                    self.assertAlmostEqual(curvature, expected)

    def test_future_bend_limits_speed_before_current_steering_changes(self):
        waypoints = [[3.,0.],[6.,0.],[9.,0.],[11.,1.],[12.,3.]]
        result = self.make_controller().step(waypoints, 5.0)
        diag = result["diagnostics"]
        self.assertEqual(result["steer"], 0.0)
        self.assertEqual(diag["curvature_per_m"], 0.0)
        self.assertGreater(diag["preview_curvature_per_m"], 0.2)
        self.assertTrue(diag["curvature_speed_limited"])
        self.assertGreater(result["brake"], 0.0)
        self.assertEqual(result["throttle"], 0.0)

    def test_future_bend_left_and_right_have_identical_speed_targets(self):
        right = [[3.,0.],[6.,0.],[9.,0.],[11.,1.],[12.,3.]]
        left = [[x,-y] for x,y in right]
        rd = self.make_controller().step(right, 5.0)["diagnostics"]
        ld = self.make_controller().step(left, 5.0)["diagnostics"]
        self.assertEqual(rd["effective_target_speed_mps"], ld["effective_target_speed_mps"])
        self.assertEqual(rd["preview_curvatures_per_m"], ld["preview_curvatures_per_m"])

    def test_pi_error_tracks_effective_curvature_target_and_antiwindup_still_operates(self):
        controller = self.make_controller()
        waypoints = circle_points(10.0)
        for _ in range(100):
            stopped = controller.step(waypoints, 0.0)
        self.assertEqual(stopped["diagnostics"]["speed_integral_m"], 0.0)
        self.assertTrue(stopped["diagnostics"]["speed_pi_antiwindup"])
        result = controller.step(waypoints, 6.0)
        diag = result["diagnostics"]
        self.assertAlmostEqual(diag["speed_error_mps"], diag["effective_target_speed_mps"]-6.0)
        self.assertEqual(result["throttle"], 0.0)
        self.assertGreater(result["brake"], 0.0)

    def test_turn_command_caps_lateral_acceleration_even_for_collinear_offset_targets(self):
        controller = self.make_controller()
        result = controller.step(points(3.0), 4.0)
        diag = result["diagnostics"]
        self.assertEqual(diag["preview_curvatures_per_m"], [0.0,0.0,0.0])
        self.assertGreater(diag["speed_limit_curvature_per_m"], 0.0)
        self.assertLessEqual(diag["effective_target_speed_mps"]**2 * diag["speed_limit_curvature_per_m"],
                             2.0+1e-12)

    def test_degenerate_preview_input_is_rejected_without_controller_state_change(self):
        controller = self.make_controller()
        first = controller.step(points(), 5.0)
        tiny = [[float(index)*0.000002,0.] for index in range(1,6)]
        with self.assertRaisesRegex(ControllerError, "preview-points-too-close"):
            controller.step(tiny, 5.0)
        second = controller.step(points(), 5.0)
        replay = self.make_controller()
        self.assertEqual(replay.step(points(),5.0), first)
        self.assertEqual(replay.step(points(),5.0), second)


if __name__ == "__main__":
    unittest.main()
