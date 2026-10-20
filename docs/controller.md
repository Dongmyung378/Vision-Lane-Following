# Pure Pursuit와 속도 제어

[controller.py](../carla_tools/controller.py)와 [controller_d8.json](../configs/controller_d8.json)은 같은 `5 × 2` waypoint를 privileged reference와 RGB 기반 방법의 공통 제어 입력으로 사용한다. 라벨의 actor 원점을 유지하고, Pure Pursuit의 기하 계산에 측정 rear axle offset을 사용한다. 속도 피드백은 controller 입력이며 perception 입력과 구분한다.

## Native 차량 geometry

`VehicleGeometry.from_physics_control(physics_control, vehicle_transform)`은 wheelbase, front/rear axle XY, 전륜 track, MAX native 전륜 조향 gain과 적용 steering curve를 측정한다. CARLA 0.9.16 wheel position은 world Unreal cm이므로 100으로 나눈 뒤 같은 측정 pose의 inverse transform으로 actor-local m에 변환한다. 앞/뒤 두 wheel 평균으로 axle을 구하며 임의 blueprint 추정값을 사용하지 않는다.

Python physics API는 steering curve의 interpolation tangent를 보존하지 않는다. Native runner는 owned 차량의 원본 curve를 기록하고 `[[0,1],[100,1]]`을 적용·readback하여 상수 factor를 사용한다. 임의 원본 curve에는 controller interpolation과 실제 simulator의 일치 확인이 필요하다.

## 목표와 조향

Rear axle 중심 radial lookahead는 `clamp(3.5 + 0.35 * speed_mps, 4.0, 8.0)`m다. 다섯 waypoint polyline과 lookahead 원의 교점을 보간한다. 첫 점이 이미 멀면 첫 점을, 전체 경로가 짧으면 마지막 점을 사용하며 외삽하지 않는다.

Actor 타깃이 `p=(x,y)`, actor-local rear offset이 `r=(rx,ry)`이면 현재 actor 원점과 미래 타깃이 같은 rear-axle ICR 원에 놓이도록 계산한다.

```text
rear_curvature = 2 * p_y / (|p-r|² - |r|²)
bicycle_centre_steering_rad = atan(wheelbase * rear_curvature)
inner_radius_factor = 1 - front_track * abs(rear_curvature) / 2
native_inner_steering_rad = sign(rear_curvature) * atan2(wheelbase * abs(rear_curvature), inner_radius_factor)
normalized_steer = native_inner_steering_rad / (native_front_gain_rad * steering_curve_factor)
```

곡률 0일 때 조향은 0이며 분모는 양수여야 한다. Actor 타깃을 단순히 rear 궤적으로 취급하는 식과 구분한다. 우측 y 양수는 우회전 steer 양수다. Native gain은 전륜 두 최대 조향각의 MAX이며 명령은 Ackermann 안쪽 wheel 각도로 정규화한다. 안쪽 반경이 0 이하이면 diagnostics에 남기고 actuator 범위로 제한한다.

Steer는 `[-1,1]`, 변화율은 초당 2.0으로 제한한다. Simulator 내부 응답과 tyre dynamics가 있으므로 계산 조향각을 같은 순간의 실제 wheel 각도로 해석하지 않는다.

## 곡률 속도 계획과 PI

다섯 waypoint의 연속된 세 점 조합에서 Menger curvature를 계산하고 최대 절댓값을 preview로 사용한다. Pure Pursuit rear 곡률과 preview 중 큰 곡률 `k`로 목표를 정한다.

```text
effective_target_mps = min(25 / 3.6, sqrt(2.0 / k))
# k == 0이면 nominal target 유지
```

2.0m/s²는 속도 계획 설정이며 실제 순간 횡가속도 보장이 아니다. PI는 `error=effective_target_mps-speed_mps`, `kp=0.45`, `ki=0.15`, integral ±10m를 사용한다. 양수 effort는 throttle(최대 0.65), 음수는 brake(최대 0.5)로 전달하며 동시에 양수로 적용하지 않는다. 포화 방향으로 integral을 키우는 후보는 거부하고 ±0.05m/s deadband에서는 누적을 멈춘다.

`reset()`은 integral·이전 steer·step을 초기화한다. 잘못된 입력은 상태 변경 전에 거부하며 episode 간 제어 상태를 재사용하지 않는다. 설정은 누락·중복·unknown field와 nonfinite/range 오류를 거부하고 canonical hash를 보존한다.

## API와 시간 계약

```python
from carla_tools.controller import PurePursuitController, VehicleGeometry, load_controller_config

cfg = load_controller_config()
geometry = VehicleGeometry.from_physics_control(physics_control, vehicle_transform)
controller = PurePursuitController(cfg, geometry)
command = controller.step(waypoints_ego_m, speed_mps, dt_seconds=0.05)
```

`command`는 throttle/brake/steer·handbrake/reverse와 diagnostics를 담는다. Simulator에는 control 필드만 전달한다. Diagnostics는 actor/rear 타깃·곡률·Ackermann 변환·lookahead·clipping/slew·속도 목표·PI/antiwindup와 config hash를 구분한다.

Native 시간 정책은 `frame F snapshot/label → command → frame F+1 applied readback`이다. F+1 pose를 F label에 사용하지 않는다. Frame 정합·원래 route identity·실제 적용 확인은 [주행 runner와 validator](driving.md)가 담당한다.

## 제어기 비교와 Stanley

[controller_study.py](../carla_tools/controller_study.py)는 기존 Pure Pursuit 설정과 구현을 유지하면서 비교용 설정과 Stanley 제어기를 제공한다. [controller_study.json](../configs/controller_study.json)은 기존 전체 설정을 `base`에 담고 `method`, Stanley gain·저속 완화, 조향 smoothing을 명시한다. 비교 대상마다 canonical configuration hash가 다르며 누락·중복·unknown field와 nonfinite 값을 거부한다.

```python
from dataclasses import replace
from carla_tools.controller_study import build_controller, load_study_controller_config

config = load_study_controller_config()
config = replace(config, method="stanley", stanley_cross_track_gain=1.0)
controller = build_controller(config, geometry)
command = controller.step(waypoints_ego_m, speed_mps, dt_seconds=0.05)
```

`build_controller`에 기존 `ControllerConfig`를 전달하면 기존 Pure Pursuit 객체를 만든다. 비교용 Pure Pursuit에 `steer_smoothing_alpha=1.0`을 사용하면 같은 base 설정·입력 이력에서 기존 throttle/brake/steer 명령과 일치한다. Episode마다 `reset()`하고 검증에서도 같은 설정으로 command를 독립 재계산한다.

Stanley는 측정한 front axle 위치에서 가장 가까운 경로점과 경로 접선 방향을 사용한다. Waypoint는 actor 원점의 미래 위치이므로 이를 그대로 front axle 경로로 취급하면 곡선에서 기준점 차이에 의한 오차가 생긴다. 처음 세 점이 정의하는 actor 국소 원 또는 직선을 대응하는 front axle 원·직선으로 변환한 뒤 투영한다. Actor 원의 반경이 `Ra`, actor-local rear offset이 `(rx,ry)`, 회전 부호가 `s`이면 rear 반경과 front 반경은 다음과 같다.

```text
Rrear = -ry + s * sqrt(Ra² - rx²)
Rfront = hypot(front_x - rx, front_y - ry - Rrear)
steering = heading_gain * front_path_tangent_angle
         + atan2(cross_track_gain * signed_cross_track_error,
                 speed_mps + softening_mps)
```

두 원은 같은 중심을 사용한다. 직선은 경로 방향으로 회전한 측정 front offset을 더한다. 우측 경로 오차와 우회전 조향은 양수다. 이상적인 actor 원 위에서 cross-track error가 0일 때 front-path 접선 항은 `atan(wheelbase/Rrear)`를 제공한다. 물리적으로 도달할 수 없는 원, 역방향 국소 경로와 잘못된 입력은 상태를 변경하기 전에 거부한다. 국소 원은 예측 waypoint를 이용한 근사이므로 잡음이나 급격한 곡률 변화에서의 성능은 실제 개발 주행으로 비교해야 한다.

Stanley도 bicycle steering을 같은 측정 Ackermann inner-wheel 명령으로 변환한다. 같은 preview curvature 속도 제한과 PI·antiwindup을 사용하며 조향은 actuator 범위로 제한한 뒤 `filtered = previous + alpha * (clipped - previous)`로 smoothing하고 마지막에 변화율을 제한한다. `alpha=1.0`이면 smoothing을 건너뛴다. Diagnostics는 heading/cross-track 각도, actor/front 경로 변환, smoothing·slew와 설정 hash를 남긴다. 계산 시간과 차량 제어기의 실제 동역학은 별도로 평가한다.
