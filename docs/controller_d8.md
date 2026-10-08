# D8 Pure Pursuit와 25km/h 속도 제어

작성일: 2026-10-08 (Asia/Seoul). 구현은 [controller.py](../carla_tools/controller.py), 고정 설정은 [controller_d8.json](../configs/controller_d8.json), CPU 검증은 [test_controller.py](../tests/test_controller.py)에 있다. **Controller v4/schema 4의 39개 CPU test와 Kaggle version 11의 전체 15조건 native 주행 검증은 PASS했다.** 실제 주행 결과와 version 9·10의 실패·수정 근거는 아래 Native 결과 항목에 보존한다.

D8은 [로드맵](ROADMAP.md)의 공통 controller를 구현한다. D9와 D10에서는 정답 경로를 입력하는 privileged reference로 직선·좌우 곡선과 작은 초기 편차를 확인한다. RGB를 입력하는 B0/M1의 주행이나 최종 성능 검증은 이후 단계다. 모델의 경로 예측 입력은 RGB이고, 공통 controller의 속도 계산과 저수준 속도 제어에는 차량 속도 피드백을 사용한다.

## Actor 원점과 rear axle 제어 기준점

[D5 계약](DATA_CONTRACT_D5.md)의 waypoint는 actor transform 원점 기준 `5 × 2` 좌표이며, x는 전방, y는 우측, 단위는 m다. 3/6/9/12/15m는 고정 목표 차선 centerline의 arc horizon이다. Controller를 연결하면서 저장 라벨의 원점을 camera나 axle로 바꾸지 않는다.

Pure Pursuit 계산에만 측정한 rear axle 좌표를 사용한다. Actor 좌표의 타깃이 `(x_actor, y_actor)`, rear axle이 `(x_rear, y_rear)`이면 다음과 같다. Rear 좌표는 lookahead 거리와 타깃 선택에 사용하고, 타깃이 나타내는 궤적은 actor 원점의 궤적으로 유지한다.

```text
x_control = x_actor - x_rear
y_control = y_actor - y_rear
wheelbase = front_axle_x - rear_axle_x
```

`VehicleGeometry`는 wheelbase, front/rear axle XY, **측정한 전륜 track 폭 `front_track_m`**, native 전륜 steering gain과 적용 steering curve를 보존한다. `front_track_m`은 v3의 필수 입력이며 앞 두 wheel의 actor-local y 차이로 계산한다. 임의의 wheelbase·track 폭이나 blueprint 이름에 따른 추정값을 넣지 않는다. 네 바퀴의 앞뒤·좌우 배치, axle 간 일관성, track 폭과 물리적인 범위를 검사한다.

CARLA 0.9.16의 `WheelPhysicsControl.position`은 **world Unreal cm**다. Native 측정에서는 physics가 활성화된 차량의 warmup 이후 바퀴 좌표와 해당 측정 pose를 읽는다. 각 wheel world XYZ를 100으로 나누고, vehicle transform의 inverse matrix로 actor-local m에 변환한다. 앞 두 wheel의 평균을 front axle, 뒤 두 wheel의 평균을 rear axle로 계산한다. Translation 또는 yaw가 있는 차량의 wheel world x 차이를 그대로 actor offset으로 기록하지 않는다. World wheel location의 반환 방식은 [고정 버전 simulator 소스](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp#L433-L457), RPC의 position 전달은 [WheelPhysicsControl](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/rpc/WheelPhysicsControl.h#L64-L87)에 근거한다. Wheel 순서는 [0.9.16 API](https://carla.readthedocs.io/en/0.9.16/python_api/#carla.VehiclePhysicsControl)의 front-left, front-right, rear-left, rear-right를 사용한다.

## Lookahead와 조향 명령

목표점은 현재 속도로 계산한 rear-axle radial lookahead를 사용한다.

```text
requested_lookahead_m = clamp(3.5 + 0.35 × speed_mps, 4.0, 8.0)
```

25km/h에서 요청 거리는 약 5.93m다. 다섯 점을 rear axle 기준으로 이동한 뒤, 이 반경의 원과 타깃 polyline segment의 교점을 보간한다. 첫 타깃이 이미 요청 반경보다 멀면 첫 타깃을 사용하고, 전체 타깃이 짧으면 마지막 타깃을 사용한다. 관측되지 않은 경로를 외삽하지 않으며 요청 거리와 실제 선택 거리를 각각 기록한다. 입력 타깃의 수·shape·유한 값·전방 순서를 검사하고 rear axle 뒤의 타깃은 거부한다.

선택한 **actor 원점 타깃**이 `p = (x, y)`, 측정 rear offset이 `r = (rx, ry)`라고 하자. Rear wheel의 횡방향 속도를 0으로 두는 bicycle geometry의 순간 회전 중심(ICR)은 actor 좌표에서 `C = (rx, ry + R)`이며 `R`은 rear axle의 signed 회전 반경이다. Actor 원점과 미래 actor 타깃이 같은 원 위에 있어야 하므로 `|p - C|² = |C|²`다. 이를 전개하면 다음 식을 얻는다.

```text
|p|² - 2 × p·r - 2 × y × R = 0
R = (|p - r|² - |r|²) / (2 × y)
rear_curvature = 2 × y / (|p - r|² - |r|²)
```

V4는 이 off-axle actor 궤적 식을 사용한다. 기존 `2 × target_rear_y / |target_rear|²`는 입력 타깃이 rear axle 자체의 궤적일 때의 식이다. D5의 actor 원점 타깃에는 rear offset 크기의 제곱을 분모에서 빼고, 분자는 **actor y**를 사용해야 한다. `r=(0,0)`이면 일반 Pure Pursuit 식과 같고 actor y가 0이면 직선 곡률 0이다. 저장 waypoint, horizon, camera·actor 좌표 계약과 controller의 다섯 점 입력은 그대로다.

측정 wheelbase가 `L`, 전륜 track 폭이 `T`이면 rear 곡률에서 가상 bicycle 중심각을 계산한 뒤 native 안쪽 전륜 조향각으로 변환한다.

```text
target_rear = target_actor - measured_rear_offset
curvature = 2 × target_actor_y / (target_rear_norm² - measured_rear_offset_norm²)
bicycle_centre_steering_rad = atan(L × curvature)
inner_radius_factor = 1 - T × abs(curvature) / 2
native_inner_steering_rad = sign(curvature) × atan2(L × abs(curvature), inner_radius_factor)
# curvature == 0이면 native_inner_steering_rad = 0
native_front_gain_rad = radians(max(front_left.max_steer_angle, front_right.max_steer_angle))
effective_max_steering_rad = native_front_gain_rad × steering_curve_factor
normalized_steer = native_inner_steering_rad / effective_max_steering_rad
```

y 우측 양수는 우회전 steer 양수로 대응한다. 반경 `R = 1 / abs(curvature)`에서 안쪽 wheel의 회전 반경은 `R - T/2`다. 양수 반경 영역에서는 위 식이 signed `atan(L × curvature / (1 - T × abs(curvature)/2))`와 같다. `atan2`는 요청이 너무 급해 안쪽 반경이 0 이하가 되어도 조향 부호를 유지한다. 이 경우 `ackermann_inner_radius_nonpositive=true`를 기록하고 명령을 native actuator 범위로 제한하므로, 요청한 곡률을 실현했다고 해석하지 않는다.

[PhysX 3.4의 4W 조향 구현](https://github.com/NVIDIAGameWorks/PhysX-3.4/blob/master/PhysX_3.4/Source/PhysXVehicle/src/PxVehicleUpdate.cpp#L1646-L1684)은 전륜 두 최대 조향각의 **MAX**를 gain으로 사용하고 `normalized steer × gain`을 안쪽 wheel 각으로 전달한다. 바깥 wheel은 Ackermann geometry와 accuracy에 따라 계산한다. V4는 v3에서 수정한 이 입력 계약을 유지하며, 기존 중심각을 그대로 전달하던 v2와 구분한다. 이번 Tesla native wheel pair는 안쪽 각과 완전 Ackermann 바깥 각의 관계에 일치했다. 다른 차량이나 Ackermann 설정에는 native wheel readback의 재확인이 필요하다.

명령은 `[-1, 1]` 범위와 초당 2.0의 normalized steer 변화율로 제한한다. 0.05초 tick에서 steer 변화는 최대 0.1이다. Simulator 내부 steering 응답과 tyre dynamics가 있으므로 계산한 각이 같은 순간의 실제 wheel angle이라고 해석하지 않는다.

설정된 steering curve의 speed 단위는 km/h이며 controller에는 m/s를 입력한다. CARLA 고정 버전 NW 구현은 [steering curve key를 km/h에서 cm/s로 변환](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/WheeledVehicleMovementComponentNW.cpp#L269-L283)한다. 이 NW 소스만으로 Tesla 4W의 모든 내부 interpolation 동작을 확인했다고 주장하지 않는다.

Python physics API는 curve의 `(speed, factor)`를 전달하지만 Unreal의 interpolation tangent를 보존하지 않는다. [Curve serialization과 재구성 소스](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/rpc/VehiclePhysicsControl.h#L166-L170)는 이를 확인하는 근거다. Native D9/D10 정책은 소유 차량의 기존 curve를 기록한 뒤 **`[[0, 1], [100, 1]]`을 적용하고 readback**한다. 이 상수 curve는 factor가 1이어서 숨겨진 tangent에 따른 정규화 추정을 피한다. 공통 controller와 비교 방법에 같은 적용 정책을 사용하고 변경 사실을 metadata에 남긴다. Arbitrary 원본 curve에는 controller의 piecewise-linear interpolation과 실제 simulator curve의 일치 확인이 별도로 필요하다.

## 속도 PI, 상태와 적용 시점

Nominal 속도 상한은 25km/h, timestep은 0.05초로 고정한다. Controller v4는 v2에서 도입한 곡률 속도 계획을 유지한다. 같은 다섯 waypoint에서 연속된 세 점씩 세 조합의 Menger curvature를 계산한다. `a`, `b`, `c`가 세 점이면 `κ = 2 × cross(b−a, c−a) / (|b−a| × |c−b| × |c−a|)`이며 collinear 조합의 곡률은 0이다. 절댓값의 최댓값을 곡선 preview로 사용하고 Pure Pursuit가 요청한 rear 곡률과 함께 큰 쪽을 속도 계획에 사용한다. 좌표의 translation·yaw나 좌우 반전은 preview 곡률을 바꾸지 않는다.

```text
speed_planning_curvature = max(max(abs(three_waypoint_curvatures)), abs(pure_pursuit_curvature))
effective_target_mps = min(25/3.6, sqrt(2.0 / speed_planning_curvature))
# Zero curvature keeps the nominal target.
```

2.0m/s²는 속도 계획에 쓰는 lateral acceleration 설정이며 실제 순간 가속도의 측정 보장은 아니다. 곡선이 미래 15m 안에 들어오면 진입 전부터 속도를 줄이고, 추가 지도 정보를 controller에 공급하지 않는다. 이 정책은 GT·B0·M1 모두 같은 waypoint 입력에 적용한다. Version 8에서 +0.3m 좌곡선 조건의 복귀 유지가 0.95초에 그친 실제 실패를 보존하고 도입했으며 복귀·차체·이벤트 threshold는 유지한다.

PI error는 `effective_target_mps - speed_mps`이고, `kp=0.45`, `ki=0.15`다. 양수 effort는 throttle, 음수 effort는 brake이며 둘을 동시에 양수로 적용하지 않는다. Throttle 최대는 0.65, brake 최대는 0.5다. Integral의 단위는 m이고 범위는 ±10m다.

포화 중 같은 방향 error가 integral을 더 키우려 하면 candidate integral을 채택하지 않는 conditional integration을 사용한다. 속도 error가 ±0.05m/s 안이면 integral 누적을 멈추고 proportional term은 유지한다. 이는 정지한 차량에 장시간 throttle을 요구한 뒤 과속에도 throttle이 남는 integral windup을 줄인다. Hand brake와 reverse는 주행 명령에서 false다.

`reset()`은 PI integral, 이전 steer와 step index를 초기화한다. 각 episode에서 새 controller를 만들거나 reset하고, 이전 episode의 상태를 재사용하지 않는다. Invalid input은 PI나 steer 상태를 변경하기 전에 거부한다. 설정 JSON은 누락·unknown·duplicate field, bool 숫자, NaN/Infinity와 범위 위반을 거부하며, canonical SHA-256을 진단에 기록한다.

Moving runner의 시간 정책은 **frame F의 snapshot과 label을 입력하여 계산한 명령을 다음 tick에 적용**하는 것이다. `frame F → label F → command → frame F+1`을 연결하고, F+1에서 관찰한 자세를 F의 waypoint 생성에 사용하지 않는다. Controller 모듈은 frame ID를 입력받지 않으므로 frame 정합·route identity·실제 적용 확인은 native runner와 별도 validator의 책임이다. Curve 또는 recovery에서도 원래 고정 목표 route를 사용하며 이동한 차량의 nearest lane으로 목표를 바꾸지 않는다.

## CPU 검증과 사용 API

V4의 **39개 CPU test는 PASS**했다. Checked-in config, frozen 상태와 config hash, translated/rotated native world-cm wheel 변환, axle/rigid transform/curve 오류 거부, 직선·좌우 조향 부호, lookahead 보간, steering factor와 slew bound, 과속 brake, PI antiwindup, reset 및 오류 입력 시 state 보존을 확인한다. V2의 곡률 preview·진입 전 감속·effective target 연결과 v3의 필수 front track·MAX gain·Ackermann 입력 검증도 유지한다. V4에서는 actor와 rear 원의 반경 구분, nonzero rear XY offset의 양방향 actor 원 궤적, lateral rear offset이 있어도 actor y=0이면 조향 0, offset 0에서 일반식으로의 환원, 분모 오류 시 state 보존과 독립 Ackermann plant의 actor 원 추종을 확인한다. Synthetic plant의 복귀는 native CARLA의 주행 성공 기록을 대신하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_controller.py -v
```

```python
from carla_tools.controller import (
    PurePursuitController, VehicleGeometry, load_controller_config,
)

cfg = load_controller_config()
# Native runner: warmup 후 stationary 측정 pose와 physics control을 전달한다.
geometry = VehicleGeometry.from_physics_control(physics_control, vehicle_transform)
controller = PurePursuitController(cfg, geometry)
command = controller.step(waypoints_ego_m, speed_mps, dt_seconds=0.05)
```

`command`에는 throttle/brake/steer, hand_brake/reverse, diagnostics가 있다. 실제 simulator에는 diagnostics를 제외한 control 값만 전달한다. Diagnostics는 config hash, step/timestep, 선택 타깃과 원점, lookahead, unclipped/clipped steer와 rate-limit 여부, speed error/integral/PI effort와 antiwindup/포화 여부를 보존하며 다음 계산도 구분한다.

| Diagnostics 필드 | 의미 |
| --- | --- |
| `target_actor_xy_m`, `target_rear_axle_xy_m` | 같은 선택 타깃의 actor·rear 원점 좌표 |
| `curvature_geometry`, `actor_circle_denominator_m2` | `actor-origin-path-circle-with-measured-rear-offset`과 `|p−r|²−|r|²` 분모 |
| `curvature_per_m`, `steering_rad` | Actor 궤적 원에서 유도한 rear 곡률과 가상 bicycle 중심각 |
| `steering_normalization` | `ackermann-inner-front-wheel-angle` |
| `front_track_m`, `native_inner_steering_rad`, `native_inner_steering_deg` | 측정 전륜 track과 변환한 안쪽 wheel 각 |
| `ackermann_inner_radius_nonpositive` | 요청 곡률의 안쪽 회전 반경이 0 이하인지 여부 |
| `steering_curve_factor`, `effective_max_steering_rad` | 적용 curve factor와 MAX native gain을 반영한 정규화 분모 |
| `preview_curvatures_per_m`, `preview_curvature_per_m`, `speed_limit_curvature_per_m` | 세 Menger 곡률, preview 최댓값과 속도 계획용 곡률 |
| `nominal_target_speed_mps`, `curvature_speed_cap_mps`, `effective_target_speed_mps`, `effective_target_speed_kmh` | 명목 상한, 곡률 상한과 PI가 사용하는 목표 속도 |
| `max_lateral_acceleration_mps2`, `curvature_speed_limited` | 계획 설정과 곡률로 목표 속도를 줄였는지 여부 |

직선의 `curvature_speed_cap_mps`는 JSON의 `null`이며 Infinity를 저장하지 않는다. Schema 4와 `vlf-pure-pursuit-d8-v4`는 이전 정책의 설정을 거부한다. Native geometry metadata와 전체 command diagnostics는 오프라인에서 재측정·재계산하여 비교한다.

## Native 결과

**현재 상태: version 11 PASS.** Controller v4/schema 4를 2026-10-08 13:59 UTC에 기존 Kaggle notebook `dongmyungpark/vlf-carla-rgb-smoke-20261005`의 같은 ID와 제목을 유지하여 T4에 제출했다. 실제 15조건의 주행·정지와 다운로드 후 CPU·손상·지도 검증이 모두 통과했다. 새 notebook이나 로컬 GPU 실행은 없었다.

Version 9에서는 직선 5조건과 좌곡선 5조건이 통과했으나 우곡선 center는 113번째 기록, reference arc 17.302m에서 **차체 여유 0.13891m로 고정 기준 0.15m를 충족하지 못해 FAILED**가 되었다. 충돌·차선 침범 이벤트는 0이었다. 같은 Town01 원본 XODR의 독립 native Map 질의에서도 실제 차체 corner 여유가 0.13953m였으며 모든 corner는 원래 Driving lane 안에 있었다. 차선을 벗어나기 전에 요구 안전 여유가 부족해 중단한 기록이다. [Version 9 실패·원본 trace 표본·Map 교차 검증·수정 근거](evidence/carla_driving_d8_d10_2026-10-08_ackermann_failure.json)를 보존한다.

이 기록의 wheelbase는 3.00465m, front track은 1.66684m, native front gain은 약 70°였다. Frame 5163에서 계산하여 5164에 적용한 steer 0.33936에 gain을 곱하면 23.75496°로 실제 안쪽 wheel 각과 일치했다. 실제 좌우 wheel 각에서 복원한 중심 곡률은 0.13054/m로, 이전 frame의 요청 곡률 0.14648/m보다 약 10.9% 작았다. V2가 bicycle 중심각을 native 안쪽 wheel 입력으로 전달하던 계약 오류를 v3에서 수정했다.

Version 10의 v3은 우곡선 center에서 122개 주행 기록으로 reference arc 19.074m에 도달했고 주행 최소 차체 여유는 0.16609m였다. 이후 service brake를 적용하고 마지막 실제 조향을 유지한 정지 3tick의 여유가 0.15889→0.15070→0.14384m로 줄어 **정지 중 FAILED**가 되었다. 독립 native Map의 마지막 차체 여유도 0.14530m로 0.15m에 미달했다. 이때 충돌·차선 침범 이벤트는 0이었고 원래 차선 안에 있었지만 안전 여유 기준을 충족하지 못했다.

Terminal actor pose의 목표점을 rear 원점으로 이동한 뒤 기존 rear-target 식으로 계산한 곡률은 0.13444/m였다. V4의 actor 궤적 ICR 식은 같은 타깃·측정 geometry에서 약 0.14659/m로 계산되어 약 9% 차이가 난다. 곡선에서는 rear axle과 actor의 궤적 반경이 다르므로 actor 타깃을 rear 궤적으로 취급한 영향이 나타난다. V4는 이 계산을 수정하며 마지막 적용 조향을 유지하는 제동 정책, 원래 route, 회복 조건, 0.15m 차체 여유, 이벤트 0건과 곡률 속도 계획을 유지한다. 전체 15조건을 재실행하며 계산 수정 자체를 native 성공으로 보고하지 않는다.

| 기록 항목 | Native 결과 |
| --- | --- |
| 기존 notebook 실행 version와 GPU | Version 11 / Kaggle T4 / COMPLETE |
| Native measured wheelbase·rear/front axle XY·front track·MAX gain | 약 3.00465m / rear `(-1.38641, -0.000006)m`, front `(1.61824, -0.000004)m` / 1.66684m / 70°; episode별 재측정 proof PASS |
| Original/applied/readback steering curve | 원본 `[[0,1],[20,0.9],[60,0.8],[120,0.7]]` 보존; 적용·readback `[[0,1],[100,1]]` PASS |
| D9 직선·좌곡선·우곡선 정답 경로 주행 | 중앙 3조건 포함 전체 15조건 PASS; 좌/우 실제 전륜·명령의 조향 부호도 확인 |
| D10 고정 목표 차선의 lateral/yaw recovery | 12조건 PASS; ±0.3m/±5° 초기 편차, 연속 1초 유지 완료 2.15~2.35 simulation seconds |
| RGB·snapshot·label·control frame 연결과 다운로드 후 재검증 | 1,540 frame·RGB 105장; full CPU gate·16 negative QA·Map 6,980 corner 모두 PASS |
| 소유 actor/server 종료와 실행 evidence 링크 | 15회 cleanup·RPC 복구 PASS, bootstrap `server_stopped=true`; [최종 증거](evidence/carla_driving_d8_d10_2026-10-08_summary.json) |

Native 결과를 채울 때 실제 failure와 수정·재실행 버전을 함께 기록한다. D8–D10의 개발 QA 결과를 학습 pilot, B0/M1 평가, D14 gate 또는 Town02 final benchmark의 완료로 확대하지 않는다.
