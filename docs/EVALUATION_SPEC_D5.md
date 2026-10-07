# D5 Closed-loop 평가 명세 초안

작성일: 2026-10-07 (Asia/Seoul). **상태: DRAFT / thresholds PROVISIONAL / metric implementation NOT_IMPLEMENTED.** 이 문서는 D14 평가 구현에 사용할 의미와 기록 항목을 정리한다. 아직 주행 결과, metric 계산기 또는 최종 threshold 동결 증거가 아니다. Train/dev 실험으로 정의·수치를 검증하고 D42에 version/hash와 함께 동결한 뒤 D43부터 final을 실행한다.

## 비교와 episode 단위

B0 OpenCV와 M1은 RGB에서 [같은 waypoint 계약](DATA_CONTRACT_D5.md)의 `5 × 2` 출력을 만든다. 최종 controller, 25km/h target, 0.05초 제어 주기, steering limit, smoothing과 저수준 속도 제어를 공통 사용한다. Pose/map/target-route가 perception input으로 전달되지 않게 분리한다. Camera shift에서도 B0의 nominal calibration을 유지하고 M1에 실제 perturbed calibration을 추가 입력하지 않는다.

GT waypoint를 controller에 공급하는 **privileged reference**는 라벨/제어 검증용으로 따로 표시한다. Controller 비교와 perception 비교도 별도 표로 기록한다. Methods별 controller를 바꿔놓고 개선을 모델 효과로 주장하지 않는다.

Episode의 평가 구간은 warmup/settling 후 미리 정한 start에서 시작한다. 고정된 reference centerline에 대해 route progress와 오차를 측정한다. 실제 moving evaluation의 유효 route 길이 `L`을 실행 전에 정하고, 그 종료점 뒤에도 최소 15m의 branch-free label corridor를 확보한다. D5의 35m topology survey와 정지 smoke에는 주행 완주율을 계산하지 않는다.

Raw record에는 dataset/route/scene/episode, method/checkpoint/config/hash, controller, seed 또는 실제 초기 조건, weather/camera, simulation frame/time, actor pose/speed/control, 원본 sensor event, 처리 시간과 종료 이유를 남긴다. 모든 방법이 동일한 `(route, condition, initial state)` 쌍을 실행한다. 설정이 같은 deterministic 재실행은 독립 초기 조건으로 세지 않는다.

## Progress와 오차

현재 actor XY를 **원래 목표 route polyline**의 국소 연속 구간에 투영하여 arc position `s_t`를 구한다. 매 frame 전체 도로의 nearest lane을 새 목표로 선택하지 않는다. 이전 progress에서 물리적으로 도달 가능한 범위만 검색하여 교차/근접 구간으로 projection이 튀지 않게 한다. 이 검색 범위도 D14에서 test하고 D42에 고정한다.

- 진행 거리 `p_t = clip(max(p_(t-1), s_t - s_start), 0, L)`.
- Route completion `100 × p_end / L`; 각 episode의 성공 여부와 종료 원인을 함께 보고한다.
- Signed lateral error는 고정 route의 진행 방향 tangent 기준 우측을 양수로 한 XY 거리다. Mean/p95는 `abs(error)` m다.
- Heading error는 actor yaw와 같은 target-route tangent의 최소 signed angle 차이다.
- 실제 주행 거리는 연속 actor XY displacement의 합이며 route progress와 따로 저장한다. 뒤로 이동·차선 편차도 exposure에 포함한다.

모든 유효 frame과 **실패 episode**를 포함한다. Pooled mean/p95, episode별 평균/최댓값/종료 시점, 완주율을 함께 제시한다. 조기 실패로 관측 frame이 짧아진 episode가 낮은 오차만 보여 성능이 좋아 보이는 일을 완주율·종료 기록 없이 해석하지 않는다. 초기 정지/warmup은 평가 frame에서 제외하되 제외 범위를 version에 명시한다.

## 종료 규칙 초안

아래 수치는 구현·개발 실험용 **초기 후보**다. Simulator 기능이 보장하는 기본값이나 최종 성공 기준이 아니다. D14에서 event/geometry·차량 크기에 맞는지를 확인하고 train/dev로 조정하며 D42에 동결한다. 주행하지 않은 D5 smoke에 적용하여 성공을 표시하지 않는다.

| 종료 이유 | Provisional 판정 |
| --- | --- |
| `route_completed` | Progress가 사전 route 종료점 0.5m 이내 도달 |
| `collision` | 평가 시작 후 ego의 첫 유효 collision event; 같은 frame 원본 중복은 한 event |
| `lane_departure` | 차량 footprint가 목표 차선 경계 밖에 있는 상태가 연속 0.5초 지속; footprint·lane width는 실제 차량/route에서 확인 |
| `stalled` | Speed <0.1m/s 및 progress 증가 <0.1m 상태가 연속 5초 지속 |
| `timeout` | 평가 시작부터 simulation time 120초 초과 |
| `route_horizon_exhausted` | 남은 branch-free corridor가 15m 미만; 사전 route 설계 오류를 함께 기록 |
| `technical_failure` | Frame/queue 불일치, sensor timeout, nonfinite model/controller output 또는 파일/RPC 오류 |
| `intervention` | 사람이나 외부 안전 제어가 steering/control을 바꾼 시점 |

동일 frame에서 충돌·차선 이탈·완료가 함께 발생하면 원본 flag를 모두 저장하고 주 종료 사유는 **technical failure → collision → lane departure → intervention → route completion → stall → timeout → horizon exhaustion** 순으로 정한다. 누적 주행·event·오차는 마지막 유효 frame까지 보존한다. Invalid output을 임의 0 steering으로 바꾸고 정상 episode처럼 계속 평가하지 않는다.

Footprint가 경계를 벗어났는지는 actor origin만으로 판정하지 않는다. 차량 bounding box/geometry와 목표 차선 경계를 써야 하며, 중심 좌표계와 제어 기준점 offset을 혼동하지 않는다. Recovery 시작점은 차량이 실제로 차선 안에 있고 시작 편차가 사전 정의된 범위에 있을 때만 사용한다.

Technical failure도 전체 실행 시도 목록에서 삭제하지 않는다. 측정 성능과 실행 무효 건수를 구분하여 원인을 보고하고 같은 조건 재실행 여부를 기록한다. 코드 수정으로 metric 의미가 바뀌면 영향받는 모든 방법을 다시 평가한다. 주행 실패를 technical failure로 재분류하여 완주율에서 빼지 않는다.

## Event 및 거리 집계 초안

Collision은 raw `(frame, other_actor_id, other_actor_type, impulse)`를 보존한다. 같은 frame·같은 other actor의 중복 callback만 합친다. 첫 collision에 종료하므로 terminal event도 count에 포함한다.

Lane invasion은 raw frame/time과 crossed marking 정보를 보존한다. 동일 frame·동일 marking 집합의 중복 callback은 하나로 합친다. 연속 경계 crossing을 반복 count하지 않도록 **같은 경계/방향의 0.5초 window**를 묶는 안을 D14에서 검증한다. Sensor가 고유 경계 ID를 주지 않으면 고정 target-route의 좌/우 경계와 pose로 구분하고, 실제 event와 대조하여 false merge가 없는지 확인한 뒤 확정한다. 하나의 event에 여러 marking이 있어도 event count는 하나며 marking 상세는 따로 저장한다.

Condition/group별 event rate는 `sum(event_count) / (sum(actual_distance_m)/1000)`다. Episode rate의 단순 평균을 사용하지 않는다. 총 주행 거리가 0이면 `/km`는 `N/A`로 기록하며 0으로 만들지 않는다. Collision과 invasion이 발생한 실패 episode의 거리도 분모에 포함한다.

Intervention-free distance는 평가 시작부터 첫 intervention 또는 종료까지의 실제 거리다. 종료 원인을 같이 제시하여 자발적인 route completion과 충돌 직전 거리의 의미를 구분한다. 자동 안전 종료·사람 개입도 구분해서 남긴다.

## Recovery·steering·latency

Recovery 대상은 사전에 고정된 원래 목표 lane의 lateral/yaw offset 초기 조건이다. 초기 후보 성공 조건은 **5초 안에 `abs(lateral_error) ≤0.2m`, `abs(heading_error) ≤3°`를 연속 1초 유지**하며 그 전에 collision/departure/intervention이 없는 경우다. 성공률 `successes/all_valid_recovery_episodes`, 복귀 시간, 최대 오차 및 종료 원인을 보고한다. 수집·검증을 하지 않은 현재 D5에는 recovery 성능값이 없다.

적용한 normalized steering `u ∈ [-1,1]`에 대해 동일 simulation 주기에서 `Δu/Δt`와 `Δ²u/Δt²`를 계산한다. 각각 absolute mean/p95와 최대치를 기록하고 raw prediction·clipped·smoothed·applied 값을 구분한다. 이를 차량의 물리적인 lateral jerk라고 부르지 않는다. 실제 조향각 변환이나 차량 가속도 jerk는 별도 정의·구현이 필요하다.

동기 simulator는 client를 기다리므로 완주가 실시간 처리의 증거가 아니다. 동일 하드웨어·batch 1·동일 resolution에서 전처리, 모델 추론, controller, 전체 pipeline의 wall-clock **median/p95**를 측정한다. GPU 비동기 작업은 실제 완료까지 포함하도록 측정 경계를 정하고 warmup 횟수와 측정 frame 수를 기록한다. Sensor/render/queue 대기 시간은 별도 기록한다. 20Hz의 50ms 예산과 전체 pipeline p95를 비교하며 hardware·GPU workload도 같이 공개한다. D4의 `tick_to_rgb_ms`는 capture 진단이며 M1 inference latency로 인용하지 않는다.

## Aggregation과 final 독립성

먼저 episode CSV를 남기고 condition별 completion 평균·표준편차와 성공 수/전체 수, pooled lateral mean/p95, event total·distance·rate, recovery 및 latency를 생성하는 계획이다. 비교는 동일 route/initial-condition pair의 차이를 사용한다. 서로 다른 route와 seed/실제 초기 상태에서 변동을 구하며, 같은 scene의 3회 결정적 재실행을 3개의 독립 장면으로 표시하지 않는다. Training seed 수와 evaluation 초기 조건 수를 별도로 기록한다.

Final 기본안은 clean unseen route, 별도 town, rain, sunset, camera pitch의 5개 condition group이다. 초안 예산은 group별 2 routes ×3 initial conditions = 방법당 30 episodes이며 B0/M1/privileged reference 합계 90이다. 실제 설치 map·주행 비용·복구 범위를 확인한 뒤 D42 이전에 규모를 확정한다. 현재 Town02는 **예약, UNCOLLECTED, UNVERIFIED**이고 final route·sample·metric은 없다.

D42 freeze manifest에는 checkpoint·code/config/dataset/split/route/metric definition hash, controller와 calibration 정책, 조건/초기 상태, termination·dedup/recovery threshold, 평가 수 및 실패 처리 규칙을 담는다. D43부터 final 데이터를 처음 평가하며 결과를 보고 모델, controller, route 또는 threshold를 고르지 않는다. 결과 기반 개선은 새 개발 버전과 새 held-out 평가로 진행한다.

## D14 인계 및 현재 증거의 한계

D14 구현 전 fixed-route projection, 실패 포함 completion/error aggregation, zero-distance rate, terminal-event count, duplicate-event merge, simultaneous termination, recovery duration, normalized steering difference, timing 경계를 unit fixture와 privileged reference에서 확인한다. 이후 moving-route 실행으로 올바른지 검증한다.

현재 구현된 것은 D4의 frame capture/disk integrity와 D5의 contract/split guard다. **이 문서에 정의된 route progress·closed-loop metrics·termination 계산 코드는 아직 구현하지 않았다.** Schema와 정지 smoke PASS를 lane following 성공, final benchmark 또는 성능 threshold 동결로 표시하지 않는다.

공식 simulator 동작 근거: [CARLA 0.9.16 collision/lane invasion 센서](https://carla.readthedocs.io/en/0.9.16/ref_sensors/), [Map/Waypoint](https://carla.readthedocs.io/en/0.9.16/core_map/), [동기화와 timestep](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/). 위 수치·집계·평가 예산은 검토할 프로젝트 설계안이다.
