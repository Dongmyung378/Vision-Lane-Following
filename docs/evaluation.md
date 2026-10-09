# Closed-loop 평가

평가기 `carla_tools/evaluation.py`는 simulator와 독립적으로 원본 frame과 event를 재생한다. 설정은 `configs/evaluation.json`이며 정규 JSON의 SHA-256을 각 metric에 포함한다. 주행 실패도 CSV의 한 행으로 남긴다. 짧은 개발 주행은 timeout 등 설정을 변경할 수 있지만 실제 적용한 설정과 hash를 episode metadata에 고정해야 한다.

기본 thresholds는 개발 비교 기준이다. Held-out final 전에 threshold와 metric 정책을 확정하고 freeze한다.

## 공통 비교 조건

B0 OpenCV와 M1은 RGB에서 [같은 `5 × 2` waypoint](data-contract.md)를 예측한다. Controller·속도 상한·0.05초 주기·steering limit·smoothing·저수준 속도 제어를 공통으로 사용한다. GT waypoint reference는 privileged 방법으로 따로 표시한다. Perception에는 pose/map/route를 전달하지 않으며 camera shift에서도 실제 perturbed calibration을 추가 입력하지 않는다.

Episode는 warmup/settling 뒤 고정 start에서 시작한다. 유효 주행 길이 L과 초기 조건을 실행 전에 정하고 종료점 뒤 최소 15m의 branch-free label corridor를 확보한다. 같은 route/condition/initial state 쌍을 각 방법에 적용하며 결정적 재실행을 독립 초기 조건으로 세지 않는다.

## Progress와 오차

Actor XY를 원래 route의 물리적으로 도달 가능한 국소 연속 구간에 투영해 `s_t`를 구한다. `p_t=clip(max(p_(t-1),s_t-s_start),0,L)`, completion은 `100*p_end/L`이다. 우측 양수 signed lateral error와 target tangent 대비 최소 signed heading error를 기록한다. 실제 이동 거리는 연속 actor XY displacement 합으로 별도 저장한다. Completion tolerance로 성공한 episode도 계산한 percentage를 그대로 보고한다.

실패 episode를 포함해 pooled abs lateral mean/p95, episode별 평균·최대·종료 시점, completion과 종료 원인을 함께 보고한다. 조기 실패의 짧은 관측 구간을 낮은 오차만으로 비교하지 않는다. 평가는 첫 driving frame부터 마지막 driving frame까지이다. Warmup과 controlled-stop sample은 제외한다.

Frame은 정수이며 인접 driving frame은 `F → F+1`이어야 한다. Timestamp는 증가해야 하며 설정 주기와 차이는 `timestamp_tolerance_seconds` 이내여야 한다. Missing, stale, nonfinite observation은 마지막 유효 거리·오차를 보존한 채 integrity failure로 기록한다. 관측이 끊긴 시간을 recovery나 lane departure 유지 시간으로 세지 않는다.

## 종료 규칙

| 이유 | 기본 판정 |
| --- | --- |
| `route_completed` | Progress가 종료점 0.5m 이내 |
| `collision` | 평가 시작 후 첫 유효 collision |
| `lane_departure` | Footprint가 목표 경계 밖에 연속 0.5초 |
| `stalled` | Speed <0.1m/s·progress 증가 <0.1m 상태가 연속 5초 |
| `timeout` | 평가 simulation time이 120초에 도달 |
| `route_horizon_exhausted` | Branch-free 미래 corridor 15m 미만 |
| `technical_failure` | Sensor/frame/queue·nonfinite output·파일/RPC 오류·예측 결과 없음 |
| `intervention` | 사람 또는 외부 안전 제어가 control을 변경 |

동시 발생 원본 flag는 모두 저장한다. 주 이유 우선순위는 technical failure → collision → lane departure → intervention → route completion → stall → timeout → horizon exhaustion이다. Callback이 늦게 도착해 먼저 완료가 관측되었더라도 마지막 frozen snapshot의 유효 driving collision이 우선한다. 그동안 실제 관측한 sample과 거리를 삭제하지 않는다. Footprint는 actor 중심만으로 판정하지 않는다.

예측 실패로 정상 waypoint가 없으면 `technical_failure` outcome을 저장하고 control을 임의 정상 값으로 대체하지 않는다. 유효 frame evidence를 가진 방법 실패는 `integrity_valid=true`일 수 있다. 누락·손상·불연속 frame, 검증되지 않은 종료 선언과 손상 event는 `integrity_valid=false`로 구분한다. 방법 실패와 파이프라인 증거 손상을 구분하면서 둘 다 총 episode 수에 포함한다. 시작 전에 실패해 유효 길이를 알 수 없으면 completion은 `N/A`이고 실패 행은 유지한다.

## Event·recovery·latency

Collision 원본은 frame·other actor·impulse를 저장하고 같은 frame/other actor 중복만 합친다. Lane invasion은 marking의 type·color·lane-change·width를 보존하고 동일 frame/marking 집합 중복을 합친다. Marking 순서는 중복 판정에 영향을 주지 않는다. Native payload에는 안정적인 boundary/direction identity가 없어 서로 다른 frame의 event를 0.5초 window로 합치지 않는다. Lane event 수와 footprint 기반 연속 lane departure 종료는 서로 다른 metric이다.

Condition별 event rate는 `sum(events)/(sum(actual_distance_m)/1000)`이다. 실패 episode의 거리도 포함하며 총 거리가 0이면 `N/A`다. Intervention-free distance는 시작부터 첫 intervention 또는 종료까지의 실제 거리이며 종료 원인을 함께 제시한다. Warmup·stop event는 driving rate에 포함하지 않고 runtime 안전 기록으로 별도 보존한다.

Recovery는 시작 뒤 5초 안에 abs lateral ≤0.2m·abs heading ≤3°를 연속 1초 유지하면 성공이다. 복귀 시간은 유지 구간이 완료된 시점이며 처음 tolerance에 들어온 시점이 아니다. 중간에 tolerance를 벗어나면 유지 시간을 초기화한다. Center/normal 조건은 recovery 성공률의 분모에 넣지 않으며 offset/yaw 조건을 `recovery_required`로 구분한다.

실제 applied normalized steering에서 `Δu/Δt`와 인접 slope 차이의 second derivative를 계산한다. Unequal timestep에는 `2*(slope_next-slope_previous)/(dt_previous+dt_next)`를 쓴다. Absolute mean/p95/max를 보고하며 물리 lateral jerk로 부르지 않는다.

동기 simulator의 완주는 실시간 처리 증거가 아니다. 같은 하드웨어·batch 1·resolution에서 전처리·추론·controller·전체 pipeline의 wall-clock median/p95를 측정하며 GPU 작업 완료를 포함한다. Sensor/render/queue 대기는 별도 기록하고 20Hz의 50ms 예산과 비교한다. Capture의 `tick_to_rgb_ms`를 inference latency로 사용하지 않는다.

## 집계와 held-out 독립성

Episode CSV에는 종료 이유/flag, evidence integrity, completion, 실제 거리, event count/rate, recovery, footprint margin, steering과 latency가 들어간다. P95는 sample을 정렬한 뒤 `(n-1)*0.95` 위치를 선형 보간한다. Aggregate는 method·condition·evaluation config hash로 구분한다. 실패를 포함한 전체 frame sample을 pooling하며 episode별 mean을 다시 평균하지 않는다. Technical-failure outcome 수와 integrity-failure 수를 별도 집계한다. Missing latency는 0으로 채우지 않는다. 동일 route/초기 조건 쌍의 차이로 비교하고 scene·training seed·evaluation 초기 조건 수를 구분한다.

## API와 원본 재생

```python
from carla_tools.evaluation import EvaluationTracker, load_evaluation_config

config = load_evaluation_config()
tracker = EvaluationTracker(config, route_length_m=18.0,
                            start_reference_m=1.0, condition="center")
decision = tracker.update(row, events=[], remaining_horizon_m=20.0)
# 종료 뒤 callback을 drain/seal한 frozen snapshot으로 metric을 확정한다.
metric = tracker.finalize(metadata, events=frozen_events)
```

`route_length_m`은 추가 label corridor를 제외한 유효 평가 길이 `L`이다. Row는 `frame`, `timestamp`, `tracking.reference_arc_m/lateral_error_m/heading_error_deg`, `vehicle.transform.location`, `vehicle.applied_control.steer`, `speed_mps`, `body_lane_margin_m`을 포함한다. 같은 의미의 flat `time_seconds`, `reference_m`, `actor_transform`, `applied_control`도 지원한다. 전처리·추론·controller·pipeline 등의 timing은 `latency_ms`의 별도 key로 입력한다.

독립 재생은 native `episode.json` 또는 generic `metadata.json`의 `evaluation_route_length_m`, `evaluation_start_reference_m`, condition, method와 종료 이유를 읽는다. 두 metadata 파일이 동시에 있으면 모호한 입력으로 거부한다. Embedded `evaluation_config`가 있으면 기본적으로 해당 설정을 사용하며 hash를 검증한다. 명시적인 `--config` override는 각 episode의 frozen 설정과 일치해야 한다. `steps.jsonl` 원본과 metadata의 frozen events를 변경하지 않는다. 별도 `events.json`이 있으면 그 파일을 event 입력으로 쓴다. CSV는 exclusive atomic publication으로 작성하며 이미 존재하는 출력과 원본 입력 파일을 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_episodes.py `
  --episodes results/local/closed-loop/episode-a results/local/closed-loop/episode-b `
  --config configs/evaluation.json --output results/local/closed-loop/metrics.csv
```

Town02는 held-out final용으로 예약한다. 개발에서 metric·종료·dedup·controller/calibration 정책과 threshold를 확정한 뒤 checkpoint/code/config/dataset/split/route/metric hash 및 조건을 freeze manifest에 고정한다. Final 결과를 보고 모델·controller·route·threshold를 선택하지 않는다. 결과 기반 개선은 새 개발 버전과 새 held-out 평가로 진행한다.
