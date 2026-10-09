# Closed-loop 평가 설계

이 문서는 전체 benchmark의 설계 초안이다. Threshold와 event dedup·실패 포함 aggregation·latency 규칙은 provisional이며 전체 평가기는 구현 대상이다. 구현된 짧은 [privileged 주행 QA](driving.md)의 기준과 구분한다.

## 공통 비교 조건

B0 OpenCV와 M1은 RGB에서 [같은 `5 × 2` waypoint](data-contract.md)를 예측한다. Controller·속도 상한·0.05초 주기·steering limit·smoothing·저수준 속도 제어를 공통으로 사용한다. GT waypoint reference는 privileged 방법으로 따로 표시한다. Perception에는 pose/map/route를 전달하지 않으며 camera shift에서도 실제 perturbed calibration을 추가 입력하지 않는다.

Episode는 warmup/settling 뒤 고정 start에서 시작한다. 유효 주행 길이 L과 초기 조건을 실행 전에 정하고 종료점 뒤 최소 15m의 branch-free label corridor를 확보한다. 같은 route/condition/initial state 쌍을 각 방법에 적용하며 결정적 재실행을 독립 초기 조건으로 세지 않는다.

## Progress와 오차

Actor XY를 원래 route의 물리적으로 도달 가능한 국소 연속 구간에 투영해 `s_t`를 구한다. `p_t=clip(max(p_(t-1),s_t-s_start),0,L)`, completion은 `100*p_end/L`이다. 우측 양수 signed lateral error와 target tangent 대비 최소 signed heading error를 기록한다. 실제 이동 거리는 연속 actor XY displacement 합으로 별도 저장한다.

실패 episode를 포함해 pooled abs lateral mean/p95, episode별 평균·최대·종료 시점, completion과 종료 원인을 함께 보고한다. 조기 실패의 짧은 관측 구간을 낮은 오차만으로 비교하지 않는다. Warmup 제외 범위도 metric version에 명시한다.

## 종료 후보

| 이유 | Provisional 판정 |
| --- | --- |
| `route_completed` | Progress가 종료점 0.5m 이내 |
| `collision` | 평가 시작 후 첫 유효 collision |
| `lane_departure` | Footprint가 목표 경계 밖에 연속 0.5초 |
| `stalled` | Speed <0.1m/s·progress 증가 <0.1m 상태가 연속 5초 |
| `timeout` | 평가 simulation time 120초 초과 |
| `route_horizon_exhausted` | Branch-free 미래 corridor 15m 미만 |
| `technical_failure` | Sensor/frame/queue·nonfinite output·파일/RPC 오류 |
| `intervention` | 사람 또는 외부 안전 제어가 control을 변경 |

동시 발생 원본 flag는 모두 저장한다. 주 이유 우선순위는 technical failure → collision → lane departure → intervention → route completion → stall → timeout → horizon exhaustion이다. 마지막 유효 frame까지 누적 거리·event·오차를 보존한다. Invalid output을 임의 정상 control로 바꾸지 않으며 주행 실패를 technical failure로 재분류해 평가에서 빼지 않는다. Footprint는 actor 중심만으로 판정하지 않는다.

## Event·recovery·latency

Collision 원본은 frame·other actor·impulse를 저장하고 같은 frame/other actor 중복만 합친다. Lane invasion은 marking 상세를 보존하고 동일 frame/marking 집합 중복을 합친다. 연속 같은 경계/방향 crossing을 0.5초 window로 묶는 안은 실제 이벤트와 대조하여 확정한다.

Condition별 event rate는 `sum(events)/(sum(actual_distance_m)/1000)`이다. 실패 episode의 거리도 포함하며 총 거리가 0이면 `N/A`다. Intervention-free distance는 시작부터 첫 intervention 또는 종료까지의 실제 거리이며 종료 원인을 함께 제시한다.

Recovery 후보 기준은 5초 안에 abs lateral ≤0.2m·abs heading ≤3°를 연속 1초 유지하는 것이다. 성공률·복귀 시간·최대 오차·종료 이유를 함께 보고한다. Normalized steering의 `Δu/Δt`, `Δ²u/Δt²` absolute mean/p95/max를 기록하되 물리 lateral jerk로 부르지 않는다.

동기 simulator의 완주는 실시간 처리 증거가 아니다. 같은 하드웨어·batch 1·resolution에서 전처리·추론·controller·전체 pipeline의 wall-clock median/p95를 측정하며 GPU 작업 완료를 포함한다. Sensor/render/queue 대기는 별도 기록하고 20Hz의 50ms 예산과 비교한다. Capture의 `tick_to_rgb_ms`를 inference latency로 사용하지 않는다.

## 집계와 held-out 독립성

Episode 단위 CSV에서 condition별 completion·성공 수/전체 수, pooled 오차, event total/distance/rate, recovery와 latency를 집계한다. 동일 route/초기 조건 쌍의 차이로 비교하고 scene·training seed·evaluation 초기 조건 수를 구분한다.

Town02는 held-out final용으로 예약한다. 개발에서 metric·종료·dedup·controller/calibration 정책과 threshold를 확정한 뒤 checkpoint/code/config/dataset/split/route/metric hash 및 조건을 freeze manifest에 고정한다. Final 결과를 보고 모델·controller·route·threshold를 선택하지 않는다. 결과 기반 개선은 새 개발 버전과 새 held-out 평가로 진행한다.
