# D9 고정 route 주행·D10 초기 오차 복귀 검증

작성일: 2026-10-08 (Asia/Seoul). 범위는 `D8-D10-privileged-controller-development-QA`, gate version은 `vlf-privileged-driving-d10-v1`이다. D8 controller에 원래 목표 차선의 GT waypoint를 공급하여 주행과 초기 오차 복귀를 확인한다. 구현은 [driving.py](../carla_tools/driving.py), [driving_route.py](../carla_tools/driving_route.py), [driving_validation.py](../carla_tools/driving_validation.py)이며, 실행 조건은 [driving_validation_d10.json](../configs/driving_validation_d10.json)에 기록한다.

## 현재 검증 상태

**D8/D9/D10 gate PASS.** 기존 Kaggle notebook `dongmyungpark/vlf-carla-rgb-smoke-20261005`의 version 11, T4·CARLA 0.9.16·Town01에서 15개 native 조건이 모두 통과했다. 전체 다운로드의 independent CPU gate, 16가지 negative QA와 actual Map corner audit도 PASS했다. Version 5~10의 실패와 수정 증거를 보존하며, native 성공 전까지 PENDING으로 관리했던 기록을 이번 실제 결과로 확정한다.

| Route | 중앙 reference 주행 | 4개 초기 편차 복구 | 전체 5조건의 최소 차체 여유 | 중앙 조건 평균 횡오차 |
| --- | --- | --- | ---: | ---: |
| 직선 | PASS / 19.223m | PASS / 2.15~2.30초 | 0.60356m | 0.000587m |
| 좌곡선 | PASS / 19.204m | PASS / 2.15~2.30초 | 0.48180m | 0.07029m |
| 우곡선 | PASS / 19.115m | PASS / 2.15~2.35초 | 0.24783m | 0.09101m |

시간은 simulation time이고, 복구 시간은 tolerance 안에서 연속 1초 유지가 끝난 시점이다. 15개 episode의 주행 시간은 4.10~6.05초이며 최초 저속 정지 확인은 0.50~0.70초였다. 원래 35m 경로에서 종료 reference 19m 이상을 검증한 개발 QA다. 실제 주행 1,540 frame·RGB 105장, 모든 주행·정지의 최대 횡오차 0.34613m, 주행 최대 속도 23.645km/h, 충돌·차선 침범·callback 오류 0건이다. 차체 모서리 6,980개 모두 원래 Driving lane 안이며 height-sensitive corner와 bad corner는 0개였다. 전체 원본 hash가 유지됐고, 28개 runtime source와 원격 notebook source도 제출 hash와 같다.

[최종 source·gate·cleanup 증거](evidence/carla_driving_d8_d10_2026-10-08_summary.json), [15조건 metric CSV](evidence/carla_driving_d8_d10_2026-10-08_metrics.csv), [복구·곡선 속도 그래프](evidence/carla_driving_d8_d10_2026-10-08_recovery.png), [직선](evidence/carla_driving_d8_d10_2026-10-08_straight_left_offset_overlay.png)·[좌곡선](evidence/carla_driving_d8_d10_2026-10-08_left_curve_left_offset_overlay.png)·[우곡선](evidence/carla_driving_d8_d10_2026-10-08_right_curve_left_offset_overlay.png) 동일 frame overlay를 보존한다. Full source artifact는 `results/local/kaggle-d10-output-v11/driving/`에 있고 Git에는 작은 검토용 증거만 저장한다.

최종 일반 CPU suite는 **450 tests PASS·0 skip, 49.122초**였다. 앞선 두 재검증에는 서로 다른 기존 episode/split 테스트에서 OS 파일 쓰기 오류가 1개씩 발생했다. 두 파일의 독립 재검증, 원래 동작을 유지한 진단 실행, 계측 없는 최종 전체 실행은 모두 PASS했다. 당시 OSError 원인은 재현하지 못했으며 실패 로그는 `results/local/d10-unit-tests-450-first-attempt.log`와 `results/local/d10-unit-tests-450-second-attempt.log`에 보존했다. Runtime을 수정하거나 오류를 통과로 처리하지 않았다.

Version 9의 우곡선 마지막 frame에서 실제 8모서리 최소 여유는 0.13891m로, 사전 기준 0.15m를 만족하지 못했다. 동일 pose를 CARLA `Map` API로 독립 확인한 최소 여유도 0.13953m였다. CARLA의 normalized steer는 안쪽 전륜 각도로 적용되는데 기존 controller는 가상 중앙 조향각을 정규화했다. 실제 다음 frame의 전륜 readback에서 요청 곡률 대비 약 10.88% 부족한 조향을 확인했다. Controller v3는 측정한 전륜 간격을 사용하여 Ackermann 안쪽 각도로 변환하며 안전·복귀 threshold를 유지한다. [실패와 실제 바퀴 증거](evidence/carla_driving_d8_d10_2026-10-08_ackermann_failure.json)를 보존한다.

Version 10의 우곡선은 122 frame·reference 19.074m의 주행을 끝냈고 마지막 주행 여유는 0.16609m였다. 이후 세 번째 감속 frame에서 여유가 0.14384m로 줄었으며 실제 Map 조회도 0.14530m였다. 라벨이 actor-origin 경로인데 기존 pursuit 식은 rear-axle 경로를 가정했다. Controller v4는 현재·미래 actor 원점을 같은 rear-axle 회전 중심의 원에 놓는 식으로 수정한다. Actor target `p`, 측정 rear offset `r`에 대해 `κ_rear = 2*p_y / (|p-r|² - |r|²)`이며 라벨·목표 경로를 옮기거나 임의 편향을 더하지 않는다. 원래 target 선택과 Ackermann 변환도 유지한다. [실패 원본 요약](evidence/carla_driving_d8_d10_2026-10-08_actor_origin_failure.json)을 남긴다.

CPU 검증기는 정상 synthetic trace와 손상·오정합 trace를 검사하는 테스트를 통과했다. 이 결과는 native CARLA 실행 결과와 별도로 해석한다. 원격 작업의 `COMPLETE` 상태만으로 완료를 판정하지 않고 bootstrap, 15개 episode와 전체 gate의 PASS를 모두 확인한다.

## 원래 목표 차선과 좌표

Town01의 D6 고정 route catalog를 사용한다. 각 native episode를 시작하기 전에 현재 simulator의 map/version과 해당 spawn의 물리적 route를 다시 읽어 기존 계약과 일치하는지 확인한다. Route는 dense XYZ 차선 중심선을 누적하여 0.25m 물리 거리 간격으로 재표본화한 35m polyline이다. OpenDRIVE reference `s`를 실제 외측 차선의 이동 거리로 대신 사용하지 않는다.

초기 횡변위나 yaw를 적용해도 원래 route, scene, 목표 start-lane ID를 유지한다. 차량이 다른 차선에 가까워졌다는 이유로 목표 차선을 다시 배정하지 않는다. 차량 actor 위치를 고정 route에 투영하여 현재 reference arc를 구하고, 이후 frame에서는 이전 reference의 `[-1m, +3m]` 범위와 속도 제한을 함께 검사한다. 오래된 frame·timestamp, 큰 진행 점프, 모호한 route 투영, 충분하지 않은 미래 horizon은 실패로 처리한다.

Label은 **현재 sensor·snapshot frame의 vehicle actor transform**을 원점으로 하는 `x-forward, y-right`, 단위 m의 `5 × 2` 배열이다. 원래 차선 중심선을 따라 현재 reference 이후 3·6·9·12·15m의 world 타깃을 보간한 다음, 같은 frame의 actor inverse transform으로 변환한다. 횡오차와 heading error의 양수는 CARLA 진행 방향의 우측이다. 진행 거리의 현재 값과 지금까지의 최대값을 각각 기록하여 뒤로 이동한 모습을 최대 진행 값으로 숨기지 않는다.

D8 Pure Pursuit 계산은 측정한 rear axle 좌표를 제어 원점으로 사용한다. Actor-origin label을 유지하면서 rear axle까지 좌표를 이동하고, actor-origin 궤적을 따르는 회전 원으로 rear curvature를 계산한다. Native wheel의 world-centimetre 위치를 기록한 actor pose로 역변환해 wheelbase·axle 위치·front track·front steering limit을 측정한다. Controller v4는 이 곡률을 Ackermann 안쪽 전륜 각도로 변환한 뒤 native gain으로 정규화한다. Python API에 native steering curve 보간 tangent가 없으므로 constant-factor curve를 적용하고 readback한다. 기존 curve와 실제 적용 curve도 증거에 남긴다.

## 사전에 정한 15개 조건과 통과 기준

직선·좌곡선·우곡선 각각에 아래 5개 조건을 적용한다. 초기 편차는 고정 catalog의 시작점(곡선 route는 접근 구간)에 적용하고, 복귀 후 곡선 진입과 감속까지 확인한다. 총 15개 episode이며, 가운데 조건 3개가 D9 reference 주행이고 나머지 12개가 D10 recovery 확인이다. 동일 설정 재실행은 독립적인 새로운 초기 조건으로 세지 않는다.

| 조건 | 요청 횡변위 | 요청 yaw |
| --- | ---: | ---: |
| `center` | 0m | 0° |
| `left_offset` | −0.3m | 0° |
| `right_offset` | +0.3m | 0° |
| `left_yaw` | 0m | −5° |
| `right_yaw` | 0m | +5° |

Camera, weather, 차량 blueprint, controller와 제어 주기를 공통으로 유지한다. CARLA spawn이 차선 중심에 있다는 가정을 하지 않는다. Warmup 후 차량의 z·pitch·roll을 보존하고 원래 고정 route의 첫 anchor XY와 tangent yaw에 정렬한 뒤, 그 우측 방향으로 lateral/yaw 편차를 적용한다. Spawn·중앙 정렬·편차 요청 pose를 각각 기록한다. 요청값만 저장하고 성공으로 표시하지 않고, 실제 주행 첫 frame의 횡오차와 heading error가 요청값의 각각 0.05m·0.5° 이내인지 검사한다.

| 항목 | D9/D10 개발 QA 기준 |
| --- | --- |
| 제어 주기·속도 설정 | 0.05초, 20Hz, nominal 속도 상한 25km/h; 같은 waypoint의 preview 곡률에 2.0m/s² 속도 계획 적용 |
| 종료 목표 | 원래 route의 absolute reference 19m 이상 |
| 유효 미래 label | 종료 frame에도 3·6·9·12·15m 타깃이 모두 route 안에 존재 |
| 실행 제한 | 최대 10초의 simulation time, 속도 ≤12m/s |
| 횡오차 제한 | 모든 주행 frame에서 절댓값 ≤0.8m |
| 차체 차선 여유 | 첫 frame을 포함한 모든 주행 frame에서 ≥0.15m |
| 초기 오차 복귀 | 횡오차 절댓값 ≤0.2m, heading error 절댓값 ≤3°를 **연속 1초 유지**, 해당 유지 구간 종료가 시작 후 5초 이내 |
| Sensor 이벤트 | collision 0, lane invasion 0, callback 오류 0 |
| 정지와 종료 후 관찰 | 마지막 적용 조향을 유지한 service brake, 2초 안에 속도 ≤0.1m/s; 최소 10 braking tick 이후에만 handbrake 주차·2 tick 추가 관찰, cleanup 이후 최종 이벤트 재확인 |
| 완료 증거 | episode `COMPLETE`, full disk gate PASS, owned actors 제거와 원래 world/weather 복원 PASS |

35m route에서 reference가 20m를 넘으면 마지막 15m 타깃을 만들 수 없다. 20Hz 주행의 마지막 tick은 종료 기준을 조금 넘을 수 있으므로 실행 전에 19m 기준을 정했다. Native 속도 상한에서 tick당 최대 이동은 약 0.6m다. Label horizon이 부족하면 마지막 타깃을 잘라내거나 반복하여 성공으로 만들지 않는다. 실제 종료 reference를 별도로 기록한다.

복귀는 한 frame이 tolerance 안에 들어오는 것만으로 통과하지 않는다. 조건을 벗어나면 연속 유지 시작 시점을 초기화하고, 충분한 1초 유지가 끝난 시간을 복귀 완료 시점으로 보고한다. Heading ±3°는 초기 오차 복귀 유지 구간의 기준이다. 곡선에서 actor 원점과 rear axle의 위치 차이 때문에 이후 heading error가 커질 수 있어 전체 주행의 ±3° 상한으로 사용하지 않는다. 이후에도 모든 frame의 차체 여유·횡오차·속도·센서 기준은 적용한다. 이 기준은 D10 개발 QA의 사전 기준이며, D14 전체 평가 명세와 D42 최종 threshold 동결을 대신하지 않는다.

## Sensor·차체·제어 기록 검증

CARLA collision과 lane-invasion sensor를 해당 차량에 부착하고 session의 owned actor로 등록한다. 주행·감속·주차와 cleanup 뒤 확인된 late event 및 callback 오류를 원본 metadata에 남긴다. 이벤트나 센서 처리 오류가 있으면 episode를 완료 데이터로 사용하지 않는다. `NONE`, width 0 marking도 무시하지 않는다. [공식 lane crossing 계산](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/road/element/LaneCrossingCalculator.cpp#L51-L98)은 도색이 없는 경계도 보고하며 road/section 전환 자체는 별도로 제외한다.

Python callback 진입 때부터 진행 중인 작업을 등록한다. Sensor stop과 session cleanup 뒤 이미 접수한 callback 완료를 최대 2초 기다리고, 미완료이면 영구 오류를 남겨 실패한다. 접수 종료 이후 호출은 거부하며, 최종 이벤트 snapshot·완료 판정·metadata 저장을 같은 잠금으로 처리한다. 느린 payload 변환 중 두 snapshot이 비어 보여 PASS로 저장되던 경합을 CPU에서 재현하고 수정했다. 이는 접수한 Python callback의 제한 시간 내 처리와 저장 일관성의 보장이며, native unsubscribe 이후 모든 미래 호출이 없어졌다는 보편적 보장은 아니다.

차체 여유는 actor 중심의 횡오차만으로 판단하지 않는다. Native bounding box의 중심·x/y/z extent에서 여덟 모서리를 만들고, 같은 frame의 3D actor transform으로 world에 옮긴다. 각 모서리를 원래 route의 가장 가까운 XY segment에 투영해 그 위치의 lane width와 signed lateral error로 여유를 계산하고 최솟값을 기록한다. Curve에서 actor 중심의 tangent만 사용하는 근사는 front corner의 여유를 크게 추정할 수 있어 사용하지 않는다. Bbox 자체 rotation이 있으면 실패한다. Native collision·lane sensor와 모서리 기하 검사를 함께 확인하며, 전체 benchmark의 경계 평가기는 D14에서 추가 검토한다.

종료 순간 조향을 0으로 바꾸거나 이동 중 handbrake로 주차하지 않는다. 마지막 적용 조향을 유지하고 `throttle=0, brake=1, hand_brake=false`로 감속한다. 실제 속도가 0.1m/s 이하이고 최소 10 tick을 관찰한 뒤에만 handbrake를 적용한다. 정지 관찰의 모든 frame·pose·속도·applied control·wheel angle·모서리 여유를 metadata에 저장하고 validator가 재검사한다. 이 구간에는 미래 horizon이 소진될 수 있으므로 주행용 waypoint를 생성하거나 잘라내지 않는다.

`steps.jsonl`은 모든 20Hz frame의 sensor/world/snapshot 번호와 시간, pose·velocity, applied control, original-route tracking과 label, controller 전체 결정·diagnostics, native front-wheel angle, 차체 margin을 기록한다. 첫 control은 정지 brake/hand-brake이며, frame `t`의 controller 명령은 `t+1`의 readback과 비교한다. 마지막 frame의 결정은 기록만 남기고 `command_target_frame=null`로 표시한다.

주행·감속·주차 명령은 `apply_batch_sync([ApplyVehicleControl(owned_vehicle_id, control)], False)`로 전송한다. 응답이 정확히 하나이고 오류가 없으며 actor ID가 소유 차량과 일치해야 기존 tick을 진행한다. `False`는 batch 안에서 숨은 tick을 만들지 않는다는 뜻이다. 다음 frame의 `get_control` 비교 허용치 1e-6도 그대로 유지하며 불일치 때 frame·expected·observed를 저장한다. [공식 RPC 구현](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/rpc/Server.h#L100-L114)에서 일반 `apply_control`의 비동기 호출은 game-thread 작업 완료를 기다리지 않는다. Version 7에는 실패 readback 값이 없어 비동기 순서 문제를 확정하지 않고 실행 순서를 보장하는 개선으로 처리했다.

RGB는 첫 frame·20 frame 간격·마지막 frame을 저장한다. 주행 pose와 제어 trace는 그 사이 frame도 모두 보존한다. 저장 RGB에 대해 PNG 전체 구조·CRC·SHA-256, 원래 BGRA→RGB hash와 red/blue witness를 확인한다. Camera mount·calibration도 같은 frame pose에서 검증한다.

[verify_driving.py](../scripts/verify_driving.py)는 원본 metadata와 JSONL을 다시 읽고, tracker와 controller를 첫 frame부터 동일 설정으로 재실행한다. 별도의 D5 waypoint validator도 원래 route의 world 타깃과 actor-origin label을 검사한다. Native wheel 기록에서 vehicle geometry를 다시 계산하고, episode 간에는 같은 차량의 world-to-local floating point 오차를 고려해 metre 필드는 1mm 이내, steering limit·curve는 1e-5 이내로 비교한다. 각 episode의 geometry hash도 보존한다. Map, controller, camera calibration 및 spawn을 제외한 collection config는 공통 hash가 같아야 한다.

Manifest에는 세 case × 다섯 condition이 빠짐없이 한 번씩 있어야 한다. Episode ID/path 중복, condition과 source identity 불일치, 같은 case에서 복귀 조건의 목표 route 변경을 거부한다. PASS 직전 metadata·JSONL·RGB를 다시 읽어 검사 도중 source 변경이 없었는지 확인한다. `RUNNING`·`FAILED` episode는 통과하지 않는다.

## 기존 Kaggle notebook 재사용과 재현

기존 ID와 제목 `VLF CARLA RGB smoke 20261005`를 유지한다. 새 notebook이나 Copy/Fork를 생성하지 않는다. GPU는 Kaggle의 T4를 사용하며, 다운로드 후 재검증은 로컬 CPU만 사용한다. 제출 전에 기존 notebook의 존재·접근과 실행 상태를 읽기 전용으로 확인한다. `RUNNING`이면 완료 또는 오류 종료까지 기다린 뒤 같은 notebook을 업데이트한다. 인증정보는 public bundle이나 Git에 넣지 않는다.

아래 PowerShell 명령은 프로젝트 root에서 실행한다. Output directory는 새 로컬 경로를 사용하며 기존 증거를 덮어쓰지 않는다.

```powershell
.venv-kaggle\Scripts\kaggle.exe kernels status dongmyungpark/vlf-carla-rgb-smoke-20261005
.venv-kaggle\Scripts\kaggle.exe kernels pull dongmyungpark/vlf-carla-rgb-smoke-20261005 -p results/local/kaggle-driving-existing -m
.venv\Scripts\python.exe scripts/prepare_kaggle_driving.py --existing-metadata results/local/kaggle-driving-existing/kernel-metadata.json --output-dir results/local/kaggle-driving-bundle
.venv-kaggle\Scripts\kaggle.exe kernels status dongmyungpark/vlf-carla-rgb-smoke-20261005
.venv-kaggle\Scripts\kaggle.exe kernels push -p results/local/kaggle-driving-bundle --accelerator NvidiaTeslaT4 -t 1500
```

실행 종료 후 전체 source artifact ZIP과 bootstrap·driving 로그를 선택하여 다운로드한다. 원격 `COMPLETE` 뒤에도 JSON 결과가 실패하면 원인을 조사하고 실패 증거를 보존한다.

```powershell
.venv-kaggle\Scripts\kaggle.exe kernels status dongmyungpark/vlf-carla-rgb-smoke-20261005
.venv-kaggle\Scripts\kaggle.exe kernels output dongmyungpark/vlf-carla-rgb-smoke-20261005 -p results/local/kaggle-driving-output --file-pattern '(^|/)(driving_artifacts\.zip|kaggle_carla_bootstrap\.json|carla_driving\.log|carla_server\.log)$'
Expand-Archive -LiteralPath results/local/kaggle-driving-output/driving_artifacts.zip -DestinationPath results/local/kaggle-driving-expanded
.venv\Scripts\python.exe scripts/verify_driving.py --manifest results/local/kaggle-driving-expanded/driving/driving_manifest.json --output results/local/driving-cpu-gate.json
.venv\Scripts\python.exe scripts/audit_driving.py --native results/local/kaggle-driving-expanded --output results/local/driving-cpu-negative.json
.venv\Scripts\python.exe scripts/audit_driving_map.py --native results/local/kaggle-driving-expanded/driving --output results/local/driving-cpu-map-gate.json
```

Native artifact는 `driving/episodes/{case}__{condition}/episode.json`, `steps.jsonl`, `images/*.png`와 `driving_manifest.json`, `driving_report.json`으로 구성된다. 원격 bootstrap은 자신이 시작한 CARLA server process group을 종료한다. 최종 작업 종료는 Kaggle 상태와 bootstrap의 종료 증거를 함께 확인한다.

[audit_driving.py](../scripts/audit_driving.py)는 전체 정상 gate를 먼저 통과한 실제 artifact의 임시 복사본에 16가지 손상을 독립적으로 적용해 모두 거부하는지 검사한다. Waypoint 단위·원래 목표 차선·적용 제어·명령 frame·controller 결정·native geometry·PNG CRC·누락 RGB·callback 오류·late collision·cleanup·trace hash·중복/누락 frame·중복/누락 manifest를 확인하며 원본 파일 hash가 그대로여야 PASS다.

[audit_driving_map.py](../scripts/audit_driving_map.py)는 CARLA server를 실행하거나 client에 접속하지 않는다. 로컬의 동일 SHA Town01 OpenDRIVE와 CARLA 0.9.16 `Map` API로 native lane sensor 방식의 yaw-only 네 차체 모서리를 주행·정지 frame마다 조회한다. 모서리의 실제 위치가 원래 route의 driving lane 안에 있어야 하며, 높이를 0으로 변경한 supplementary 조회에서 판정이 달라지면 실패한다. 이 독립 확인은 3D 여덟 모서리 margin 및 실제 lane sensor 결과와 함께 해석한다. 기본 OpenDRIVE 경로는 로컬 배포이고 다른 위치는 `--xodr`로 지정한다. 두 audit는 원본 내부·기존 보고서 출력 경로를 거부하며 CARLA GPU를 사용하지 않는다.

## 이후 단계와 해석 범위

이 QA는 GT waypoint를 입력으로 사용한다. Camera-only 인지 모델이나 B0 OpenCV의 waypoint 정확도, 모델 inference latency, 전체 benchmark 완주율·독립적인 운전 성능을 검증한 결과로 해석하지 않는다. 15개 짧은 route episode는 D13의 moving training pilot을 대신하지 않는다.

D14에서 [평가 명세 초안](EVALUATION_SPEC_D5.md)에 따라 전체 metric·종료 규칙을 구현하고, train/dev에서 검증한 뒤 D42 freeze 이후 final을 실행한다. Town02 final은 계속 `UNCOLLECTED`/`UNVERIFIED`이며 이 개발 QA에서 수집·열람·사용하지 않는다.
