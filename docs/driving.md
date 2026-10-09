# 고정 경로 주행과 초기 오차 복귀 QA

[driving.py](../carla_tools/driving.py), [run_driving_pipeline.py](../scripts/run_driving_pipeline.py)와 [driving_validation_d10.json](../configs/driving_validation_d10.json)은 GT waypoint를 사용하는 privileged controller QA를 제공한다. RGB perception의 정확도나 전체 benchmark 성능 평가와 구분한다.

## 고정 경로와 조건

직선·좌곡선·우곡선의 35m original route를 사용한다. Native 수집 전에 live map/version·OpenDRIVE·route hash와 지정 spawn을 확인한다. Tracker는 같은 frame의 actor pose를 원래 route에 투영하고 이전 reference의 `[-1m,+3m]` 범위 및 물리적 진행 한계를 검사한다. 편차가 생겨도 nearest lane으로 목표를 바꾸지 않는다.

각 route에 `center`, lateral `-0.3m/+0.3m`, yaw `-5°/+5°` 조건을 적용한다. Warmup 뒤 z·pitch·roll을 보존하고 원래 route 첫 anchor와 tangent에 정렬한 후 편차를 적용한다. 첫 실제 frame에서 lateral/yaw가 요청의 0.05m/0.5° 이내인지 확인한다.

| 개발 QA 항목 | 기준 |
| --- | --- |
| 제어 | 20Hz·0.05초, nominal 최대 25km/h, 공통 곡률 속도 계획 |
| 종료 | Absolute reference 19m 이상; 마지막 주행 frame도 15m 미래 label 확보 |
| 실행 제한 | Simulation 최대 10초, 속도 ≤12m/s |
| 횡오차·차체 | 모든 주행 frame에서 abs lateral ≤0.8m, 차체 차선 여유 ≥0.15m |
| 초기 복귀 | Abs lateral ≤0.2m·abs heading ≤3°를 연속 1초 유지하고 유지 완료가 시작 후 5초 이내 |
| 이벤트 | Collision·lane invasion·callback 오류가 없어야 함 |
| 정지 | 마지막 적용 조향 유지, service brake, 2초 안에 속도 ≤0.1m/s |
| 완료 | 정상 episode·disk gate·owned actor 제거·원래 world/weather 복구 |

복귀 조건을 벗어나면 연속 유지 시간을 초기화한다. Heading 3°는 초기 복귀 유지 구간 기준이며 전체 곡선 주행 상한으로 사용하지 않는다. 차체 여유·속도·sensor 기준은 이후에도 적용한다. 미래 horizon을 자르거나 반복하여 종료하지 않는다.

## Frame·sensor·정지·cleanup

RGB는 첫 frame·20 frame 간격·마지막 frame을 저장하고 전체 20Hz pose/control trace를 보존한다. World/snapshot/RGB frame과 timestamp를 맞추며 frame F의 controller 결정은 F+1 applied control과 비교한다. 명령은 `apply_batch_sync([ApplyVehicleControl(owned_vehicle_id, control)], False)`로 전송하고 단일 응답·actor identity·오류와 다음 frame readback을 확인한다. Batch 내부 hidden tick을 만들지 않는다.

차체 bounding box의 여덟 corner를 같은 frame의 3D actor transform으로 world에 옮기고 original route의 각 nearest segment·보간 lane width로 최소 여유를 계산한다. Actor 중심 오차만으로 차체를 판단하지 않으며 bbox 자체 rotation은 거부한다.

Collision/lane-invasion sensor의 원본 이벤트와 callback 오류를 주행·감속·주차·cleanup 후까지 보존한다. `NONE` 또는 width 0 marking도 무시하지 않는다. Callback 진입 시 접수하고 sensor stop/cleanup 뒤 접수된 작업을 최대 2초 기다린다. 미완료는 실패이며 최종 event snapshot·완료 판정·metadata 발행을 같은 잠금으로 처리한다.

정지는 마지막 applied steer를 유지한 `throttle=0, brake=1, hand_brake=false`로 수행한다. 속도 ≤0.1m/s와 최소 10 braking tick 이후 handbrake를 적용하고 2 tick 추가 관찰한다. 모든 stop pose·속도·control·wheel·margin을 재검사한다. Stop 구간은 horizon 소진이 가능하므로 주행 waypoint를 생성하지 않는다. Cleanup은 owned actor의 실제 부재, world/weather 복원과 종료 후 RPC를 확인한다.

## 기존 Kaggle notebook 준비

제출 전에 등록된 기존 notebook의 접근성과 실행 상태를 읽기 전용으로 확인하고 metadata를 pull한다. 동일 ID와 제목을 유지하며 RUNNING이면 종료를 기다린다. 아래 준비 도구는 기존 metadata를 검사하고 bundle만 만든다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_kaggle_driving.py --existing-metadata results/local/kaggle-existing/kernel-metadata.json --output-dir results/local/kaggle-driving-bundle
```

Kaggle CLI로 같은 notebook을 갱신해 실행하고 source artifact ZIP·bootstrap·로그를 `results/local/`로 다운로드한다. Kernel의 `COMPLETE`만으로 QA를 판정하지 않는다. Bootstrap, episode와 disk 검사를 각각 확인하며 bootstrap의 owned server 종료도 확인한다.

## 독립 CPU 확인

Artifact는 `driving/episodes/{case}__{condition}/episode.json`, `steps.jsonl`, `images/*.png`와 manifest/report로 구성한다. 아래 CPU 도구는 CARLA GPU/server를 시작하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_driving.py --manifest results/local/driving-example/driving_manifest.json --output results/local/driving-gate.json
.\.venv\Scripts\python.exe scripts/audit_driving.py --native results/local/driving-example --output results/local/driving-negative-audit.json
.\.venv\Scripts\python.exe scripts/audit_driving_map.py --native results/local/driving-example --output results/local/driving-map-audit.json
```

Validator는 original tracker/controller를 처음부터 replay하고 별도 waypoint geometry, native wheel geometry, applied control과 frame 연결, PNG CRC/hash/color witness, 동일 case의 route identity와 manifest completeness를 검사한다. 판정 직전 source를 다시 읽어 변경 여부를 확인한다.

Negative audit는 정상 source의 임시 복사본에 독립 손상을 적용하여 거부되는지 확인하고 원본 hash 보존을 요구한다. Map audit는 동일 OpenDRIVE와 CARLA 0.9.16 `Map` API로 yaw-only 네 차체 corner가 원래 Driving lane 안인지 독립 확인하고 height 0 supplementary 조회가 판정을 바꾸면 실패한다. OpenDRIVE의 다른 위치는 `--xodr`로 지정한다.

이 QA의 기준은 짧은 privileged reference의 개발 조건이다. 전체 RGB closed-loop 비교의 종료·집계·latency와 held-out 정책은 [평가 설계](evaluation.md)를 따른다. 실행 기록과 출력은 로컬에 보관한다.
