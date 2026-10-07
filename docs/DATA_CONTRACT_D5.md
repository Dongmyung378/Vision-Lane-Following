# D5 데이터·route·waypoint·split 계약

작성일: 2026-10-07 (Asia/Seoul). 계약 버전은 `vlf-dataset-d5-v1`이며 [설정](../configs/dataset_contract_d5.json)을 기준으로 한다. D5는 저장된 짧은 정지 episode의 정합·분할과 **향후 waypoint 라벨의 형식**을 검증한다. 실제 frame별 라벨 생성은 D6, 주행·학습 데이터 수집은 이후 단계다. 정지 smoke를 학습 pilot이나 주행 성능으로 해석하지 않는다.

## 입력과 privileged 정보

경로 예측기의 입력은 전방 **RGB 한 장**이다. 차량 자세, camera transform/calibration, 지도, 목표 route 및 정답 라벨은 라벨 생성·검증·평가용 privileged 정보다. 모델 입력에 포함하지 않는다. 속도 피드백은 공통 저수준 controller에 사용할 수 있다. Camera-only 범위는 [PROJECT_SCOPE.md](PROJECT_SCOPE.md)를 따른다.

현재 수집은 CARLA 0.9.16, Town01, 고정 vehicle/camera, ClearNoon, 0.05초 timestep, 정지 제어다. Capture의 `control_mode=stationary_D4_smoke`, episode `labels_state=not-generated-D6`를 유지한다. 모든 frame에는 다음 상태를 기록한다.

```json
{"label_state": "NOT_GENERATED", "label": null}
```

`[[0,0], ...]`, 빈 객체 또는 임의 예측값을 미생성 정답의 대용으로 넣지 않는다. `GENERATED` 상태는 D6에서 실제 라벨을 생성하고 geometry 검증을 통과한 뒤에만 사용한다. D5 split builder는 `NOT_GENERATED` frame만 받아 이번 snapshot의 의미를 고정한다.

## Route의 고정과 제외 규칙

[routes.py](../carla_tools/routes.py)가 현재 map을 **읽기만** 하여 한 목표 차선의 centerline corridor를 조사한다. `next(1m)`에 하나의 successor만 있어야 하며 driving lane이어야 한다. 앞으로 15m 안의 junction, branch, lane end를 제외하고, D5 조사 corridor는 시작점부터 최소 35m 전체에서 같은 조건을 확인한다. 이 검증은 topology 증거이며 frame 라벨 또는 실제 주행 성공 증거가 아니다.

`route_contract`는 episode metadata에 보존하며 다음 필드를 명시한다.

| 필드 | 의미 |
| --- | --- |
| `route_id`, `scene_group_id` | 동일 route 및 물리적 장면의 고정 식별자 |
| `town`, `carla_version` | Map과 simulator/API 버전 |
| `map_package`, `opendrive_sha256` | API가 반환한 원래 package 문자열, OpenDRIVE UTF-8의 SHA-256 |
| `spawn_point_index` | 지정된 spawn 하나; 다른 위치로 fallback하지 않음 |
| `start` | 시작 차선의 `road_id/section_id/lane_id/s`, 위치 및 첫 anchor |
| `horizon_m` | Frame의 미래 경로 범위 15m |
| `min_required_route_m`, `route_length_m`, `sample_step_m` | 최소 35m의 조사 길이, 1m 이하 anchor 간격 |
| `centerline_anchor_sequence` | 순서대로 고정된 지도 centerline anchor |
| `selection_scope` | `D5-topology-only-not-frame-labels` |
| `route_sha256` | 자신을 제외한 전체 route JSON의 canonical SHA-256 |

각 anchor는 `road_id`, `section_id`, `lane_id`, `s`, `world_location{x,y,z}`, `arc_distance_m`, `is_junction=false`, `successor_count=1`를 가진다. `lane_width_m` 및 `world_rotation{pitch,yaw,roll}`은 명시적인 선택 필드다. Arc distance는 0부터 조사 길이까지 증가하며 시작점은 anchor 0과 같아야 한다. Lane ID 0은 driving target으로 허용하지 않는다.

Native Windows API에서 실제 관측한 package는 **`Carla/Maps/Town01`**이다. 이를 `/Game/...`으로 바꿔 관측값처럼 기록하지 않는다. Validator는 정확한 `Carla/Maps/{town}` 및 `/Game/Carla/Maps/{town}` 두 형태만 허용한다. `../` 또는 다른 package는 거부한다.

이번 [route catalog](../configs/routes_d5.json)는 Town01 spawn **0/train**, **4/dev**의 두 corridor를 담는다. 실제 API 조사에서 junction/branch/lane end 조건을 통과했으며 catalog의 anchor 간 최소 XY 거리는 약 **221.11m**다. Map hash, target lane과 route hash를 capture 전 같은 live map에서 다시 확인한다. 이 두 scene이 다양한 직선·곡선이나 unseen-town 성능을 대표한다고 주장하지 않는다.

Recovery에서도 원래 목표 route와 scene group을 유지한다. 차량이 옆 차선에 가까워졌다는 이유로 새 `get_waypoint(vehicle_location)`의 nearest lane으로 목표를 바꾸지 않는다. 지도에서 연결된 road/section이 바뀌더라도 사전에 고정한 corridor를 따라가며 lateral lane change는 허용하지 않는다.

## 향후 생성 라벨의 좌표와 geometry

출력은 `5 × 2`, 순서대로 목표 차선 centerline의 누적 길이 **3·6·9·12·15m** 앞의 `[x,y]`다. 이는 ego x가 3·6·9·12·15로 고정된다는 뜻이 아니다. 곡선에서도 centerline arc distance를 따른다.

- 원점: 해당 simulator frame의 **차량 actor transform**.
- 축: x 전방, y 우측. 거리 단위 m. World anchor는 CARLA x/y/z m이며 회전 metadata는 degree다.
- 기준점: 해당 actor의 XY 위치를 **고정된 target route polyline**에 투영한 `reference_arc_m`.
- World 목표: `reference_arc_m + [3,6,9,12,15]`의 route anchor 사이를 arc distance로 보간한 3D 점.
- Ego 목표: 같은 frame의 rigid vehicle matrix로 world 점을 변환한 x/y. Camera 원점을 사용하지 않는다.

`waypoints_world_m`은 `5 × 3`의 privileged QA 값이고 `waypoints_ego_m`은 학습 target이다. 라벨에는 `ego_frame`, `route_id`, 시작 목표 lane의 `road_id/section_id/lane_id`, coordinate/axis/unit/distance 계약을 함께 기록한다. [waypoint schema](../schemas/waypoint_label.schema.json)는 **생성된 라벨 객체**의 schema다. 미생성 상태의 `null`은 [frame schema](../schemas/episode_frame.schema.json)의 envelope에서 표현한다.

[contracts.py](../carla_tools/contracts.py)는 유한 숫자, 정확한 5개 점, fixed target identity, 같은 frame, route상의 미래 위치, pose에 대응하는 reference 및 world→ego 변환을 확인한다. Geometry 허용 오차는 0.001m다. Synthetic test의 lateral/yaw 변경 사례는 좌표 변환을 검증할 뿐, 실제 recovery 데이터 수집 성공을 뜻하지 않는다. 실제 native interpolation의 QA와 시각 overlay는 D6에서 수행한다. Pure Pursuit 기준점 offset·wheelbase는 D8에서 따로 확인한다.

## Episode와 frame 저장 형식

한 episode는 `episode.json`, `frames.jsonl`, `images/{frame:08d}.png`로 저장한다. Image를 완전히 저장한 뒤 대응 JSONL 행을 추가하고, capture·cleanup·무결성 검사까지 성공한 episode만 `COMPLETE`로 표시한다. 세부 D4 저장 동작은 [episode.py](../carla_tools/episode.py)의 `EpisodeWriter`와 `validate_episode`가 담당한다.

| 레코드 | 필수 정보 |
| --- | --- |
| Episode | Dataset/episode/route/scene/town identity, 전체 수집 config 및 SHA, camera calibration, 적용 readback manifest, perturbation, 예상/저장 frame 수, 완료·cleanup 상태, `route_contract` |
| Frame | `schema_version`, 0-based index, world/snapshot/sensor와 동일한 frame, 동일 timestamp, `delta_seconds` |
| Image | 상대 path, SHA-256, width/height; 실제 PNG 파일과 일치 |
| Vehicle | 같은 frame의 transform/location/rotation/matrix, velocity, angular velocity, acceleration, 적용 control |
| Camera | 해당 sensor frame의 image transform/location/rotation/matrix |
| Label | 현재 `NOT_GENERATED` 및 `null` |

Native logger의 선택 diagnostics도 schema에 명시했다: `tick_to_rgb_ms`, `mount_matrix_max_abs_error`, `rgb`의 dimension/byte/channel/nonconstant 통계, vehicle/camera `actor_id`, vehicle `control_source=client-cache-at-world-frame`. 이 값들도 유한 범위와 identity를 검사한다. 알 수 없는 frame 필드는 자동 허용하지 않는다. 수집 지연은 모델 inference latency가 아니다.

JSON Schema는 외부 도구용 structural contract다. Runtime 검증은 표준 라이브러리만 사용하며 `jsonschema` 설치를 요구하지 않는다. Duplicate JSON key, NaN/Infinity/overflow, missing field, bool-as-number, path escape, 중복·누락 frame, pose/mount·hash·PNG 불일치를 거부한다. Schema를 통과했다는 것만으로 disk integrity나 실제 simulator 확인을 주장하지 않는다.

## Split manifest와 누수 검사

CSV header는 다음 순서다. [Row schema](../schemas/split_manifest.schema.json)는 한 CSV 행의 JSON 표현을 설명한다.

```text
dataset_version,sample_id,route_id,episode_id,scene_group_id,split
```

`sample_id = episode_id + ':' + frame의 최소 10자리 숫자`다. Assignment는 `{episode_id: 'train'|'dev'}`이며 frame별 assignment를 받지 않는다. 모든 지정 episode가 필요하고 미사용 assignment도 거부한다. 동일 manifest의 dataset version은 하나이며 sample·episode identity가 중복되지 않아야 한다.

한 snapshot에서 동일 town의 CARLA version·OpenDRIVE hash는 하나로 일치해야 한다. Route ID가 다르더라도 서로 다른 map source를 조용히 섞지 않는다. 허용된 두 package prefix는 동일한 town을 가리키며 원래 관측 문자열을 보존한다. 동일 route ID는 하나의 route hash를 유지한다.

같은 route·episode·scene group은 반드시 같은 split이다. Central/recovery, 동일 scene의 재수집, 반대 방향에서 본 같은 물리적 장면은 같은 scene group으로 관리한다. ID만 다르게 붙여 누수를 피할 수 없다. 같은 Town의 다른 split끼리 survey corridor의 **XY 선분 전체**와 관측 actor origin을 비교하여 최소 거리가 **75m 이하이면 거부**한다. 두 가까운 scene의 시간 범위가 겹치거나 1초 이내로 인접하면 temporal leakage로도 표시한다. Episode 내부 모든 연속 frame이 함께 이동하므로 temporal window가 경계를 넘지 않는다. 이 buffer는 D5 정지 scene의 보수적인 기준이며, 긴 moving-route dataset의 교차·전체 경로 범위는 이후 snapshot에서 다시 검증해야 한다.

```python
from carla_tools.splits import build_split_manifest

report = build_split_manifest(
    episode_paths=["results/local/dataset-d4-smoke-v1/train_smoke_01",
                   "results/local/dataset-d4-smoke-v1/train_smoke_02",
                   "results/local/dataset-d4-smoke-v1/dev_smoke_01"],
    output_csv="results/local/dataset-d4-smoke-v1/split_manifest.csv",
    assignments={"train_smoke_01": "train", "train_smoke_02": "train", "dev_smoke_01": "dev"},
)
assert report["status"] == "PASS", report["errors"]
```

Builder는 `validate_episode(require_complete=True)`의 fresh disk integrity PASS를 먼저 요구하고 metadata/frame 계약을 다시 검사한다. 실패 시 CSV를 새로 만들지 않는다. 완성된 CSV는 같은 volume의 원자적 exclusive publication으로 발행한다. 기존 동일 내용은 재사용하고 dataset version 또는 내용이 다르면 **덮어쓰지 않는다**. 새 snapshot은 다른 destination과 version을 사용한다. Windows managed sandbox가 hard link를 차단하는 환경에서는 실제 파일 테스트·실행에 workspace 범위의 sandbox 외 실행이 필요할 수 있다.

Town02는 향후 final test용 **예약 명세**일 뿐이다. 현재 `UNCOLLECTED`/`UNVERIFIED`이며 final sample·frame·성능은 없다. D5 builder는 Town02와 final-test assignment를 거부한다. D42 개발 동결 전에는 final 데이터를 모델 선택에 쓰지 않는다.

## 검증과 후속 단계

이번 실제 로컬 실행은 **3개 COMPLETE episode, 180 RGB frame**을 저장하고 fresh disk integrity 및 split 검증을 통과했다. 동일 target scene의 central 정지 반복 2개는 train 120 frame, 별도 scene은 dev 60 frame이다. 실제 actor 위치까지 포함한 cross-split 최소 거리는 **220.749m**로 75m buffer를 넘었다. Capture·cleanup 및 종료 후 RPC가 통과했고 owned server를 종료했다. Recovery, waypoint generation 및 final data를 수집한 결과가 아니다.

실제 QA에서 같은 route의 train/dev assignment는 실패하고 CSV를 만들지 않았다. 기존 manifest를 충돌하는 내용으로 덮어쓰려는 실행과 기존 episode 재수집도 거부되어 원본이 보존됐다. 별도 복사본 PNG 한 byte를 변경한 경우 integrity validator가 실패했다. [실제 D4/D5 요약](evidence/carla_dataset_d4_d5_2026-10-07_summary.json)과 [실제 split manifest](evidence/carla_dataset_d4_d5_2026-10-07_split_manifest.csv)에 작은 증거를 보존한다. 이 작업에서 Kaggle 실행을 제출하거나 새 notebook을 생성하지 않았다.

계약 tests는 native package 형식, topology exclusion, missing/duplicate/nonfinite/path errors, 같은 pose와 target route의 geometry, recovery 좌표 변환 및 미생성 `null`을 검증한다. Split tests는 실제 persisted PNG/JSONL fixture로 complete integrity, route/scene/time/spatial leakage, 75m 경계, 반대 방향, version/content conflict를 검증한다. 이 synthetic fixture는 실제 capture 증거와 구분한다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_data_contracts.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_splits.py -v
```

D6에서는 이 계약을 사용해 실제 직선·곡선 frame 라벨을 생성하고 시각 검증한다. D5에서 waypoint generator, 학습 모델, controller 또는 평가 metric 계산기는 구현하지 않았다. [평가 초안](EVALUATION_SPEC_D5.md)은 종료·집계 의미를 개발 전에 검토하기 위한 문서이며 threshold는 아직 provisional이다. Kaggle를 사용하게 되면 [AGENTS.md](../AGENTS.md)의 기존 ID·제목을 유지하고 제출 전 읽기 전용 접근 확인을 수행한다. 이번 계약 작업은 새 notebook을 만들지 않는다.

공식 동작 근거: [CARLA 0.9.16 Map/Waypoint](https://carla.readthedocs.io/en/0.9.16/core_map/), [frame/timestamp와 센서 좌표](https://carla.readthedocs.io/en/0.9.16/ref_sensors/), [동기화와 timestep](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/). 수집 규모·split buffer·tolerance는 프로젝트 설계 선택이다.
