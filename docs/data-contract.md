# 데이터·route·waypoint·split 계약

[dataset_contract_d5.json](../configs/dataset_contract_d5.json)과 [contracts.py](../carla_tools/contracts.py)가 계약을 정의한다. 경로 예측기의 입력은 전방 RGB 한 장이며 pose·camera transform/calibration·map·route·정답 라벨은 생성과 QA용 privileged 정보다. 공통 저수준 controller에는 차량 속도 피드백을 사용할 수 있다.

## 고정 route와 waypoint

Route는 driving lane의 고정 centerline corridor다. 15m 미래 horizon 안의 junction·branch·lane end를 제외하며 topology 조사 corridor는 최소 35m 전체에서 조건을 확인한다. Spawn은 지정 위치를 사용하고 실패 시 fallback하지 않는다. Recovery에서도 원래 route와 scene group을 유지하며 차량 위치의 nearest lane으로 목표를 다시 선택하지 않는다.

Route metadata는 route/scene/town/version, 원래 map package, OpenDRIVE SHA-256, spawn, start-lane identity, 순서 있는 XYZ anchor·arc distance, horizon과 canonical route SHA-256을 연결한다. 허용 map package는 `Carla/Maps/{town}` 또는 `/Game/Carla/Maps/{town}`이며 관측 문자열을 보존한다. Anchor는 driving lane이며 lane ID 0, junction, 다중 successor를 허용하지 않는다.

| Waypoint 항목 | 계약 |
| --- | --- |
| 출력 | `5 × 2`의 `[x, y]` |
| 원점·축·단위 | 동일 frame의 차량 actor transform, x 전방·y 우측, m |
| 거리 기준 | 원래 target centerline의 현재 reference 이후 arc distance 3·6·9·12·15m |
| World QA 값 | `5 × 3`의 XYZ; actor inverse rigid transform으로 ego XY 생성 |
| Geometry tolerance | 0.001m |

Horizon은 ego x 고정값이 아니다. 현재 actor XY를 고정 route에 투영한 reference에서 world 목표를 보간한다. 남은 route가 충분하지 않으면 목표를 자르거나 반복하지 않고 실패한다. Camera 또는 axle 원점으로 저장 라벨을 바꾸지 않는다.

## Raw episode와 derived label

Raw episode는 `episode.json`, `frames.jsonl`, `images/*.png`로 구성한다. Episode는 identity·config/hash·camera calibration·readback·perturbation·frame 수·cleanup·route를, frame은 snapshot/sensor 번호와 시간·image hash·차량/카메라 transform·운동 상태·control을 기록한다.

정지 수집 raw의 미생성 라벨은 `label_state="NOT_GENERATED", label=null`이다. 영점 배열이나 임의 예측값으로 정답을 대신하지 않는다. 생성된 라벨은 raw 밖의 sidecar에 저장하며 raw의 상태와 bytes를 유지한다. [라벨 생성 문서](waypoint-labels.md)를 따른다.

외부 구조 계약은 [frame schema](../schemas/episode_frame.schema.json), [generated waypoint schema](../schemas/waypoint_label.schema.json), [split row schema](../schemas/split_manifest.schema.json)에 있다. Runtime은 duplicate JSON key, nonfinite/overflow, missing field, bool 숫자, 경로 탈출, frame·pose·mount·hash·PNG 불일치를 거부한다. JSON Schema만으로 disk integrity나 실제 simulator 정합을 판단하지 않는다.

## Split manifest

CSV header는 `dataset_version,sample_id,route_id,episode_id,scene_group_id,split`이다. `sample_id`는 `episode_id + ':' + frame의 최소 10자리 숫자`다. Assignment는 episode 단위의 train/dev만 허용하며 같은 route·episode·scene group을 한 split에 둔다. 동일 snapshot의 dataset version, 같은 town의 simulator/OpenDRIVE source와 route별 hash가 일치해야 한다.

같은 town의 다른 split은 corridor XY 선분 전체와 관측 actor origin의 최소 거리를 비교하며 75m 이하이면 거부한다. 가까운 scene의 시간이 겹치거나 1초 이내로 인접하면 temporal leakage도 표시한다. 반복 수집·recovery·반대 방향의 같은 물리 scene은 ID를 바꿔 다른 group으로 나누지 않는다. 긴 moving dataset에는 전체 route 범위의 별도 누수 검사가 필요하다.

```python
from carla_tools.splits import build_split_manifest

report = build_split_manifest(
    episode_paths=["results/local/dataset-example/sample_train_01",
                   "results/local/dataset-example/sample_dev_01"],
    output_csv="results/local/dataset-example/split_manifest.csv",
    assignments={"sample_train_01": "train", "sample_dev_01": "dev"},
)
```

Builder는 COMPLETE episode의 fresh integrity를 먼저 확인하며 이 계약의 raw `NOT_GENERATED` snapshot을 받는다. 실패하면 CSV를 만들지 않는다. 원자적 exclusive publication을 사용하고 기존 동일 내용은 재사용하지만 다른 version·내용은 덮어쓰지 않는다. Town02는 held-out final용으로 예약하며 이 builder는 Town02/final assignment를 거부한다. Final 데이터는 개발·모델 선택에 사용하지 않는다.
