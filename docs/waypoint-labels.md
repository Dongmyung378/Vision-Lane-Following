# Waypoint 라벨 생성과 overlay

[labels.py](../carla_tools/labels.py)는 [데이터 계약](data-contract.md)의 actor-origin waypoint를 생성하고 [projection.py](../carla_tools/projection.py)는 같은 frame의 RGB에 투영한다. Pose·camera·map·route는 생성/QA에 사용하며 RGB 예측기의 입력에 포함하지 않는다.

## Physical arc와 좌표

[geometry_survey.py](../carla_tools/geometry_survey.py)는 dense lane-center XYZ의 chord 길이를 누적하여 physical polyline arc를 만들고 0.25m 간격의 35m corridor를 고정한다. CARLA `next(s)`의 OpenDRIVE reference 간격을 외측 차선의 물리 이동 거리로 대신 사용하지 않는다. 저장 arc는 dense polyline 수치 근사이며 analytic 적분값이 아니다.

같은 frame의 actor 위치를 원래 route에 투영한 reference 이후 3·6·9·12·15m에서 world 목표를 보간한다. Vehicle pose가 `(R, t)`이면 `p_vehicle = R^T * (p_world - t)`로 변환한 x/y가 `5 × 2` 라벨이다. 원래 route/lane identity를 유지하며 부족한 horizon, junction·branch·lane end, XY 자기 교차 또는 모호한 nearest projection을 거부한다.

## Raw 보존과 sidecar

Raw metadata·JSONL·PNG와 미생성 라벨 필드를 수정하지 않는다. 생성 sidecar는 source identity와 원본 byte hash, 순서 있는 frame/time/image identity, `GENERATED` 라벨, route/reference·world/ego 좌표와 geometry diagnostics를 담는다.

`generate_labels(episode_path, output_path)`는 COMPLETE raw의 fresh integrity와 계약을 확인하고 별도 validator로 임시 파일을 검사한 뒤 새 output을 독점적으로 발행한다. 기존 output은 덮어쓰거나 자동 재사용하지 않는다. `validate_labels(episode_path, labels_path)`는 원본·sidecar를 다시 읽고 hash·frame·shape·unit·route·동일 frame transform과 geometry를 재계산한다. Geometry tolerance는 0.001m이며 원본 metadata의 whitespace 변경도 byte hash 변경으로 거부한다.

## CPU 실행

아래는 존재하는 raw episode에 대한 예제다. 생성 output과 overlay 폴더는 raw 밖의 새로운 경로를 지정한다. CARLA 서버나 GPU가 필요하지 않다.

```powershell
.\.venv\Scripts\python.exe scripts/generate_labels.py --episode results/local/dataset-example/sample_train_01 --output results/local/labels-example/train.json
.\.venv\Scripts\python.exe scripts/generate_labels.py --episode results/local/dataset-example/sample_train_01 --labels results/local/labels-example/train.json --validate-only
.\.venv\Scripts\python.exe scripts/render_waypoint_overlay.py --episode results/local/dataset-example/sample_train_01 --labels results/local/labels-example/train.json --output-dir results/local/overlay-example
```

`--frame`을 생략하면 첫 라벨 frame을 그린다. 특정 frame은 원본 JSONL에 존재하는 simulator frame 번호를 지정한다.

## Projection 해석

같은 frame의 camera inverse transform으로 world 목표를 camera-local로 옮긴다. CARLA `(전방, 우측, 위)` 좌표 `(x,y,z)`를 optical `(우측, 아래, 전방)`의 `(y,-z,x)`로 변환하고 nominal intrinsics와 zero lens distortion 계약으로 pinhole projection한다.

Overlay report는 depth·원래 UV·visibility와 frame/time/image/label/route provenance를 보존한다. 영상 안의 점만 그리며 영상 밖의 점은 `OUTSIDE_IMAGE`와 원래 UV를 남긴다. 가까운 목표가 mount/FOV에 따라 영상 밖에 있을 수 있다. UV clamp나 behind-camera 좌표의 부호 반전을 하지 않는다.

Overlay는 geometry를 보여주고 color provenance를 인증하지 않는다. BGRA→RGB witness·PNG integrity와 전체 frame visibility는 별도 native/CPU QA에서 검사한다. 원본에 없는 witness를 사후 생성하여 붙이지 않는다.
