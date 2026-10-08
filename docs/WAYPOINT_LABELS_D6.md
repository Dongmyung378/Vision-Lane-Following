# D6 Waypoint 라벨과 동일 프레임 overlay

작성일: 2026-10-08 (Asia/Seoul). 구현은 [labels.py](../carla_tools/labels.py), [projection.py](../carla_tools/projection.py)에 있다. **D6 완료: 기존 Kaggle notebook version 4의 라벨 생성과 전체 다운로드 후 최종 CPU 재검증이 PASS했다.** Town01의 직선·좌곡선·우곡선에서 각각 60 frame, 총 180 frame을 실제 저장하고, 원본과 분리된 180개 frame 라벨을 검증했다.

## 좌표와 고정 목표 경로

[D5 계약](DATA_CONTRACT_D5.md)을 유지한다. 모델 입력은 RGB이며, 차량 pose·camera transform·map·route는 라벨 생성과 검증에 사용하는 privileged 정보다. 출력은 차량 actor origin 기준의 `5 × 2` 좌표로, x는 전방, y는 우측, 단위는 m다. Camera origin이나 차량 bounding-box 중심으로 바꾸지 않는다.

다섯 타깃은 **원래 목표 차선 centerline의 현재 reference에서 전방 arc distance 3/6/9/12/15m**인 점이다. 곡선에서도 ego x를 3/6/9/12/15로 고정하지 않는다. `reference_arc_m`은 같은 프레임의 차량 위치를 이미 고정된 route polyline에 투영해 구한다. Lateral/yaw 편차가 생겨도 전 도로의 nearest lane을 새 목표로 고르지 않으며, `target_lane`은 원래 route start의 road/section/lane identity를 유지한다.

각 world 타깃을 `p_world`, 같은 프레임 vehicle transform을 `(R_vehicle, t_vehicle)`라 하면 다음처럼 변환하고 x/y 두 값만 모델 타깃으로 저장한다.

```text
p_vehicle = R_vehicle^T × (p_world - t_vehicle)
waypoint_ego = [p_vehicle.x, p_vehicle.y]
```

원래 route의 ID, map package, OpenDRIVE SHA-256와 route SHA-256를 source에 연결한다. Junction·branch·lane end가 라벨 horizon에 들어오거나, 남은 고정 경로가 reference 이후 15m를 제공하지 못하면 생성에 실패한다. XY에서 자기 교차하는 route와 서로 떨어진 arc의 동등한 nearest projection도 거부한다. 이 보수적인 검사는 교차부나 복잡한 loop를 지원하는 moving-route progress tracker를 대신하지 않는다.

## Native 곡선의 physical arc

[geometry_survey.py](../carla_tools/geometry_survey.py)는 Town01의 고정 spawn들을 읽고 직선·좌회전 곡선·우회전 곡선을 선택한다. CARLA의 `next(s)` 인수를 그대로 차선 centerline의 물리 거리라고 기록하지 않는다. OpenDRIVE reference의 0.1m 간격 waypoint들에서 **실제 lane-centerline XYZ 간 chord 길이를 누적한 dense polyline arc**를 만들고, 그 arc의 0.25m마다 world anchor를 보간하여 35m corridor를 고정한다.

따라서 arc는 dense XYZ polyline의 수치 근사이며 analytic curve arc나 실제 연속 도로의 정확한 적분값을 주장하지 않는다. 생성기는 이 route에 기록된 arc를 사용한다. Offline 검증은 라벨이 해당 고정 route와 일치하는지를 확인하고, native 수집은 수집 직전 live map에서 route를 다시 survey하여 같은 route hash인지 확인한다.

## 원본과 derived sidecar

Raw episode의 `episode.json`, `frames.jsonl`, PNG는 수정하지 않는다. Raw metadata의 `labels_state=not-generated-D6`, frame의 `label_state=NOT_GENERATED`, `label=null`도 그대로 유지한다. 실제 라벨은 raw episode 밖의 새로운 JSON sidecar에 `GENERATED`로 저장한다.

| Sidecar 위치 | 저장 내용 |
| --- | --- |
| Header | `schema_version=1`, `status=COMPLETE`, contract/generation version, D6 derived scope |
| `source` | Dataset/episode/route/scene/town identity, config·route·원본 metadata·원본 JSONL SHA-256, frame 수와 첫/끝 frame |
| `rows` | 원본 순서의 index/frame/timestamp/sample ID, image 상대 path/hash, `GENERATED`, label 및 geometry diagnostics |
| `label` | 고정 target lane, ego frame, 좌표·단위·horizon 계약, reference arc, `5 × 3` world 점과 `5 × 2` ego 점 |
| `metrics` | Frame 수와 첫/끝 frame, waypoint shape, world/ego/reference/roundtrip 최대 오차 |

`generate_labels(episode_path, output_path)`는 `COMPLETE` raw의 fresh disk integrity와 D5 계약을 확인한다. 모든 행을 생성한 뒤 별도 validator로 임시 파일을 검사하고 exclusive publication으로 새 파일을 만든다. 기존 output은 동일한 성공 파일이어도 덮어쓰거나 자동 재사용하지 않는다. 재검증은 `validate_labels(episode_path, labels_path)`를 사용한다.

Validator는 raw와 sidecar를 디스크에서 다시 읽고 원본 해시·frame 연속성·image identity·shape·단위·고정 route·동일 프레임 transform·유한 geometry와 diagnostics를 검사한다. 허용 geometry 오차는 D5의 0.001m다. 원본 metadata에 whitespace만 추가해도 byte hash가 달라져 기존 sidecar 검증은 실패한다. 실패한 보고서는 `metadata`, `frames`, `sidecar`를 usable data로 반환하지 않는다. 내부 `.labels-*.pending` 파일은 public validator가 거부한다.

## CPU에서 생성과 재검증

Version 4의 다운로드 위치는 `results/local/kaggle-week1-output-v4/week1`이다. Generator와 validator, overlay는 CARLA server와 GPU를 요구하지 않는다. Label 생성 output과 overlay output은 raw 밖의 **아직 없는 경로**를 사용한다.

```powershell
$week1Artifact = 'results/local/kaggle-week1-output-v4/week1'
.\.venv\Scripts\python.exe scripts\generate_labels.py --episode "$week1Artifact/raw/geometry_straight_01" --output "$week1Artifact/labels/straight_recreated.json"
.\.venv\Scripts\python.exe scripts\generate_labels.py --episode "$week1Artifact/raw/geometry_straight_01" --labels "$week1Artifact/labels/straight.json" --validate-only
.\.venv\Scripts\python.exe scripts\render_waypoint_overlay.py --episode "$week1Artifact/raw/geometry_straight_01" --labels "$week1Artifact/labels/straight.json" --output-dir results/local/week1-overlay-recheck-straight
```

`--frame`을 생략하면 첫 라벨 frame을 그린다. 특정 simulator frame을 확인할 때는 원본 JSONL에 실제 존재하는 frame 번호를 지정한다. CLI의 정상 종료 코드 0은 해당 작업의 PASS이며, native 수집이나 D7 전체 게이트의 PASS를 자동으로 뜻하지 않는다.

## Overlay 해석

World 타깃은 같은 프레임의 `camera.transform.matrix`로 camera-local 좌표에 변환한다. CARLA camera의 `(전방, 우측, 위)`를 optical `(우측, 아래, 전방)`으로 `(x,y,z) → (y,-z,x)` 변환하고, 기록된 nominal intrinsics와 zero lens distortion 계약으로 pinhole projection한다.

Overlay의 `report.json`은 각 점의 depth, 원래 UV, visibility status, frame/time/image/label/route provenance를 남긴다. `overlay.png`에는 영상 안의 점만 그린다. Mount와 FOV에 따라 **3m 타깃이 영상 밖에 있을 수 있다.** 이때 `OUTSIDE_IMAGE`와 원래 UV를 보존하며, 경계 pixel로 clamp하거나 behind-camera 점의 부호를 뒤집지 않는다.

Overlay renderer는 기하 정합을 보여주며 색상 증명을 인증하지 않는다(`color_provenance_certified=false`). 새 native capture의 BGRA witness와 저장 RGB 확인, 전체 frame의 visibility 조건은 [D7 게이트](WEEK1_GATE_D7.md)가 별도로 검사한다. D4/D5 legacy raw에 없는 witness를 사후 생성해 끼워 넣지 않는다.

## 개발 검증과 실제 결과 기록

[test_waypoint_labels.py](../tests/test_waypoint_labels.py)의 15개 로컬 fixture test를 통과했다. 실제 EpisodeWriter로 만든 PNG/JSONL fixture를 사용해 직선, 좌·우 곡선 arc, lateral/yaw 변환, 원본 byte 보존, nonfinite/reflection/잘못된 frame, 부족한 horizon, 자기 교차·모호한 투영, row 누락·중복·순서·source hash 변조, PNG 손상, 기존 output 보호와 게시·정리 실패를 확인했다. 이 fixture 검증을 이번 native 직선/좌곡선/우곡선 수집 결과로 기록하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_waypoint_labels.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_projection.py -v
```

**실제 D6 native 결과: PASS.** 기존 `dongmyungpark/vlf-carla-rgb-smoke-20261005` notebook의 version 4에서 CARLA 0.9.16, Town01, dataset `d6-geometry-v1`로 수집·라벨 생성·독립 검증을 완료했다. 다음 수치는 다운로드한 bootstrap의 case별 label audit에 기록되어 있다. 전체 source/label/config hash와 원격·로컬 검증 결과는 [실행 증거 요약](evidence/carla_week1_d6_d7_2026-10-08_summary.json)에 보존한다.

| Case | 고정 spawn | 실제 frame 범위 | Frame/라벨 수 | World↔ego 최대 roundtrip 오차 |
| --- | --- | --- | --- | --- |
| `straight` | 0 | 1142–1201 | 60 / 60 | 8.653 × 10⁻¹² m |
| `left_curve` | 108 | 1323–1382 | 60 / 60 | 8.539 × 10⁻¹² m |
| `right_curve` | 174 | 1478–1537 | 60 / 60 | 8.550 × 10⁻¹² m |

세 case 모두 라벨 shape은 `5 × 2`이고, 고정 route 대비 world/reference 및 ego 재계산 최대 오차는 0m로 기록됐다. 이는 저장된 route·동일 프레임 pose와 계산 결과의 일치이며 학습 정확도나 주행 오차가 아니다. Raw의 미생성 label 필드와 PNG/JSONL은 그대로 유지했고, 라벨은 별도 sidecar로 저장했다.

실제 native RGB 위의 첫 frame overlay는 [직선](evidence/carla_week1_d6_d7_2026-10-08_straight_overlay.png), [좌곡선](evidence/carla_week1_d6_d7_2026-10-08_left_curve_overlay.png), [우곡선](evidence/carla_week1_d6_d7_2026-10-08_right_curve_overlay.png)에서 확인할 수 있다. 각 case의 모든 60 frame에서 다섯 타깃은 camera 전방에 있었고, 6/9/12/15m 네 점은 영상 안에 있었다. 3m 점은 영상 아래의 `OUTSIDE_IMAGE`로 기록됐으며 clamp하지 않았다. 전체 frame의 색상·projection 검증 결과는 [D7 문서](WEEK1_GATE_D7.md#실제-결과와-다음-단계)에 있다.

**최종 로컬 CPU 재검증: PASS.** PNG를 포함한 version 4 전체 artifact를 다운로드한 뒤 최종 코드로 세 episode의 180개 frame/라벨을 새로 검사했다. 원격 audit과 로컬 audit의 source identity·hash와 label SHA-256가 모두 일치했고 오류는 0건이었다. 별도의 `jsonschema 4.26.0` 검사도 raw frame 180개와 derived label 180개를 통과했다. 스키마 검사와 실제 geometry/color 정합 검증은 서로 다른 검사이며, [증거 요약](evidence/carla_week1_d6_d7_2026-10-08_summary.json)에 함께 기록했다. 이 CPU 검증에는 로컬 GPU나 CARLA server를 사용하지 않았다.

이번 단계는 정지 RGB/pose/label의 기하 정합 검증이다. D8 controller, D13 moving training pilot과 recovery 수집, D14 closed-loop metric 구현·주행 평가는 후속 작업이다. Town02 final test는 여전히 `UNCOLLECTED`/`UNVERIFIED`다.
