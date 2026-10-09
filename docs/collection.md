# 동기 RGB 수집과 episode 저장

설정은 [collection_d3.json](../configs/collection_d3.json), episode 저장 설정은 [collection_d4.json](../configs/collection_d4.json)에 있다. [config.py](../carla_tools/config.py)는 누락·중복·unknown field, bool 숫자, NaN/Infinity와 범위 오류를 거부하고 canonical 설정 SHA-256을 계산한다.

## 수집 설정

| 항목 | 기본 설정 |
| --- | --- |
| Simulator·map | CARLA 0.9.16, 현재 실행 중인 Town01 |
| 차량 | `vehicle.tesla.model3`, 지정 spawn, 정지 smoke는 brake 1·handbrake true |
| 시간 | 동기 모드, `fixed_delta_seconds=0.05`, 20Hz |
| Physics substep | `max_substep_delta_time=0.01`, `max_substeps=10` |
| 날씨·RGB | ClearNoon, 640×360, 수평 FOV 90°, `sensor_tick=0` |
| Mount | Actor 기준 `(1.5, 0, 2.4)`m, `(pitch=-12, yaw=0, roll=0)`°, Rigid |
| 렌더링 | Rendering 활성, postprocess false, gamma 2.2, `lens_k=0`, `lens_kcube=0` |
| Sensor queue | 용량 8; overflow·timeout·누락·중복·미래 frame은 실패 |

Nominal pinhole intrinsics는 `fx=fy=width/(2*tan(horizontal_fov/2))`, `cx=width/2`, `cy=height/2`다. 기본 설정에서 `fx=fy=320`, `cx=320`, `cy=180` pixel이며 별도 실측 calibration 값으로 해석하지 않는다. Substep은 `fixed_delta_seconds <= max_substep_delta_time * max_substeps`를 만족해야 한다. Simulation 20Hz는 wall-clock 처리율이 아니다.

[CollectionSession](../carla_tools/session.py)은 설정·날씨·actor 속성을 적용한 뒤 readback한다. 이미 동기 world이거나 다른 차량·보행자·센서가 있으면 진입을 거부하며, 지정 spawn 실패 시 다른 위치를 선택하지 않는다. 현재 Town01을 사용하고 map을 자동 변경하지 않는다.

각 tick의 RGB frame/time을 world snapshot과 연결하고 동일 frame의 차량 pose·velocity·angular velocity·acceleration을 읽는다. Camera image transform과 차량 transform의 상대 matrix로 mount를 검사한다. Control 값은 같은 world frame의 client cache readback이며 출처를 기록한다.

## 저장과 완료 상태

```text
episode_id/
  episode.json
  frames.jsonl
  images/{simulator_frame:08d}.png
```

[EpisodeWriter](../carla_tools/episode.py)는 같은 volume의 임시 PNG를 fsync하고 독점적으로 공개한 뒤 JSONL 행을 추가한다. 기존 destination은 빈 폴더도 덮어쓰지 않는다. 예정 frame 저장, 적용 readback, capture, 센서 정지·owned actor 제거·설정/날씨 복구와 fresh disk integrity가 모두 성공해야 `COMPLETE`가 된다. 중단·실패 episode는 완료 데이터로 사용하지 않는다.

Cleanup은 더 최신 비동기 snapshot과 현재 actor 목록의 owned ID 부재를 확인한다. 캐시에 남을 수 있는 `get_actor(id)`만으로 부재를 판단하지 않는다. Validator는 strict JSON, 설정 hash, frame/time 연속성, PNG CRC·압축 데이터·크기·SHA-256, 경로, pose/matrix, camera mount/calibration와 orphan image를 검사한다.

## 실행

[환경 설치](setup.md) 후 저장소 루트에서 실행한다. 아래 정지 episode는 logger와 데이터 계약 확인용이며 학습용 moving dataset과 구분한다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_collection.py --config configs/collection_d3.json --validate-only
.\.venv\Scripts\python.exe scripts/verify_collection.py --config configs/collection_d3.json --output-dir results/local/collection-probe --runs 2
.\.venv\Scripts\python.exe scripts/collect_episode.py --route-id town01_train_p000 --episode-id sample_train_01 --dataset-version smoke-example --output-root results/local/dataset-example --validate-only
.\.venv\Scripts\python.exe scripts/collect_episode.py --route-id town01_train_p000 --episode-id sample_train_01 --dataset-version smoke-example --output-root results/local/dataset-example
.\.venv\Scripts\python.exe scripts/validate_episode.py results/local/dataset-example/sample_train_01
```

`--validate-only`는 CARLA import·RPC·수집 없이 설정을 검사하고 `CONFIG_VALIDATED`를 반환한다. 실제 실행 결과로 해석하지 않는다. 설정 반복 확인은 같은 config와 초기화 signature를 비교하며 physics 또는 pixel 단위 결정성을 보장하지 않는다. 재수집은 새 로컬 episode 경로를 사용하되 Kaggle notebook은 기존 ID를 재사용한다.

Frame의 라벨 상태와 split 규칙은 [데이터 계약](data-contract.md), raw를 보존한 라벨 생성은 [waypoint 문서](waypoint-labels.md)를 따른다. 모든 실행 출력은 `results/local/`에 보관한다.
