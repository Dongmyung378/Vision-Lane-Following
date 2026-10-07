# D3 고정 차량·카메라와 동기 수집 설정

작업일: 2026-10-07 (Asia/Seoul). [D2 환경](ENVIRONMENT_D2.md)의 CARLA 0.9.16과 프로젝트 `.venv`를 사용한다. D3의 목표는 고정된 차량·카메라·날씨·simulation 주기를 설정하고, 같은 초기화와 calibration을 반복 적용할 수 있는지 실제로 확인하는 것이다.

**D3 gate: PASS.** 로컬 CARLA에서 같은 설정으로 2회 실행하여 측정 RGB 40프레임의 정합과 calibration을 확인했다. 각 실행의 actor 정리·날씨·world 설정 복구, 종료 후 RPC 및 서버 종료도 통과했다. 단위 검증은 123개 PASS, 0 skip이다.

## 설정 계약

수집 설정은 [configs/collection_d3.json](../configs/collection_d3.json), 검증과 nominal calibration은 [carla_tools/config.py](../carla_tools/config.py)에 있다. 모든 JSON 필드를 명시해야 하며 누락·중복·알 수 없는 키, bool을 숫자로 사용하는 값, NaN/Infinity 및 잘못된 범위를 거부한다. 설정 SHA-256은 정규화한 유효 값으로 계산하므로 파일의 공백이나 키 순서에 의존하지 않는다.

| 항목 | D3 기본값 |
| --- | --- |
| 서버·map | CARLA 0.9.16, `127.0.0.1:2000`, 이미 실행 중인 Town01 |
| 차량 | `vehicle.tesla.model3`, map spawn point index **0**, 검정, `role_name=vlf_ego` |
| 정지 제어 | Throttle 0, steer 0, brake 1, hand brake true, reverse false |
| 날씨 | ClearNoon |
| 동기 주기 | `synchronous_mode=true`, `fixed_delta_seconds=0.05` (20 simulation Hz) |
| Physics substep | 활성, `max_substep_delta_time=0.01`, `max_substeps=10` |
| RGB | `sensor.camera.rgb`, 640×360, 수평 FOV 90°, `sensor_tick=0` |
| 카메라 mount | 차량 기준 위치 `(1.5, 0, 2.4)` m, 회전 `(pitch=-12, yaw=0, roll=0)` °, Rigid attachment |
| 렌더링 | `no_rendering_mode=false`, postprocess 효과 false, gamma 2.2, `lens_k=0`, `lens_kcube=0` |
| 검사 구간 | 실행당 warmup 10프레임 + 측정 20프레임, queue capacity 8, 기본 반복 2회 |
| Seed | 42, 고정 설정의 식별 기록; 무작위 actor를 추가하지 않음 |

Substep은 `fixed_delta_seconds <= max_substep_delta_time × max_substeps`를 만족해야 한다. 0.05초는 simulation 시간 간격이며 실제 wall-clock FPS를 뜻하지 않는다. [CARLA 동기화와 substep](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/).

## Nominal camera calibration

Pinhole 모델에서 `fx = width / (2 × tan(horizontal_fov / 2))`, `fy=fx`, `cx=width/2`, `cy=height/2`를 사용한다. D3 값은 pixel 단위로 `fx=320`, `fy=320`, `cx=320`, `cy=180`이다.

```text
K = [320   0 320]
    [  0 320 180]
    [  0   0   1]
```

이는 설정에서 계산한 nominal intrinsics이며 별도의 실측 calibration 결과가 아니다. Image 좌표는 u가 우측, v가 아래다. Mount의 위치는 차량 actor 기준 m이며 CARLA 좌표는 x 전방·y 우측·z 위다. Lens distortion 계수와 postprocess를 명시하여 pinhole 가정을 사용할 설정을 고정한다. [CARLA RGB 센서 정의](https://carla.readthedocs.io/en/0.9.16/ref_sensors/#rgb-camera).

[CollectionSession](../carla_tools/session.py)은 world 설정·날씨·actor 속성을 실제 API로 다시 읽는다. Tick 이후 차량 제어와 카메라의 차량 기준 상대 matrix도 확인한다. 검증 실행은 같은 frame의 차량 snapshot과 `image.transform`으로 상대 mount를 비교하며, Euler angle을 단순히 빼서 판단하지 않는다.

## 실행 전제와 보호 규칙

기존 Town01을 사용하며 `load_world()`와 map reload를 호출하지 않는다. D2에서 map 재로드 중 서버가 종료된 원인은 미확인이다. 다른 map이면 자동 map 변경을 하지 않고 실패한다.

이미 synchronous mode이거나 다른 차량·보행자·보행자 controller·센서가 있으면 session 진입을 거부한다. 사용 중인 world의 작업을 먼저 정상적으로 마친 뒤 이 검증을 실행한다. 고정 spawn point가 점유되거나 충돌하면 다른 위치로 fallback하지 않는다. 이 검사 client만 tick을 보내야 한다.

Session 종료 시 자신이 만든 센서·차량만 정리하고 원래 날씨와 world 설정을 복구한다. 복구 호출이 반환됐다는 사실만으로 성공을 표시하지 않는다. 설정·날씨 readback과 더 최신 asynchronous snapshot 및 현재 actor 목록에서 owned ID 부재를 확인한다. 대기는 연결 timeout 안에서 끝나며 오류는 `cleanup`에 기록한다. Capture 성공 후 cleanup이 실패해도 최종 결과는 FAIL이다.

## PowerShell 재현 순서

저장소 루트에서 다음 순서로 실행한다. 서버 준비가 늦으면 [D2 시작·진단 절차](ENVIRONMENT_D2.md#2-서버-시작과-상태-확인)를 따른다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Start
.\.venv\Scripts\python.exe scripts/verify_collection.py --config configs/collection_d3.json --validate-only
.\.venv\Scripts\python.exe scripts/verify_collection.py --config configs/collection_d3.json --output-dir results/local/collection-d3 --runs 2
.\.venv\Scripts\python.exe scripts/check_carla_connection.py --ticks 3 --output results/local/carla_connection_after_d3.json
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Stop
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Status
```

[verify_collection.py](../scripts/verify_collection.py)의 기본 config는 저장소의 `configs/collection_d3.json`, 기본 출력 폴더는 `results/local/collection-d3`, 기본 `--runs`는 **2**다. `--runs`는 1~10을 허용하지만 D3의 반복 확인에는 2회 이상이 필요하다.

`--validate-only`는 CARLA import나 RPC 없이 JSON과 calibration을 검증한다. 성공 출력은 **CONFIG_VALIDATED**, `simulator_verified=false`이며 종료 코드 0이다. 이것을 실제 simulator PASS로 해석하지 않는다. 실제 모드의 성공은 PASS·종료 코드 0, 검증 실패는 FAIL·종료 코드 1이다. 잘못된 CLI 인자는 종료 코드 2다. 실제 연결 전에 보고서를 `RUNNING`으로 기록하므로 중단된 재실행이 이전 PASS를 남기지 않는다. `RUNNING`은 완료 증거로 사용할 수 없다.

실제 실행은 유한 queue에 센서 이미지를 받아 world tick·snapshot·image의 frame과 timestamp를 연결한다. 같은 frame의 출력, 연속 frame, 0.05초 시간 증가, image 크기·FOV, 유효 RGB 및 mount matrix를 확인한다. Queue overflow, timeout 또는 불일치는 실패로 남긴다.

기본 2회 실행의 산출물은 다음과 같다. 같은 출력 폴더를 재사용하면 관리 대상 preview를 지우고 새 보고서를 기록하므로, 비교용 실행은 별도 출력 폴더로 보존한다.

```text
results/local/collection-d3/
  collection_report.json
  collection_preview_run_01.png
  collection_preview_run_02.png
```

보고서에는 설정·hash·nominal intrinsics, 실제 적용 manifest, frame별 검사값, 각 실행의 cleanup 및 초기화 signature가 있다. D3 gate는 최종 `status=PASS`, 2회 이상 모든 실행·cleanup PASS 및 `configuration_repeats_verified=true`를 요구한다. Actor ID·시각·차량 settling pose는 반복 signature에서 제외된다.

## 재현의 범위와 D4 인계

이번 재현의 범위는 **동일 설정·고정 spawn·calibration 및 실제 초기화 readback**이다. Strict physics determinism 또는 pixel 단위 이미지 일치를 주장하지 않는다. 공식 determinism 조건에는 동기 설정 후 world reload와 반복별 reload가 포함되지만, 현재 Windows 환경의 미해결 map reload 문제 때문에 이 절차를 적용하지 않는다. [CARLA physics determinism 조건](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#physics-determinism).

D3의 frame별 검사는 짧은 정지 시험용이다. 모든 원본 이미지와 차량 자세를 함께 저장하는 episode logger, waypoint 라벨, 데이터 분할 및 주행은 완료 범위에 포함하지 않는다. D4에서 이미지·snapshot·자세를 frame ID로 연결해 보존하는 logger를 구현한다.

이번 D3은 로컬에서 검증했고 Kaggle 작업을 제출하지 않았다. Kaggle가 필요해지면 [AGENTS.md](../AGENTS.md)의 기존 3개 ID·제목을 유지하고 제출 전 읽기 전용 접근 확인을 수행한다. 날짜나 일차를 바꾸어 새 노트북을 만들지 않는다.

## 실제 결과

| 항목 | 실제 결과 |
| --- | --- |
| 실제 반복 | 2회 PASS, 같은 config 및 초기화 signature |
| 측정 frame | Run 1: 112960~112979, Run 2: 113020~113039; 각 warmup 10 + 측정 20 |
| 시간 정합 | Tick·snapshot·RGB frame 일치, sensor/snapshot timestamp 일치, 간격 약 0.050000000745초 |
| Calibration | 크기·FOV·속성 일치, 같은 frame 상대 mount matrix 최대 오차 약 `1.05e-5` (허용 `0.001`) |
| Actor·world·weather | 매 실행 sensor stop·destroy·현재 ID 부재·원래 전체 설정과 날씨 복구 PASS |
| 종료 후 RPC | API·server 0.9.16, Town01, asynchronous tick 3회 PASS |
| 서버 종료 | 소유 process 종료, state 파일 제거, port 2000·2001 listener 없음 |
| 로컬 검사 | 단위 검증 123 PASS·0 skip, `pip check`, strict 정적 환경 사전 검사 PASS |
| D3 gate | **PASS** |

[실제 반복 보고서](evidence/carla_collection_d3_2026-10-07.json), [종료 후 RPC](evidence/carla_collection_connection_after_d3_2026-10-07.json), [D3 요약](evidence/carla_collection_d3_summary_2026-10-07.json)에 근거를 보존했다. Config SHA-256은 `4063be7db617865045a92ba3b73399ea3b747a4d59d713bd9950979724800e78`이고 두 실행의 초기화 signature는 `be2e3207e114988c7434019ea6b0afb23cf38110ddd0f8954d99c37a70bf1641`이다.

측정 loop는 각 simulation 1초를 wall-clock 0.3470 / 0.3392초에 확인했다(약 57.63 / 58.96 FPS). RGB·기하 검사는 포함하고 warmup·PNG 저장·GPU snapshot은 제외하며, 주행 모델의 처리율이 아니다. 정지 제어를 적용해도 spawn 직후 차량 서스펜션의 안정화로 초기 속도가 최대 약 0.0915m/s였고 측정 후반에는 0이었다. GPU 전체 장치 메모리 snapshot은 실행 전후 4,074MiB of 6,144MiB였으며 process별 사용량이나 peak는 아니다.

첫 두 실행은 RGB·설정 검사 후 정리 판정에서 실패했다. [최초 기록](evidence/carla_collection_cleanup_initial_2026-10-07.json)과 [원인 확인 기록](evidence/carla_collection_cleanup_cache_2026-10-07.json)을 보존했다. CARLA의 `get_actor(id)`는 제거된 actor 설명도 캐시에 유지하므로 부재 판정에 사용할 수 없다. 더 최신 snapshot의 `find(id)`와 인자 없는 `get_actors()`의 현재 ID 목록을 함께 확인하도록 수정하고 회귀 테스트를 추가했다. 수정 후 2회 실행이 모두 통과했다. [공식 actor 캐시 구현](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/client/detail/CachedActorList.h#L23-L27).

![첫 번째 실행의 실제 RGB](evidence/carla_collection_d3_2026-10-07_run01.png)

[두 번째 실행 이미지](evidence/carla_collection_d3_2026-10-07_run02.png)도 직접 확인했다. GitHub Actions는 기존 계정 제한으로 실행되지 않았으며 로컬 검증과 구분한다.
