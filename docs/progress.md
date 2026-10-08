# 진행 기록

작업 일차와 실제 날짜를 함께 기록한다. 완료 여부는 산출물과 검증에 근거하며, 시간 경과나 설치 파일 존재만으로 다음 단계의 성공을 표시하지 않는다. 전체 계획은 [ROADMAP.md](ROADMAP.md)에 있다.

**현재 D1~D7 완료.** 다음 단계는 D8 Pure Pursuit controller다. D6/D7은 실제 Kaggle native 실행과 전체 다운로드의 CPU 재검증을 통과했으며, 주행·학습 성능은 후속 단계에서 검증한다.

## D1 프로젝트 범위와 개발 환경 조사

- 실제 날짜: **2026-10-05 (Asia/Seoul)**
- 상태: **D1 범위 정리 및 환경 조사 완료**
- 목표: Camera-only 입력 정의, 첫 실행 범위, 출력 계약 및 하드웨어·작업 환경 확인

### 완료한 작업과 증거

| 작업 | 산출물·검증 근거 |
| --- | --- |
| 경로 예측기 입력을 RGB로 제한하고 속도 피드백의 사용 범위 명시 | [PROJECT_SCOPE.md](PROJECT_SCOPE.md), 저장소 README |
| 고정 차량·카메라, 분기 없는 route 및 초기 무교통 조건 확정 | [PROJECT_SCOPE.md](PROJECT_SCOPE.md) |
| Ego waypoint의 단위·축·거리 정의와 frame 정합 원칙 정리 | [PROJECT_SCOPE.md](PROJECT_SCOPE.md) |
| B0/M1의 공통 controller 비교와 privileged reference 구분 | [PROJECT_SCOPE.md](PROJECT_SCOPE.md) |
| 실제 호스트 환경 조사 및 확인되지 않은 항목 기록 | [ENVIRONMENT_D1.md](ENVIRONMENT_D1.md) |
| 환경 진단을 다시 실행할 수 있는 진입점 제공 | [doctor.py](../scripts/doctor.py), [Windows 진입점](../scripts/doctor.ps1) |
| 실패 상황과 잘못된 준비 완료 판정 검증 | [단위 검증](../tests/test_doctor.py) 17개 통과 |
| Windows 및 Linux 자동 검증 구성 | [GitHub Actions 설정](../.github/workflows/ci.yml); 원격 실행 결과는 별도 확인 |
| GitHub 저장소 설명과 Topics 설정 | About 설명과 Topics 8개를 적용하고 API로 재조회 확인 |

### 실행한 검증

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 scripts/doctor.py --output results/local/environment.json
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/doctor.ps1 -OutputPath "results/local/report with spaces.json"
py -3.12 scripts/doctor.py --strict
```

단위 검증 17개, 실제 환경 조사, 공백이 있는 출력 경로의 Windows 진입점 실행이 통과했다. 일반 진단의 종료 코드는 `0`이며, 현재 준비되지 않은 CARLA API와 server에 대한 strict 사전 검사는 예상대로 `1`이었다. 실제 JSON의 RAM은 63.85GiB, GPU 메모리는 6,144MiB이고 simulator 연결 여부는 `false`다. PowerShell syntax와 문서의 로컬 링크도 확인했다.

GitHub About 설명은 `Camera-based lane following in CARLA: RGB-to-waypoint prediction, vehicle control, and reproducible closed-loop evaluation.`로 설정했다. Topics는 `carla`, `lane-following`, `computer-vision`, `deep-learning`, `waypoint-prediction`, `autonomous-driving`, `simulation`, `python`이다. 원격 설정을 다시 읽어 적용 여부를 확인했다.

### Gate 판정

**D1 gate: 통과.** 범위·입력·출력 계약과 환경 조사 기록을 확보했다. CARLA 연결, 영상 수집, GPU 학습 및 주행 성능은 이후 단계의 검증 항목이다. 실제 simulator 준비 상태와 제약은 환경 기록을 기준으로 판단한다.

큰 dataset, CARLA 배포 패키지 및 checkpoint 원본은 Git에 넣지 않고 외부 경로·버전·checksum으로 추적한다. 향후 기록도 실행 명령, 설정 및 결과 경로를 함께 남긴다.

### D2에서 진행할 작업

1. CARLA 0.9.16 Windows 배포와 현재 호스트의 호환성을 확인하고 설치 경로를 정한다.
2. 지원 Python 환경을 선택하고 simulator와 동일 버전의 Python API를 준비한다.
3. Simulator를 실행해 짧은 client 연결을 수행하고 client/server 버전 일치를 검증한다.
4. 실제 사용한 Python·API·server 버전, 설치 경로 및 연결 결과를 기록한다.
5. 연결 오류가 있으면 원인과 재현 명령을 남긴 뒤 해결한다. 연결 성공 전에는 D2 완료로 표시하지 않는다.

## D1 이후 추가 검증: Kaggle T4와 CARLA

- 실제 날짜: **2026-10-05 (Asia/Seoul)**
- 상태: **원격 CUDA 학습 및 CARLA RGB 실행 통과**
- 실제 작업: 공식 Kaggle CLI, 비공개 T4 작업, CARLA RGB 작업 version 2
- 결과: CARLA API/server 모두 0.9.16, Town01, RGB 640×360 20프레임의 유효 픽셀 및 frame 번호 정합 확인
- 해결: 기본 GLX Vulkan 경로 대신 기존 EGL ICD 사용, Python 3.13과 분리한 3.12 client, 공식 다운로드 헤더 명시
- 검증: 전체 단위 검증 36개 통과, 실제 원격 JSON·PNG 내려받아 확인, 제출한 작업 모두 종료
- 제한: 정지 상태의 짧은 실행이며 주행 모델·controller·장시간 성능과 로컬 6GiB 실행 여부는 미검증. 종료 구간의 Signal 11 로그는 후속 안정성 점검 대상
- 산출물: [Kaggle CARLA 검증 기록과 실제 증거](KAGGLE_CARLA_VERIFICATION.md)

이 기록은 사용자가 요청한 원격 실행 가능성의 선행 검증이다. 로컬 D2 구성이 완료됐거나 이후 주행·학습 단계가 완료됐다고 표시하지 않는다.

사용자 요청에 따라 Kaggle 작업은 이미 생성한 노트북 3개를 재사용한다. 이후 날짜·일차·실험별로 새 노트북을 만들지 않고 같은 ID에 실행 버전을 추가한다. 자세한 운영 원칙은 [작업 지침](../AGENTS.md)과 [검증 기록](KAGGLE_CARLA_VERIFICATION.md)에 있다.

## D2 로컬 CARLA 환경과 실제 연결 검증

- 실제 날짜: **2026-10-06 (Asia/Seoul)**
- 상태: **D2 gate PASS**, 로컬 Windows 서버·API 연결 및 짧은 RGB 시험 통과
- 목표: 프로젝트 가상환경, 동일 버전 Windows 패키지·API 및 실제 simulation 진행 검증

### 완료한 작업과 증거

| 작업 | 실제 결과·산출물 |
| --- | --- |
| 프로젝트 환경 고정 | CPython 3.12.7 x64 `.venv`, hash 고정 CARLA 0.9.16 API, native import 및 `pip check` 통과 |
| Windows 배포 설치 | `carla/CARLA_0.9.16`, ZIP SHA-256 manifest 일치 및 추출 완료; [설치 기록](evidence/carla_windows_install_2026-10-06.json) |
| 실제 버전과 simulation 검증 | Package·client·server 0.9.16, 실제 Town01, asynchronous tick 3회; [RPC 기록](evidence/carla_windows_connection_2026-10-06.json) |
| 시작·상태·종료 진입점 | 소유 process의 PID·경로·생성 시각 검증, TCP listener 최대 45초 대기, 시작과 RPC 최종 재실행 약 5.84초 |
| 실제 RGB | 640×360 20장, frame 22094~22113 모두 정합, 유효 RGB 및 PNG; [RGB 기록](evidence/carla_windows_rgb_2026-10-06.json), [도로 이미지](evidence/carla_windows_rgb_2026-10-06.png) |
| 정리 및 재검증 | 시험 actor 정리, asynchronous world 복구, RGB 종료 후 RPC PASS; 서버 종료 후 상태 파일 제거 및 두 port listener 없음 |
| 로컬 품질 검사 | 단위 검증 70개 PASS·0 skip, PowerShell syntax, strict 환경 사전 검사 및 `pip check` PASS |
| 재현 안내·요약 | [ENVIRONMENT_D2.md](ENVIRONMENT_D2.md), [실제 D2 요약](evidence/carla_windows_d2_2026-10-06.json) |

### 실행과 측정

설치·시작·RGB·재검증·종료 명령은 [D2 재현 순서](ENVIRONMENT_D2.md#실제-결과와-다음-단계)에 정리했다. Local RGB는 `--reuse-current-world`로 새로 시작한 Town01을 사용했다. API·서버 버전과 map은 설정 문자열이 아닌 실제 RPC 값으로 확인했다.

수집 loop는 정지 차량의 simulation 1초, 측정 20프레임에서 0.345313초·57.919 FPS였다. RGB 검사를 포함하고 warmup·PNG 저장·GPU snapshot은 제외한 값이며 주행이나 모델의 처리율이 아니다. 전체 GPU 장치의 수집 전후 snapshot은 3,156 / 3,172MiB of 6,144MiB로, process 사용량이나 peak가 아니다.

### 해결한 문제와 남은 제약

- Windows Shipping이 positional Town01 argument를 반영하지 않고 Town10HD_Opt를 로드했다. Project 내부 배포의 `DefaultEngine.ini`에서 공식 기본값만 Town01으로 변경하고 원본을 보관했다. 실제 Town01 RPC·RGB를 재검증했다. AppData는 수정하지 않았다.
- 초기 RPC 호출은 서버 준비 전 timeout이었다. 두 TCP listener를 최대 45초 기다린 뒤 별도 RPC 검사로 연결을 검증하도록 보완했다.
- 이전 `load_world("Town01")` 중 서버가 종료됐다. [실패 JSON](evidence/carla_windows_reload_failure_2026-10-06.json)을 보존했으며 근본 원인은 미확인이다. 새 Town01을 재사용하는 RGB 시험은 통과했지만 map 재로드는 추가 검증이 필요하다.
- DX12 소스·배포 기본값은 확인했으나 실제 engine RHI 로그는 확보하지 못했다. 소유 process 종료는 통과했으며 graceful engine shutdown은 미검증이다.
- 짧은 정지 RGB 시험으로 장시간 안정성, waypoint 라벨, controller 및 학습 모델의 주행을 주장하지 않는다. GitHub Actions는 계정에서 비활성화되어 원격 CI가 실행되지 않았다.
- 새 Kaggle 작업이나 노트북은 생성하지 않았다. 기존 3개 ID와 재사용 원칙을 유지한다.

### Gate 판정과 D3 인계

**D2 gate: 통과.** Windows 설치, 프로젝트 API import, 실제 버전 일치 및 tick 진행을 검증했다. RGB 20프레임은 추가 검증으로 기록한다. 실패와 검증 범위를 함께 보존했으며 설치 ZIP hash는 다운로드 파일의 동일성 확인용이다.

D3에서는 고정 차량·전방 카메라, synchronous mode 및 0.05초 timestep을 재실행 가능한 config로 구현한다. 이후 frame별 자세 기록, 라벨 생성 및 정합 검증을 진행하고 장시간 실행과 map 재로드 제약도 추적한다.

## D3 고정 차량·카메라와 동기 수집 설정

- 실제 날짜: **2026-10-07 (Asia/Seoul)**
- 상태: **D3 gate PASS**, 실제 고정 설정 2회 반복·RGB 40프레임·복구 검증 완료
- 목표: 고정 Model3·전방 RGB·ClearNoon·0.05초 동기 주기를 재실행 가능한 config로 명시하고 실제 적용과 복구 검증

### 구현 산출물과 실제 검증

| 작업 | 산출물·검증 근거 |
| --- | --- |
| 고정 수집 설정 | [collection_d3.json](../configs/collection_d3.json): Model3 spawn index 0, ClearNoon, RGB 640×360·FOV 90°, Rigid mount `(1.5,0,2.4)m`, pitch -12° |
| 설정 검증·nominal calibration | [config.py](../carla_tools/config.py): strict JSON, normalized SHA-256 및 pinhole intrinsics `(320,320,320,180)` |
| Session 소유와 복구 | [session.py](../carla_tools/session.py): 외부 sync·dynamic actor·센서 거부, 고정 spawn, 적용 readback, owned actor·weather·settings 복구 |
| 반복 실행과 실제 frame 검사 | [verify_collection.py](../scripts/verify_collection.py): 기본 2회, image·snapshot·frame·timestamp·mount·RGB 및 cleanup 검사 |
| 재현 안내 | [COLLECTION_D3.md](COLLECTION_D3.md) |
| 실제 반복 검증 | [실제 JSON](evidence/carla_collection_d3_2026-10-07.json): 2회 PASS, 각 20프레임, 같은 초기화 signature |
| 종료 후 진행과 정리 | [RPC JSON](evidence/carla_collection_connection_after_d3_2026-10-07.json): 비동기 3 tick PASS; 서버 종료 및 port 2000·2001 해제 |
| 로컬 품질 검사 | 단위 검증 123 PASS·0 skip, `pip check`와 strict 환경 사전 검사 PASS; [D3 요약](evidence/carla_collection_d3_summary_2026-10-07.json) |

`--validate-only`의 CONFIG_VALIDATED는 simulator 성공이 아니다. 실제 2회 capture·cleanup PASS, 동일 초기화 signature, 실행 후 asynchronous RPC와 종료 결과에 근거해 **D3 gate 통과**로 판정했다. Run 1 frame 112960~112979, Run 2 frame 113020~113039이며 0.05초 timestep·RGB·snapshot·카메라 mount 정합을 검증했다. [Run 1 PNG](evidence/carla_collection_d3_2026-10-07_run01.png), [Run 2 PNG](evidence/carla_collection_d3_2026-10-07_run02.png)를 직접 확인했다.

실제 시험에서 `get_actor(id)`의 제거된 actor 설명 캐시로 정리 판정이 실패했다. [실패와 원인 기록](evidence/carla_collection_cleanup_cache_2026-10-07.json)을 보존하고 더 최신 snapshot 및 현재 actor 목록으로 부재를 검사하도록 수정했다. 해당 회귀 검증과 native 재실행이 통과했다. GPU 전체 장치 메모리는 반복 실행 전후 4,074MiB of 6,144MiB였으며 peak 또는 process별 사용량은 아니다. 원격 CI는 기존 GitHub 계정 제한으로 실행되지 않았다.

### 범위와 다음 작업

현재 Town01을 재사용하고 `load_world()`를 호출하지 않는다. Windows map 재로드 종료의 원인은 미확인이며, 같은 설정·고정 spawn·calibration의 재현만 검증한다. Strict physics determinism, pixel 동일성, 장시간 주행 및 D4 episode logger의 완료를 주장하지 않는다.

D4에서는 같은 frame의 이미지·차량 snapshot·자세를 보존하는 episode logger를 구현한다. 이번 D3은 로컬에서 진행하며 기존 Kaggle 3개 노트북 재사용 원칙을 유지한다.

## D4 이미지·차량 자세 episode logger

- 실제 날짜: **2026-10-07 (Asia/Seoul)**
- 상태: **D4 gate PASS**
- 구현: bounded sensor queue, 같은 frame의 snapshot pose·kinematics와 RGB transform, PNG·JSONL·metadata 저장, 적용·cleanup·무결성에 따른 완료/실패 상태 관리
- 실제 검증: 로컬 CARLA 0.9.16의 에피소드 3개에서 RGB 180장·snapshot 180개 정합 및 independent disk validation PASS. 모든 session cleanup과 수집 후 asynchronous RPC 3 tick PASS, 소유 서버 종료 및 두 port 해제 확인
- 실패 검사: 기존 episode 재수집 거부·원본 불변, 복사본 PNG 한 byte 손상 거부
- 산출물: [D4 재현 안내](EPISODE_LOGGER_D4.md), [설정](../configs/collection_d4.json), [실제 요약](evidence/carla_dataset_d4_d5_2026-10-07_summary.json)

전체 데이터는 `results/local/dataset-d4-smoke-v1/`에 보존하고 검토용 metadata·첫/마지막 행·PNG 2장만 Git에 저장한다. 정지 상태의 짧은 logger 시험이며 학습용 주행 데이터, 라벨·controller 또는 장시간 안정성 완료로 해석하지 않는다.

## D5 경로·라벨·데이터 분할 계약과 평가 초안

- 실제 날짜: **2026-10-07 (Asia/Seoul)**
- 상태: **D5 gate PASS**
- 구현: 실제 Town01 고정 spawn 0·4의 분기 없는 35m route catalog와 OpenDRIVE hash, 5개 ego waypoint 계약·JSON Schema, episode/route/scene group 단위 manifest 생성기, 평가 지표·종료·집계 명세 초안
- 실제 분할: 3개 완료 에피소드 180행 중 train 120/dev 60, 같은 train 경로의 반복은 동일 split, route·차량 위치를 포함한 양쪽 최소 거리 약 220.75m로 75m buffer 통과
- 실패 검사: 동일 route의 train/dev 교차 배정 거부, 기존 manifest의 다른 내용 덮어쓰기 거부, 손상·미완료·누락 image는 manifest에 포함하지 않음
- 산출물: [데이터 계약](DATA_CONTRACT_D5.md), [평가 명세](EVALUATION_SPEC_D5.md), [실제 route](../configs/routes_d5.json), [실제 manifest](evidence/carla_dataset_d4_d5_2026-10-07_split_manifest.csv), [검증 요약](evidence/carla_dataset_d4_d5_2026-10-07_summary.json)

모든 실제 frame의 `label_state`는 `NOT_GENERATED`이고 `label`은 `null`이다. D5는 생성될 라벨의 좌표·단위·거리·route provenance 계약을 확정한다. D6에서 해당 frame의 ego 변환·직선/곡선 라벨 생성·overlay를 구현한다. Town02 final test는 미수집·미검증으로 예약하고 D42 freeze 전에 사용하지 않는다. 평가기 구현은 D14, 최종 기준 freeze는 D42 작업이다.

이번 D4/D5는 로컬에서 진행했다. Kaggle 제출이나 새 노트북 생성은 없으며 기존 ID 재사용 원칙을 유지한다. GitHub 계정의 Actions 제한은 계속 남아 있고 원격 CI 성공으로 표시하지 않는다.

최종 전체 로컬 검증 **204 PASS·0 skip**, `pip check`, strict 환경 사전 검사와 PowerShell syntax PASS. PNG 내부 CRC·bounded zlib stream·scanline 검사를 보강한 뒤 실제 원본 3개와 기존 manifest를 다시 읽어 PASS를 확인했다. JSON Schema 외부 검사는 Python 3.12.7·jsonschema 4.26.0으로 원본 frame 180개와 CSV 행 180개에서 오류 0개였다. 이 도구는 검토 환경에서만 사용했고 프로젝트 runtime dependency로 추가하지 않았다.

## D6 고정 목표 waypoint 라벨과 native 곡선 overlay

- 실제 날짜: **2026-10-08 (Asia/Seoul)**
- 상태: **D6 gate PASS**, native 생성·overlay와 다운로드 후 독립 CPU 라벨 검증 완료
- 실행: 기존 `dongmyungpark/vlf-carla-rgb-smoke-20261005`, 제목 `VLF CARLA RGB smoke 20261005`의 **version 4**, Kaggle T4·CARLA 0.9.16·Town01
- 실제 원격 데이터: 정지 직선·좌곡선·우곡선 각각 60 frame, 총 3 episode·180 RGB frame·900 waypoint 점
- 산출물: [D6 라벨·overlay 문서](WAYPOINT_LABELS_D6.md), [실제 요약](evidence/carla_week1_d6_d7_2026-10-08_summary.json), [직선](evidence/carla_week1_d6_d7_2026-10-08_straight_overlay.png)·[좌곡선](evidence/carla_week1_d6_d7_2026-10-08_left_curve_overlay.png)·[우곡선](evidence/carla_week1_d6_d7_2026-10-08_right_curve_overlay.png) overlay

### 구현과 실제 원격 증거

`next(d)`의 OpenDRIVE reference 간격이 곡선 outer lane의 실제 centerline 거리와 다를 수 있음을 native survey에서 확인했다. 실제 lane-centerline XYZ 간 길이를 dense polyline에서 누적하고 0.25m anchor로 보간하여 35m route를 고정했다. 이 physical arc는 수치 근사이며 analytic curve 적분값이 아니다. Original route/hash와 같은 frame actor pose에서 3/6/9/12/15m world 점 및 actor-origin `[x-forward,y-right]` m 타깃을 생성한다.

| Case | Route spawn | 실제 frame 범위 | 수집 frame | 15m net waypoint heading |
| --- | --- | --- | --- | --- |
| Straight | 0 | 1142~1201 | 60 | 0° |
| Left curve | 108 | 1323~1382 | 60 | 약 -27.19° |
| Right curve | 174 | 1478~1537 | 60 | 약 +30.99° |

원본 `episode.json`, `frames.jsonl`과 RGB PNG를 변경하지 않는다. 원본 metadata는 `labels_state=not-generated-D6`, frame은 `NOT_GENERATED`/`null`을 유지하고 라벨만 외부 `GENERATED` sidecar에 저장한다. Source metadata/JSONL/config/route/image 해시·frame identity를 연결하고 fresh raw integrity와 독립 geometry validator를 요구한다. Missing/duplicate/reordered frame, 잘못된 frame/axis/unit/target lane, nonfinite geometry, 부족한 horizon, 자기 교차·모호한 projection, source 변조와 기존 output 덮어쓰기를 거부한다.

원격 라벨의 frame당 shape는 `5 × 2`다. 고정 source route 대비 world/ego/reference 오차는 0m, actor world roundtrip 최대는 `8.66e-12m` 미만이었다. 이 수치를 실제 연속 도로의 절대 측량 오차로 해석하지 않는다. 동일 frame의 RGB/camera pose로 overlay를 만들었으며 3m 점이 영상 밖일 때 원래 UV와 `OUTSIDE_IMAGE`를 유지하고 clamp하지 않았다.

전체 raw와 sidecar는 `results/local/kaggle-week1-output-v4/week1/`에 다운로드했고, CPU에서 같은 source와 label SHA-256를 확인했다. [실제 geometry route catalog](../configs/routes_d6_geometry.json)와 작은 요약·overlay만 검토용으로 보존하며 이 정지 geometry QA로 새 training split을 만들지 않는다. 기존 D4/D5 로컬 180 frame·train 120/dev 60 snapshot도 별도로 유지한다.

## D7 Week 1 동일 frame RGB·pose·label·color 게이트

- 실제 날짜: **2026-10-08 (Asia/Seoul)**
- 상태: **D7 gate PASS**, native 원격·전체 다운로드 CPU 게이트 모두 통과
- 원격 결과: 모든 180 frame에서 다섯 타깃 camera 전방·네 타깃 visible, BGRA→stored RGB 색상 검사 통과
- 정리: Case별 capture cleanup, 수집 후 asynchronous RPC 및 bootstrap의 `server_stopped=true` 확인
- 산출물: [D7 게이트·재현 문서](WEEK1_GATE_D7.md), [실제 D6/D7 요약](evidence/carla_week1_d6_d7_2026-10-08_summary.json)

### 검증 결과와 해결한 실패

Gate는 하나의 dataset version·map/OpenDRIVE source·nominal calibration·spawn 제외 config에서 실제 직선·좌곡선·우곡선 3 case를 확인한다. 각 frame은 같은 sensor/world/snapshot frame과 timestamp, finite actor-origin 라벨과 camera projection을 만족해야 한다. 최소 세 점 visible·모든 다섯 점 전방이라는 판정에서 실제로는 모든 case가 네 점 visible이었다. 3m 점은 전체 180 frame에서 `OUTSIDE_IMAGE`였고 나머지 6/9/12/15m는 visible이었다.

새 native capture의 BGRA buffer hash, RGB hash와 source pixel witness를 보존했다. 저장 PNG를 decode하여 180 frame의 channel order와 720 pixel witness를 확인했다. BGRA 전체 buffer를 저장하지 않으므로 offline source buffer 전체 hash를 다시 계산한 증거로 주장하지 않는다. 기존 D4/D5 raw에 color witness를 사후 넣지 않았다. Camera/world inverse pinhole roundtrip 최대 오차는 약 `7.281e-7m`로 0.001m gate tolerance 안이었다.

Version 3은 첫 두 case 수집 후 세 번째 case의 `weather-readback-mismatch`로 전체 FAIL이었다. [실패 기록](evidence/carla_week1_d6_d7_2026-10-08_weather_failure.json)을 보존했다. Weather 적용·복구의 엄격한 readback 비교를 유지하고 비동기 전달을 기다리도록 최대 2초의 bounded 재시도를 추가했다. 일치하지 않으면 여전히 FAIL이며, version 4 전체 native 실행과 해당 회귀 검증이 통과했다.

Native pipeline과 다운로드 후 최신 CPU helper의 **9개 negative QA**는 잘못된 ego frame·unit·target, duplicate/missing label frame, source hash, PNG 손상, dropped raw frame과 RGB hash를 다시 계산한 R/B swap을 모두 거부했다. 원본은 보존했다. 최신 전체 로컬 unit 검증 **309 PASS·0 skip**는 라벨·overlay·색상·gate·weather strict 재시도 회귀를 포함한다. 외부 `jsonschema 4.26.0`으로 raw frame 180개·derived label 180개를 검사해 오류 0개였으며 runtime dependency로 추가하지 않았다. `pip check`, strict 환경 사전 검사와 PowerShell syntax도 PASS다. 원격 GitHub Actions는 계정 제한으로 실행되지 않았다.

실제 bootstrap 소요 시간은 **452.02초**, D6/D7 pipeline은 약 **66.60초**였다. Job 대기 시간은 별도 측정하지 않았다. Capture 직후 GPU 0의 전체 장치 메모리 표본은 **1,844MiB / 15,360MiB**, 두 번째 T4는 idle **1MiB**였다. 이는 해당 시점의 장치 표본이며 peak 메모리·FPS·모델 추론 latency가 아니다.

### CPU 재검증과 최종 gate 판정

전체 PNG·metadata·JSONL·sidecar·manifest를 다운로드한 뒤 현재 로컬 코드로 다음 fresh disk 검증을 실행해 **3 episode·180 frame·errors=[]**의 PASS를 확인했다. 모든 case의 source와 label SHA-256는 원격 version 4 결과와 일치했다. CARLA server와 로컬 GPU는 사용하지 않았다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_week1.py --manifest results/local/kaggle-week1-output-v4/week1/week1_manifest.json --output results/local/week1-v4-cpu-gate.json
```

**D6/D7 최종 gate: 통과.** 실제 source-pinned native capture·라벨·overlay·color gate, 전체 artifact의 독립 CPU 재검증, 변조 QA와 cleanup 증거를 확보했다. Kaggle에는 기존 ID/title로 version 4를 추가했으며 새 notebook은 만들지 않았다. 이번 단계의 GPU 실행은 Kaggle T4에서 수행했고 로컬 GPU 실행은 없었다.

다음 단계는 **D8 Pure Pursuit controller 구현과 privileged GT waypoint 검증**이다. D13 moving training pilot·recovery 수집, D14 주행 metric 구현·평가는 아직 진행하지 않았다. D6/D7 정지 게이트는 주행 완주율·모델 정확도·inference latency를 검증하지 않는다. Town02 final test는 사용하지 않았고 `UNCOLLECTED`/`UNVERIFIED`를 유지한다.

## D8 Pure Pursuit와 공통 속도 제어

- 실제 날짜: **2026-10-08 (Asia/Seoul)**
- 상태: **D8 gate PASS**
- 구현: actor-origin waypoint를 따르는 rear-axle 회전 원, native wheel의 world-cm 위치에서 측정한 wheelbase·rear/front axle·front track·steering gain, Ackermann 안쪽 전륜 변환, 속도 의존 lookahead·조향 변화율 제한과 PI antiwindup
- 속도: nominal 상한 25km/h, 같은 다섯 waypoint의 preview와 요청 곡률에 2.0m/s² 계획값으로 감속; 20Hz synchronous timestep
- 검증: controller v4/schema 4의 analytic circle·Ackermann plant·입력/상태·속도 검사 **39 PASS**, Kaggle native 15조건과 실제 좌우 전륜 조향 부호 확인
- 산출물: [controller 문서](controller_d8.md), [설정](../configs/controller_d8.json), [구현](../carla_tools/controller.py)

## D9 정답 waypoint 직선·좌우 곡선 주행

- 실제 날짜: **2026-10-08 (Asia/Seoul)**
- 상태: **D9 gate PASS**
- 실행: 기존 `dongmyungpark/vlf-carla-rgb-smoke-20261005`, 제목 `VLF CARLA RGB smoke 20261005`의 **version 11**, Kaggle T4·CARLA 0.9.16·Town01. 새 notebook과 로컬 GPU 실행 없음
- 실제 결과: 세 route × 다섯 조건 **15/15 PASS**, 원래 35m route의 reference 19m 이상 주행 후 service brake·저속 확인·주차까지 검증; 주행 trace **1,540 frame**, RGB **105장**
- 정합: 같은 RGB·snapshot·actor-origin GT label과 다음 frame 적용 control을 strict readback·controller replay로 확인; 원래 route와 15m 미래 horizon 유지
- 안전: collision/lane invasion/callback 오류 **0건**; 전체 주행·정지 최소 차체 여유 **0.24783m ≥0.15m**, stop 완료 **0.50~0.70 simulation seconds**
- 산출물: [검증 명세와 실제 결과](DRIVING_RECOVERY_D9_D10.md), [요약 JSON](evidence/carla_driving_d8_d10_2026-10-08_summary.json), [metric CSV](evidence/carla_driving_d8_d10_2026-10-08_metrics.csv), [그래프](evidence/carla_driving_d8_d10_2026-10-08_recovery.png)

## D10 작은 초기 lateral·yaw 편차 복구

- 실제 날짜: **2026-10-08 (Asia/Seoul)**
- 상태: **D10 gate PASS**
- 조건: catalog 시작점의 접근 구간에서 횡방향 **±0.3m** 또는 yaw **±5°**, 초기 실제 readback 확인 및 원래 목표 차선 유지
- 실제 복구: **12/12 PASS**, 횡오차 ≤0.2m·heading error ≤3°를 연속 1초 유지한 완료 시점 **2.15~2.35초 ≤5초**. 이후 곡선 주행·감속의 차체·속도·센서 기준도 별도로 확인
- 독립 검증: 전체 artifact CPU gate PASS, **16개 변조 QA 모두 거부**, 실제 Town01 `Map`의 주행·정지 차체 모서리 **6,980개 PASS**; 원본 hash 불변 및 제출·원격 readback·로컬 runtime **28개 source hash 일치**
- 종료: 15개 owned actor cleanup·world/weather 복원, 이후 RPC 진행 PASS; Kaggle `COMPLETE`와 `server_stopped=true`, 로컬 server·RPC port 중지 확인
- 최종 코드 품질: 일반 전체 suite **450 PASS·0 skip**, Python dependency·고정 설치 preflight PASS. 두 앞선 OS 파일 쓰기 실패는 로그를 보존했고, 독립·진단·최종 일반 실행은 PASS; 정확한 OS 원인은 미확인
- 해결한 실제 실패: spawn 정렬, 곡선 제동·차체 모서리 계산, 동기 batch control 적용, 곡률 preview 감속, Ackermann inner-wheel 정규화와 actor-origin/rear 궤적 원 기하. 실패 version 5~10의 증거도 Git에 보존

**D8~D10 최종 gate: 통과.** 검증 범위는 privileged GT controller의 짧은 개발 route와 초기 오차 복구다. Camera-only 모델·B0/M1 비교·학습 pilot·D14 전체 평가·실시간 latency는 이후 작업이며 Town02 final은 `UNCOLLECTED`/`UNVERIFIED`를 유지한다. GitHub Actions는 계정 제한으로 원격 실행이 막혀 있어 local/native PASS와 구분한다.

다음 구현은 **D11 OpenCV B0의 RGB 차선 검출·nominal camera 보정·차량 좌표 waypoint overlay**다.

### 이후 기록 양식

```text
일차 / 실제 날짜:
상태:
목표:
완료 작업:
산출물 경로:
실행 명령 및 검증 결과:
Gate 판정:
남은 제약:
다음 작업:
```
