# Vision-Lane-Following

CARLA에서 전방 RGB 영상으로 차선 중심 경로를 예측하고, 공통 controller로 주행 성능을 비교하기 위한 프로젝트입니다. 수집·라벨 생성·데이터 분할·제어·주행 검사 도구를 제공합니다.

현재 주행 도구는 지도에서 얻은 정답 waypoint를 사용하는 **privileged controller 개발 QA**입니다. RGB 입력만으로 경로를 예측하는 OpenCV baseline과 학습 모델은 별도의 구현 대상입니다. 정답 경로를 사용하는 검사 결과를 camera-only 모델의 주행 성능으로 해석하지 않습니다.

## 시스템과 입력

```text
전방 RGB → waypoint 예측 → waypoint 5개 → controller → steering
                                            ↑
                                       차량 속도 피드백
```

경로 예측기의 입력은 전방 RGB입니다. 차량 속도는 저수준 속도 제어와 controller의 속도 의존 계산에 사용합니다. 지도, 차량의 세계 좌표 자세와 정답 경로는 라벨 생성·평가 및 privileged reference에 사용합니다.

Waypoint는 목표 차선 중심선을 따라 3·6·9·12·15m 앞의 점입니다. 같은 simulator frame의 차량 actor 좌표계에서 전방을 x, 우측을 y로 하여 `5 × 2` 배열, m 단위로 표현합니다. 목표 차선은 처음 선택한 경로로 유지합니다.

## 제공 기능

- CARLA 0.9.16 설치·서버 관리, API/server 버전 및 연결 확인.
- 동기 20Hz 수집, RGB·snapshot frame 정합, 카메라 calibration과 적용 설정 readback.
- RGB PNG·SHA-256·차량 자세·속도·가속도 기록과 독립적인 episode 파일 검사.
- 고정 route 기반 waypoint sidecar, 차량·세계·카메라 좌표 변환과 overlay.
- Route·episode·scene group 단위 train/dev 분할과 공간 buffer 검사.
- 실제 wheel geometry를 사용하는 Pure Pursuit, Ackermann 변환, PI 속도 제어와 곡률 preview 속도 계획.
- 고정 route 주행·초기 편차 복귀, 충돌·차선 침범·차체 차선 여유, 제동·actor 정리 검사.
- 주행 artifact CPU 재검증, 손상 검사와 native Map 기반 차체 모서리 검사.

초기 범위는 Town01의 분기 없는 직선·좌곡선·우곡선, 고정 차량과 카메라, 맑은 날씨 및 다른 차량과 보행자가 없는 조건입니다. 교차로 의사결정, 차선 변경과 객체 회피는 포함하지 않습니다.

## 시작하기

Windows PowerShell에서 저장소 루트를 기준으로 실행합니다. Python 3.12 x64와 공식 CARLA 배포 파일을 위한 충분한 디스크 공간이 필요합니다. 설치 도구는 프로젝트 `.venv`와 `carla/CARLA_0.9.16`을 준비하고 같은 설치를 재사용합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_d2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Start
.\.venv\Scripts\python.exe scripts/check_carla_connection.py --timeout 30 --ticks 3
.\.venv\Scripts\python.exe scripts/carla_rgb_probe.py --reuse-current-world --output-dir results/local/rgb-probe --timeout 60
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Stop
```

서버 상태는 `scripts/carla_server.ps1 -Action Status`로 확인합니다. 수집·주행 실행에는 전용 Town01 서버를 사용합니다. 환경 설정과 기존 Kaggle notebook 재사용 절차는 [환경 준비](docs/setup.md)에 설명합니다.

환경 metadata만 조사하려면 CARLA나 PyTorch 설치 없이 다음 명령을 사용할 수 있습니다.

```powershell
py -3.12 scripts/doctor.py --output results/local/environment.json
```

## 데이터 수집과 주행 검사

설정 검사는 CARLA 서버 없이 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_collection.py --validate-only
```

실제 episode 수집에는 실행 중인 서버가 필요합니다. 아래 명령은 고정 route에서 정지 상태의 logger 시험 episode를 만들며, 학습용 주행 데이터 수집과 구분합니다.

```powershell
.\.venv\Scripts\python.exe scripts/collect_episode.py --route-id town01_train_p000 --episode-id example_train_01 --dataset-version example-v1 --output-root results/local/episodes
.\.venv\Scripts\python.exe scripts/validate_episode.py results/local/episodes/example_train_01
```

Privileged 주행 pipeline은 고정 route와 초기 편차 조건을 실행합니다. 출력 directory는 새 경로여야 합니다. CPU 재검증은 수집된 artifact를 읽으며 CARLA 서버와 GPU가 필요하지 않습니다.

```powershell
# 실행 중인 전용 Town01 서버에서 주행 조건을 수집합니다.
.\.venv\Scripts\python.exe scripts/run_driving_pipeline.py --output-dir results/local/driving-example

# 수집이 완료된 artifact를 독립적으로 재검증합니다.
.\.venv\Scripts\python.exe scripts/verify_driving.py --manifest results/local/driving-example/driving_manifest.json --output results/local/driving-example-gate.json
.\.venv\Scripts\python.exe scripts/audit_driving.py --native results/local/driving-example --output results/local/driving-example-audit.json
```

주행 판정·artifact 구조와 native Map 검사 조건은 [주행과 복귀](docs/driving.md)에 설명합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 설정·파일 계약, 좌표 변환, 제어 기하, frame별 제어 적용, 복귀·제동·정리 및 artifact 검사를 다룹니다. Simulator를 사용하는 실행 검사는 별도의 서버와 환경 준비가 필요합니다.

## 기술 문서

- [프로젝트 범위](docs/PROJECT_SCOPE.md): 입력·출력, 비교 방법과 데이터·평가 원칙.
- [환경 준비](docs/setup.md): 설치, 서버 관리, 환경 진단과 Kaggle 실행.
- [수집과 episode 기록](docs/collection.md): 동기 수집, frame 정합과 파일 검사.
- [데이터 계약](docs/data-contract.md): 좌표·라벨 schema, route와 split 규칙.
- [Waypoint 라벨](docs/waypoint-labels.md): 고정 경로, 좌표 변환과 overlay.
- [Controller](docs/controller.md): 차량 기하, Pure Pursuit와 속도 제어.
- [주행과 복귀](docs/driving.md): 초기 편차, 제어 readback, 센서·차체·정리 검사.
- [평가 명세](docs/evaluation.md): 공통 조건, 주행 지표와 final test 원칙.

수집 데이터, simulator 배포 파일, checkpoint와 실행 출력은 Git에서 제외되는 로컬 경로에 보관합니다.
