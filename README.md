# Vision-Lane-Following

CARLA에서 전방 RGB 영상으로 차선 중심 경로를 예측하고, 공통 controller로 주행 성능을 비교하기 위한 프로젝트입니다. 수집·라벨 생성·데이터 분할·제어·주행 검사 도구를 제공합니다.

OpenCV baseline은 RGB와 고정 nominal camera calibration으로 waypoint를 예측합니다. 지도에서 얻은 정답 waypoint를 사용하는 **privileged reference**와 같은 controller·초기 조건·종료 규칙으로 비교합니다. ResNet18 RGB waypoint 모델의 학습·checkpoint 재개·독립 개발 데이터 평가도 제공합니다.

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
- RGB 차선 표시 검출·고정 ground-plane 역투영·중심선 arc 기반 OpenCV waypoint와 overlay.
- 종료 원인·event 중복 제거·실패 포함 오차/완주율·복귀·제어 변화율·처리 시간의 episode CSV와 집계.
- 실제 이동 RGB pilot 수집, route·scene 기반 분할과 공간 누수 검사.
- 사전학습 ResNet18 waypoint 회귀, photometric augmentation, AMP와 상태 검증을 포함한 학습 재개.
- 개발 데이터의 근거리·원거리 및 곡선·초기 편차별 예측 오차, coverage와 동일 RGB의 baseline 비교.
- 선택 checkpoint의 계약·SHA-256 검사, 원본 RGB8 PNG의 로컬 예측 API와 JSON 출력.

초기 범위는 Town01의 분기 없는 직선·좌곡선·우곡선, 고정 차량과 카메라, 맑은 날씨 및 다른 차량과 보행자가 없는 조건입니다. 교차로 의사결정, 차선 변경과 객체 회피는 포함하지 않습니다.

## 시작하기

Windows PowerShell에서 저장소 루트를 기준으로 실행합니다. Python 3.12 x64와 공식 CARLA 배포 파일을 위한 충분한 디스크 공간이 필요합니다. 설치 도구는 프로젝트 `.venv`와 `carla/CARLA_0.9.16`을 준비하고 같은 설치를 재사용합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_d2.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
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

## RGB baseline 비교

OpenCV는 CPU에서 실행하며 simulator 렌더링에 GPU를 사용합니다. 기존 Kaggle notebook에서 GPU 주행을 실행하는 준비 절차는 [환경 준비](docs/setup.md)에 있습니다.

```powershell
# 기존 RGB8 PNG에서 nominal calibration으로 예측과 overlay를 생성합니다.
.\.venv\Scripts\python.exe scripts/predict_baseline.py --image results/local/example-rgb.png --output results/local/baseline-overlay/prediction.json --overlay results/local/baseline-overlay/overlay.png

# 실행 중인 전용 Town01 서버에서 RGB baseline과 privileged reference를 비교합니다.
.\.venv\Scripts\python.exe scripts/run_baseline_pipeline.py --output-dir results/local/baseline-example
.\.venv\Scripts\python.exe scripts/verify_baseline.py --manifest results/local/baseline-example/baseline_manifest.json --output results/local/baseline-example-gate.json
```

한쪽 yellow marking만 관측되면 고정 4m 차선 폭을 가정해 반대 경계를 추정하고, 그 가정을 diagnostics에 표시합니다. Calibration과 평면·차선 폭 가정은 frame별 map/pose로 보정하지 않습니다. 검출에 실패하면 제동하고 실패 episode를 metric CSV에 포함합니다. Artifact integrity 통과와 차선 주행 성공을 구분합니다. 자세한 범위와 제한은 [OpenCV baseline](docs/baseline.md)을 참고하세요.

## RGB waypoint 학습

Pilot은 정답 경로 제어기로 실제 이동 중 수집하며, 학습용과 개발용의 route 및 주변 scene을 분리합니다. 정지 프레임과 중복 RGB를 제외하고 영상·라벨·제어·종료 기록을 독립적으로 재검증합니다.

확대 수집은 개발용 정오 날씨와 시작 위치를 바꾸어 중앙·복귀 장면을 연결합니다. 같은 장면을 진행 거리로 짝지어 동일한 데이터 예산의 학습 선택을 고정하고, 원본 영상·라벨·분할을 재검증합니다. 사용법은 [확대 데이터셋](docs/dataset.md)에 있습니다.

중앙·복귀 데이터 비교는 같은 초기 하이브리드 모델과 학습 예산으로 여러 시드를 학습합니다. 공통 개발 영상의 예측 오차와 같은 조건의 실제 주행을 함께 평가하며, 계약과 사용법은 [복귀 데이터 학습 비교](docs/recovery-learning.md)에 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/run_pilot_pipeline.py --output-dir results/local/pilot-example
.\.venv\Scripts\python.exe scripts/verify_pilot.py --manifest results/local/pilot-example/pilot_manifest.json --output results/local/pilot-example-gate.json
```

모델은 원본 640 × 360 RGB를 입력받습니다. 전체 폭을 유지한 고정 영역 `[0, 144, 640, 360]`을 잘라 640 × 216 영상을 `INTER_AREA`로 320 × 180으로 변환하고 근거리·원거리 waypoint를 함께 회귀합니다. 라벨은 차량 actor 좌표계의 `5 × 2` 배열과 m 단위를 유지합니다. GPU 학습은 기존 Kaggle notebook의 pilot 출력을 연결해 실행합니다. 설치와 실행, checkpoint 재개 및 오차 해석은 [학습 문서](docs/learning.md)를 참고하세요. 학습 모델의 offline 예측 정확도와 실제 closed-loop 주행 성능은 별도로 검증합니다.

`BaselineAssistedWaypointPredictor`는 유효한 OpenCV waypoint를 사용하고 검출에 실패하면 같은 RGB를 학습 모델에 전달하는 선택 옵션입니다. 순수 학습 모델과 이 옵션의 성능은 각각 평가하며, 경로 선택에는 정답 라벨이나 차량 위치를 사용하지 않습니다.

선택한 `checkpoint.pt`로 simulator 없이 단일 원본 RGB8 PNG를 예측할 수 있습니다. `learned`가 기본 모드이며, 출력은 차량 actor 좌표계의 m 단위 waypoint입니다. 출력 파일은 새 경로를 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts/predict_waypoints.py --checkpoint results/local/waypoint-training/checkpoint.pt --image results/local/example-rgb.png --output results/local/local-inference/prediction.json --device cpu --mode learned
```

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-learning.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 설정·파일 계약, 좌표 변환, 제어 기하, frame별 제어 적용, 복귀·제동·정리 및 artifact 검사를 다룹니다. Simulator를 사용하는 실행 검사는 별도의 서버와 환경 준비가 필요합니다.

## 기술 문서

- [프로젝트 범위](docs/PROJECT_SCOPE.md): 입력·출력, 비교 방법과 데이터·평가 원칙.
- [환경 준비](docs/setup.md): 설치, 서버 관리, 환경 진단과 Kaggle 실행.
- [수집과 episode 기록](docs/collection.md): 동기 수집, frame 정합과 파일 검사.
- [데이터 계약](docs/data-contract.md): 좌표·라벨 schema, route와 split 규칙.
- [Waypoint 라벨](docs/waypoint-labels.md): 고정 경로, 좌표 변환과 overlay.
- [OpenCV baseline](docs/baseline.md): RGB 검출, nominal calibration, 차선 폭 가정과 예측 실패 처리.
- [Pilot 수집](docs/pilot.md): 이동 sample, scene 분할, 공간 누수와 원본 재검증.
- [확대 데이터셋](docs/dataset.md): 중앙·복귀 장면의 짝, 동일 예산 학습 선택과 데이터 버전 고정.
- [Waypoint 학습](docs/learning.md): RGB preprocessing, ResNet18, 학습·재개 및 개발 평가.
- [복귀 데이터 학습 비교](docs/recovery-learning.md): 동일 예산 A/B 학습, 시드별 평가와 공통 조건 주행 선택.
- [개발 주행 비교](docs/development-study.md): 제어기 조정, RGB 보정 검사, 실제 카메라·환경 변화와 개발 후보 회귀 평가.
- [외관 변화 추가 학습](docs/development-training.md): 동일 초기화·학습 예산의 추가 학습과 환경 변화 대응 비교.
- [선택 파이프라인 실행](docs/selected-pipeline.md): 모델·RGB 경로 보정·제어기 설정을 함께 고정한 추론과 연속 프레임 제어.
- [Controller](docs/controller.md): 차량 기하, Pure Pursuit·Stanley와 속도 제어.
- [주행과 복귀](docs/driving.md): 초기 편차, 제어 readback, 센서·차체·정리 검사.
- [평가 명세](docs/evaluation.md): 공통 조건, 주행 지표와 final test 원칙.
- [독립 평가 동결](docs/final-evaluation.md): 보류 경로·시작 상태·평가 지표·실행 예산의 사전 확정과 코드·모델·데이터 변경 검출.

수집 데이터, simulator 배포 파일, checkpoint와 실행 출력은 Git에서 제외되는 로컬 경로에 보관합니다.
