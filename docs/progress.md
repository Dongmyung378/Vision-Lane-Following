# 진행 기록

작업 일차와 실제 날짜를 함께 기록한다. 완료 여부는 산출물과 검증에 근거하며, 시간 경과나 설치 파일 존재만으로 다음 단계의 성공을 표시하지 않는다. 전체 계획은 [ROADMAP.md](ROADMAP.md)에 있다.

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
