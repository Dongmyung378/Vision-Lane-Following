# 실행 환경과 설치

CARLA 서버와 Python API는 `0.9.16`으로 맞춘다. Windows 도구는 CPython 3.12 x64와 PowerShell을 사용하며, 설치 설정은 [environment.json](../config/environment.json), API wheel과 hash는 [requirements-d2.txt](../requirements-d2.txt)에 고정한다. Simulator 배포, 가상환경, 실행 로그는 Git에서 제외한다.

## Windows 설치와 서버

저장소 루트에서 실행한다. 설치 도구는 ZIP 보관·추출·작업 공간을 검사하며 최소 35GiB의 여유 공간을 요구한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_d2.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Start
.\.venv\Scripts\python.exe scripts/check_carla_connection.py --timeout 60 --ticks 3 --output results/local/carla_connection.json
```

`setup_d2.ps1 -SkipSimulator`는 API만 설치한다. 설치 도구는 API wheel의 고정 hash, 서버 ZIP의 길이·고정 hash와 추출 CRC를 확인한다. 서버 ZIP의 고정 hash는 파일 동일성 확인이며 독립적인 공식 서명 검증을 뜻하지 않는다.

서버 시작 도구는 프로젝트 배포의 `GameDefaultMap`을 Town01로 설정하고 Low·off-screen 모드로 실행한다. 원본 설정은 `results/local/`에 보관하며, 알려진 기본값 또는 이미 설정한 Town01만 변경 대상으로 허용한다. 실제 map과 버전은 RPC로 확인한다. `-RenderOffScreen`에서도 GPU 렌더링은 수행되며 RGB 수집에 `no_rendering_mode=True`를 사용하지 않는다.

연결 검사는 package/client/server 버전, map과 snapshot을 읽고 비동기 world의 새로운 tick을 관찰한다. 다른 client가 동기 world를 관리하면 `PENDING`이며 world를 임의로 진행시키지 않는다. Process나 TCP listener의 존재만으로 API 준비를 판단하지 않는다. 서버 초기화가 늦으면 상태와 로그를 확인한 뒤 연결 검사를 다시 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Status
.\.venv\Scripts\python.exe scripts/carla_rgb_probe.py --output-dir results/local/rgb-probe --timeout 60 --warmup 5 --frames 20 --reuse-current-world
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Stop
```

RGB probe는 시험 actor를 생성하고 임시 동기 설정을 적용하므로 다른 수집 client가 없는 상태에서 실행한다. 종료 도구는 기록한 PID·executable·생성 시각으로 소유권을 확인한 process만 종료한다. `Stop`은 강제 종료 fallback을 포함한다.

## CPU 진단과 테스트

```powershell
py -3.12 scripts/doctor.py --output results/local/environment.json
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`doctor.py --strict --carla-root carla/CARLA_0.9.16`은 정적 사전 조건을 검사한다. 실제 서버 연결·렌더링·주행은 각각 해당 도구로 확인한다.

## Kaggle 실행

Kaggle에서는 GPU 작업 안에서 Linux CARLA 서버와 같은 버전의 client를 함께 실행한다. [bootstrap](../scripts/kaggle_carla_bootstrap.py)은 NVIDIA Vulkan 장치, 호환 Python 환경, RPC와 RGB 수집을 준비한다. CUDA 동작과 Vulkan 렌더링은 별도 조건이다.

OpenCV baseline은 CPU 전용 NumPy·headless OpenCV를 사용한다. Kaggle baseline workload는 고정한 Linux wheel hash와 설치 version을 확인한다. RGB 주행과 reference 비교는 같은 native controller와 평가 설정을 사용하며, 실행 종료 후 owned 서버를 정리한다.

인증은 별도 `.venv-kaggle` 환경의 Kaggle CLI에서 설정하며 인증정보를 코드·문서·제출 bundle에 넣지 않는다. 기존 notebook의 ID와 제목을 유지하고, 제출 전 읽기 전용 status/pull로 존재·접근·실행 상태를 확인한다. 접근 실패를 새 notebook 생성으로 대체하지 않는다. 준비 도구는 등록된 notebook의 metadata와 일치해야 하므로 임의 ID를 받지 않는다. 주행 bundle 준비와 다운로드 후 확인은 [주행 문서](driving.md)를 따른다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_kaggle_baseline.py --existing-metadata results/local/kaggle-existing/kernel-metadata.json --output-dir results/local/kaggle-baseline-bundle
```

이 명령은 제출 bundle만 준비한다. 기존 notebook을 갱신하여 실행한 뒤 `baseline_artifacts.zip`, bootstrap 보고서와 runtime source bundle을 로컬에 다운로드한다. [baseline](baseline.md)과 [평가](evaluation.md) 문서의 CPU 재검증으로 RGB 예측·제어·평가를 다시 확인한다.
