# Kaggle T4와 CARLA 실행 검증

검증 날짜: 2026-10-05 (Asia/Seoul). 사용자 요청에 따라 공식 Kaggle CLI로 비공개 작업을 실제 제출하고 결과를 내려받았다. GPU 존재, CUDA 학습, Vulkan 렌더링과 CARLA RGB 수집을 각각 확인한다.

## 현재 결과

**Kaggle T4에서 CARLA 0.9.16 서버와 RGB 카메라 실행 검증이 통과했다.** 비공개 CARLA 작업 version 2의 실제 결과이며, 코드 검토나 CUDA 동작만으로 판정한 것이 아니다. Simulator와 client를 같은 Kaggle 작업 안에서 실행했다.

| 항목 | 실제 결과 |
| --- | --- |
| 원격 작업 제출 및 결과 다운로드 | Kaggle CLI 2.2.4로 통과 |
| GPU | Tesla T4 2개, 각 15,360MiB |
| NVIDIA 드라이버 | 580.178.04 |
| 기본 Python / libc | CPython 3.13.15 / glibc 2.39 |
| PyTorch | 2.11.0+cu128 |
| 작은 모델의 CUDA 학습 | 각 GPU에서 Linear 모델의 역전파와 SGD 3회, 유한 loss 및 weight 변경 확인 |
| 기본 Vulkan / GLX ICD | 초기화 실패, `VK_ERROR_INCOMPATIBLE_DRIVER` |
| EGL ICD 지정 후 Vulkan | 실제 NVIDIA T4 두 개 열거 통과 |
| CARLA Python API | 별도 CPython 3.12.14 환경에 0.9.16 설치 통과 |
| CARLA 패키지 접근 | 기본 Python User-Agent는 HTTP 403, 명시한 다운로드 헤더에서는 HTTP 200 확인 |
| 실제 CARLA RPC / client·server 버전 | 통과 / 모두 0.9.16 |
| 실제 CARLA RGB 이미지 | Town01, 640×360, 측정 20장 모두 유효 RGB 및 frame 번호 일치 |
| 수집 루프 처리율 | 22.201 FPS, real-time factor 1.110; 정지 차량의 짧은 수집 시험 |
| Tick 시작부터 RGB 수신까지 | 평균 15.08ms, 최대 23.15ms |
| 수집 시점 GPU 메모리 | 첫 T4 전체 장치 5,635~5,640MiB; 두 번째 T4 1MiB |
| 전체 bootstrap 작업 시간 | 374.12초, 약 6분 14초; 다운로드·추출·실행·종료 포함 |
| 센서·차량 정리와 world 설정 복구 | RGB probe에서 통과 |

GPU별 학습 검증은 동일한 작은 모델을 각각 실행한 것이며, 분산 학습의 검증은 아니다. 이 결과만으로 프로젝트 모델의 학습 시간이나 품질을 예측하지 않는다.

시뮬레이션 수집 구간은 1초 분량의 정지 차량 20프레임이다. Capture 처리율에는 Python RGB 검사가 포함되며 PNG 저장은 제외된다. Warmup 5장과 map 전환·시작 시간은 이 처리율에 포함하지 않는다. 장시간 안정성, 주행 controller, lane-following 모델과 dataset 수집 품질을 검증한 결과는 아니다.

서버를 종료하는 단계에서 `Signal 11 caught` 로그가 남았다. 이는 성공한 RGB 수집 이후의 종료 구간이며, 정상적인 엔진 종료까지 통과했다고 주장하지 않는다. Bootstrap의 `server_stopped`는 생성한 launcher 종료를 확인한 필드다. 반복 실행 및 장시간 수집을 본 작업에 적용하기 전에 종료 절차의 안정성을 추가 확인한다.

## 해결한 환경 차이

CARLA 0.9.16은 Vulkan을 사용하고 `-RenderOffScreen`에서 카메라를 렌더링할 수 있다. `no_rendering_mode=True`는 카메라 데이터를 비우므로 사용하지 않는다. [CARLA 공식 렌더링 문서](https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/).

이번 Kaggle 세션은 기본 Vulkan ICD manifest가 없었고, 기존 GLX 라이브러리를 지정해도 초기화가 실패했다. 새 프로세스에서 기존 `libEGL_nvidia.so.580.178.04`를 지정하니 소프트웨어 렌더러가 아닌 실제 T4 장치 두 개가 나타났다. NVIDIA는 GLX와 EGL 라이브러리 모두 Vulkan ICD로 사용할 수 있다고 설명한다. [해당 드라이버의 공식 구성 요소 설명](https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/README/installedcomponents.html).

기본 Python 3.13은 CARLA 0.9.16 wheel의 지원 대상이 아니므로, `/tmp`의 별도 Python 3.12 환경에서 클라이언트를 실행한다. GPU 드라이버나 로컬 글로벌 Python 환경은 교체하지 않는다. [공식 CARLA wheel 목록](https://pypi.org/project/carla/0.9.16/).

배포 파일은 공식 다운로드 주소를 사용한다. 다운로드 헤더를 명시하고, 첫 주소에 접속할 수 없으면 CARLA 공식 도구에 등록된 Backblaze 미러를 사용한다. HTTP 200과 예상 길이 8,346,095,504바이트를 확인한다. 파일 길이는 암호학적 checksum 검증을 대신하지 않는다. [공식 릴리스](https://github.com/carla-simulator/carla/releases/tag/0.9.16), [공식 미러 정의](https://github.com/carla-simulator/carla-agentic-tools/blob/main/skills/setup/download-carla/scripts/download_carla.py).

## 재현 방법

**이 프로젝트는 기존 노트북을 재사용한다.** 사용자 요청에 따라 노트북을 계속 추가하지 않고, 같은 ID와 제목을 유지한 채 코드와 실행 버전만 갱신한다. 현재 CARLA 실행, GPU·Vulkan 진단, 다운로드 진단의 총 3개가 있으며, 다운로드 진단은 다운로드 문제를 조사할 때만 사용한다. 날짜나 일차가 바뀌어도 이름의 `20261005`를 바꿔 새 노트북을 만들지 않는다. 저장소의 [작업 지침](../AGENTS.md)에 이 원칙을 기록했다.

이 저장소의 실제 제출 대상은 `dongmyungpark` 계정의 기존 노트북이다. 제출 전에 `kernels status` 등으로 기존 대상에 접근 가능한지 확인한다. 아래 재현 예제의 계정 이름 자리에는 이 프로젝트 작업 시 `dongmyungpark`를 사용한다.

Windows 저장소 루트에서 실행한다. 인증정보는 저장소와 생성한 작업 코드에 포함하지 않는다.

```powershell
py -3.12 -m venv .venv-kaggle
.\.venv-kaggle\Scripts\python.exe -m pip install kaggle==2.2.4
.\.venv-kaggle\Scripts\kaggle.exe auth login
```

다음 명령은 결과 폴더에 코드와 비공개 metadata를 준비하며, 준비만으로 GPU 작업을 시작하지 않는다. 준비 도구는 고정된 기존 slug를 사용하고, 새로운 날짜나 실행마다 slug를 생성하지 않는다.

```powershell
.\.venv-kaggle\Scripts\python.exe scripts/prepare_kaggle_probe.py --owner YOUR_KAGGLE_USERNAME --kind vulkan --output-dir results/local/kaggle-preflight
.\.venv-kaggle\Scripts\kaggle.exe kernels push -p results/local/kaggle-preflight --accelerator NvidiaTeslaT4 -t 300

.\.venv-kaggle\Scripts\python.exe scripts/prepare_kaggle_probe.py --owner YOUR_KAGGLE_USERNAME --kind carla --output-dir results/local/kaggle-carla-smoke
.\.venv-kaggle\Scripts\kaggle.exe kernels push -p results/local/kaggle-carla-smoke --accelerator NvidiaTeslaT4 -t 1500
```

상태 확인 후 결과를 내려받는다. 검증 작업은 GPU 할당량을 사용한다. CARLA 작업 자체의 제한은 20분이고 플랫폼 제한은 결과 저장과 종료의 여유를 포함한 25분이다. Simulator와 API는 `/tmp`, JSON과 PNG는 `/kaggle/working`에 저장한다.

```powershell
.\.venv-kaggle\Scripts\kaggle.exe kernels status YOUR_KAGGLE_USERNAME/vlf-carla-rgb-smoke-20261005
.\.venv-kaggle\Scripts\kaggle.exe kernels output YOUR_KAGGLE_USERNAME/vlf-carla-rgb-smoke-20261005 -p results/local/kaggle-carla-smoke-output
```

전체 작업은 NVIDIA Vulkan 장치 확인, non-root CARLA 서버 실행, 실제 RPC 호출, 정확한 API/server 버전 일치, Town01의 RGB 카메라 수집 순서로 판정한다. 640×360, Low 렌더링, 0.05초 simulation step, warmup 5개와 측정 frame 20개를 사용한다. Frame 번호 일치, BGRA 길이, 공간적으로 변화하는 RGB 픽셀과 실제 PNG 저장을 요구한다. 단순 포트 응답이나 CUDA 성공으로 CARLA 성공을 표시하지 않는다.

카메라 latency와 capture 처리율은 동기식 수집 루프의 측정치다. 차량은 정지 상태이며, controller·모델 추론·실시간 주행 성능을 측정하지 않는다. GPU 메모리는 전체 장치의 두 시점 snapshot이며 CARLA 프로세스의 peak 측정치가 아니다. 관측한 약 5.5GiB만 보고 로컬 6GiB에서 같은 조건이 반드시 동작한다고 판단하지 않는다.

## 증거와 적용 범위

- [초기 GLX 진단과 실제 CUDA 학습 결과](evidence/kaggle_preflight_2026-10-05.json).
- [최종 EGL Vulkan 및 CUDA 진단 결과](evidence/kaggle_verified_environment_2026-10-05.json): 선행 진단 version 5. 이 도구는 CARLA 카메라를 실행하지 않으므로 자체 `carla_rgb_verified`는 false다.
- [실제 CARLA bootstrap 결과](evidence/kaggle_carla_bootstrap_2026-10-05.json): CARLA 작업 version 2, PASS.
- [20프레임의 RGB·정합·시간·메모리 기록](evidence/kaggle_carla_rgb_2026-10-05.json): 실제 CARLA API의 sensor data.
- [실제 카메라 PNG](evidence/kaggle_carla_rgb_2026-10-05.png): 받은 BGRA에서 RGB로 변환한 원본 preview.
- [비공개 선행 진단 작업](https://www.kaggle.com/code/dongmyungpark/vlf-carla-vulkan-preflight-20261005).
- [비공개 CARLA RGB 작업](https://www.kaggle.com/code/dongmyungpark/vlf-carla-rgb-smoke-20261005).

CLI 원격 실행을 검증했으며 VS Code Jupyter 연결 자체는 실행하지 않았다. 로컬 RTX 3060 6GiB의 CARLA 실행 가능성도 이 원격 결과만으로 판정하지 않는다. 이번 결과는 해당 날짜의 실제 Kaggle 이미지와 T4 세션에 대한 기록이며, 이후 환경과 GPU 공급은 달라질 수 있다.

추가한 검증 도구는 전체 단위 검증 36개를 통과했다. 이 중 RGB 증거, PNG 픽셀·CRC, frame 정합, 실패 판정과 정리를 검사하는 19개는 simulator 없이 실행하는 단위 검증이며, 실제 CARLA 실행 근거는 위 원격 산출물이다. 이번 작업 종료 시 제출한 CPU/GPU 작업은 모두 완료 또는 종료 상태임을 확인했다.
