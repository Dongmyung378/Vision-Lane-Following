# D1 개발 환경 조사

2026년 10월 5일 Asia/Seoul 기준 실제 호스트를 조사했다. D1은 범위와 환경을 확인하는 단계이며 CARLA 설치, server 연결 및 GPU 학습은 D2 이후에 검증한다. 조사 도구는 시스템을 변경하거나 패키지를 설치하지 않는다.

## 확인한 하드웨어와 도구

| 항목 | 실제 조사 결과 | 조사 방법 |
| --- | --- | --- |
| OS | Microsoft Windows 11 Pro, build 10.0.26300, x64 | Windows CIM 및 RuntimeInformation |
| CPU | Intel Core i7-11800H 2.30GHz, 8 cores, 16 logical processors | Win32_Processor |
| RAM | 68,563,558,400 bytes, 약 63.85GiB | Win32_ComputerSystem |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | nvidia-smi |
| GPU 메모리 | 6,144MiB, 6GiB | nvidia-smi memory.total |
| NVIDIA 드라이버 | 616.92 | nvidia-smi driver_version |
| 프로젝트 디스크 여유 | 조사 시 약 1,337GiB | 프로젝트 volume의 free space |
| 실행 Python | CPython 3.12.7, 64bit | py -3.12의 sys.version 및 pointer size |
| Git | 2.47.1.windows.1 | git --version |
| GitHub CLI | 2.100.0 | gh --version |

메모리와 디스크는 binary 단위인 GiB로 표기했다. 디스크 여유는 실행 시점에 따라 달라지므로 다음 실행에서도 다시 측정한다. GitHub 인증은 저장소 관리 권한이 있는 상태임을 확인했으며 계정의 인증정보는 보고서에 저장하지 않는다.

## Python과 simulator 준비 상태

| 항목 | D1 판정 | 다음 작업 |
| --- | --- | --- |
| CPython 3.12 x64 | 확인 | 프로젝트 가상환경도 같은 Python 계열로 구성 |
| CARLA 0.9.16 API | 조사한 Python 환경에 설치되지 않음 | D2에 동일 버전 API 설치 |
| CARLA server 패키지 | 설치 경로가 지정되지 않아 미확인 | D2에 패키지 위치와 server 버전 확인 |
| CARLA client/server 연결 | 실행하지 않음 | D2에 실제 연결과 버전 일치 확인 |
| PyTorch metadata | 기존 환경에 2.14.0+cu130 설치 확인 | 학습 환경을 별도로 구성하고 실제 CUDA 실행 확인 |
| PyTorch CUDA 동작 | 검증하지 않음 | 학습 단계의 작은 tensor 실행으로 검증 |
| 주행 및 학습 성능 | 측정하지 않음 | Pilot 주행과 학습에서 측정 |

기존 글로벌 환경의 PyTorch 버전은 프로젝트의 확정 dependency가 아니다. D1 진단은 package metadata만 읽으며 PyTorch와 CARLA를 import하지 않는다. `nvidia-smi`가 실행됐다는 사실도 CUDA toolkit 설치나 PyTorch CUDA 동작을 보장하지 않는다.

실제 CARLA 0.9.16 Windows 배포 wheel에는 CPython 3.10, 3.11, 3.12 x86-64가 제공된다. 현재 설치된 3.12 x64를 사용해 새 interpreter 설치 없이 시작하고, D2에서 simulator와 API의 동일 버전을 확인한다. [공식 PyPI 파일 목록](https://pypi.org/project/carla/0.9.16/).

## 실행 자원에 대한 판단

CARLA 공식 quickstart는 Windows 10과 11, 약 20GB 설치 공간, RTX 2070급 이상과 VRAM 8GB 이상을 권장한다. 현재 디스크 공간은 패키지 설치에 충분하지만, GPU의 VRAM 6GB는 권장치보다 작다. 이것만으로 실행 불가나 성공을 판정하지 않는다. [CARLA 공식 요구사항](https://carla.readthedocs.io/en/0.9.16/start_quickstart/).

D2의 첫 실행은 작은 맵, 낮은 렌더링 설정, 주행 차량 1개 및 이후 RGB 카메라 1개로 진행한다. Camera가 렌더링되어야 하므로 이미지를 출력하지 않는 no-rendering mode를 데이터 수집의 해결책으로 사용하지 않는다. 실행 실패, GPU 메모리와 실제 frame 처리 시간을 보고 구성을 결정한다. 수집과 평가를 로컬에서 수행하고 큰 학습을 Kaggle로 분리하는 원칙은 유지한다.

## 재실행 방법과 출력 의미

저장소 루트에서 실행한다.

```powershell
py -3.12 scripts/doctor.py --output results/local/environment.json
py -3.12 -m unittest discover -s tests -v
```

Windows wrapper도 같은 도구를 호출한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/doctor.ps1 -OutputPath results/local/environment.json
```

Server 폴더를 준비한 뒤에는 명시적으로 검사한다.

```powershell
py -3.12 scripts/doctor.py --carla-root "D:\Simulators\CARLA_0.9.16" --strict
```

일반 진단의 성공은 **환경 조사를 완료했다는 의미**다. 도구가 없는 경우와 권한 또는 timeout 때문에 조사하지 못한 경우를 구분하고, 확인하지 못한 항목을 성공으로 표시하지 않는다. Strict 사전 검사는 D2 준비 조건을 점검할 뿐 simulator를 실행하거나 server에 연결하지 않는다.

이번 조사는 제한된 실행 환경의 CIM 및 Python launcher 조회 실패를 권한이 있는 환경에서 재확인했다. 제한된 환경의 조회 실패를 실제 호스트의 Python 미설치로 판정하지 않았다. 단위 검증 17개, 실제 JSON 출력과 Windows 진입점 실행이 통과했으며, 아직 준비되지 않은 D2 strict 사전 검사는 예상대로 종료 코드 `1`을 반환했다.

자동 출력은 `results/local/`에 보관하고 Git에서 제외한다. 이 문서는 재현에 필요한 장치와 버전 정보만 요약한다. 사용자 이름, hostname, home 경로, 환경 변수 값, 인증정보, 하드웨어 serial 및 UUID는 기록하지 않는다.

## D1 완료와 D2 인계

**D1 환경 조사는 완료했다.** 지원되는 기존 Python과 하드웨어 구성을 확인했고, VRAM 제약과 미확인 simulator 항목을 분리했다. D2 완료 조건은 프로젝트 가상환경 구성, CARLA 0.9.16 server 및 동일 API 준비, 실제 client/server 연결과 버전 일치다. D1 조사 결과를 D2 연결 성공으로 표시하지 않는다.
