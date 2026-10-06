# D2 CARLA Windows 환경 구성과 연결 검증

작업일: 2026-10-06 (Asia/Seoul). [D1 조사](ENVIRONMENT_D1.md)에서 확인한 Windows 11, CPython 3.12.7 x64 및 RTX 3060 Laptop 6GiB 환경을 사용한다. D2는 프로젝트 가상환경과 CARLA 0.9.16 Windows 패키지를 준비하고, 실제 서버와 Python API의 버전 일치 및 simulation 진행을 확인하는 단계다.

## 완료 조건

다음 조건을 모두 만족해야 D2를 완료한다.

1. 프로젝트 `.venv`에 CARLA 0.9.16 API가 설치되고 native module import가 성공한다.
2. Windows 패키지가 `carla/CARLA_0.9.16`에 추출되고 실제 서버가 실행된다.
3. 설치 metadata·native API·server RPC로 확인한 package·client·server 버전이 모두 정확히 `0.9.16`이다.
4. Map과 world snapshot을 읽고, asynchronous world에서 **3회**의 새로운 tick을 관찰한다. Frame 번호와 simulation 시간이 증가하고 timestep이 양수여야 한다.
5. 실행 명령, 설치 정보 및 실제 연결 결과가 남아 있다.

Import 성공, 실행 중인 process 또는 TCP listener만으로 이 gate를 통과하지 않는다. 서버 누락, 버전 불일치, RPC 실패 및 `PENDING` 상태에서는 D2가 완료되지 않는다. RGB 시험은 추가 확인이며 waypoint 라벨이나 차선 주행 검증을 대신하지 않는다.

## 고정한 설정과 저장 위치

| 항목 | 값·파일 |
| --- | --- |
| 환경 manifest | [config/environment.json](../config/environment.json) |
| Python | CPython 3.12 x64, 프로젝트 `.venv` |
| API dependency | [requirements-d2.txt](../requirements-d2.txt): `carla==0.9.16`, wheel SHA-256 고정 |
| 서버 배포 | 공식 Windows `CARLA_0.9.16.zip`, 예상 길이 7,812,252,810 bytes |
| ZIP 보관 | `carla/CARLA_0.9.16.zip` |
| 설치 폴더 | `carla/CARLA_0.9.16` |
| 첫 실행 | Town01, Low, off-screen, sound 비활성 |
| 연결 | `127.0.0.1:2000` RPC, streaming port 2001 |
| 자동 산출물 | `results/local/`, Git에서 제외 |

프로젝트 환경의 dependency는 현재 CARLA API만 고정한다. PyTorch, OpenCV 및 학습 dependency는 해당 구현 단계에서 추가한다. 글로벌 Python 환경이나 기존 Kaggle 환경의 dependency를 이 환경의 확정 버전으로 취급하지 않는다.

Windows 공식 0.9.16 소스 설정의 기본 graphics API는 DX12다. 시작 script는 graphics API를 강제하는 flag 없이 배포 기본 설정을 사용한다. 실제 선택된 RHI는 서버 실행 로그로 확인한다. Linux/Kaggle의 Vulkan 설정을 Windows에 그대로 적용하지 않는다. [공식 Windows 설정](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Config/DefaultEngine.ini#L98-L104).

`-RenderOffScreen`은 화면 표시 없이 GPU 렌더링을 수행하는 설정이다. `no_rendering_mode=True`는 카메라에 빈 데이터를 반환하므로 RGB 수집에 사용하지 않는다. [CARLA 렌더링 옵션](https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/).

## 1. 설치

저장소 루트에서 실행한다. 설치 전에는 ZIP, 추출 파일 및 작업 여유를 합쳐 **최소 35GiB**를 확보한다. 설치 도구는 ZIP의 실제 비압축 크기에 필요한 공간도 별도로 확인한다. D1에서 확인한 디스크 여유는 실행 때마다 달라질 수 있다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_d2.ps1
```

[setup_d2.ps1](../scripts/setup_d2.ps1)은 `.venv`를 만들고 CPython 3.12 x64 여부를 확인한 뒤, hash가 고정된 binary API wheel을 설치하고 native import를 검증한다. 이어 [install_carla_windows.py](../scripts/install_carla_windows.py)가 ZIP을 다운로드·추출한다. API만 준비하려면 같은 명령에 `-SkipSimulator`를 추가한다. 이 옵션의 성공은 서버 설치나 연결 성공을 뜻하지 않는다.

API wheel hash는 [PyPI CARLA 0.9.16 metadata](https://pypi.org/pypi/carla/0.9.16/json)에 공개된 값이다. 다운로드한 API wheel을 `pip --require-hashes`로 검증한다. 서버 ZIP은 공식 배포 주소와 예상 길이를 확인하고 SHA-256을 계산해 설치 기록에 남긴다. 이번에 측정한 ZIP hash `0972f26fa3f1c741fbfcb4602b4ccf520415b8f8cf3d06cb83c9502fe8118bd2`를 manifest에 고정했으며, 재실행 시 일치를 요구한다. **같은 다운로드 파일에서 계산해 고정한 ZIP hash는 이후 파일의 동일성 확인용이며, 독립적인 공식 서명 검증은 아니다.** ZIP 내부 파일은 추출할 때 CRC를 검사한다.

ZIP은 복구와 재설치에 사용하도록 보관하며 ZIP·설치 폴더는 모두 `/carla/`의 Git 제외 규칙에 해당한다. 설치 결과는 `results/local/carla_install.json`, 완료 marker는 `carla/CARLA_0.9.16/.vlf-install.json`에 저장한다. 정상 설치와 같은 ZIP을 다시 실행하면 marker와 두 서버 executable을 확인하고 추출을 생략한다. ZIP checksum 검사는 보관한 배포 파일의 동일성을 확인하며, 추출 후 수정하는 `DefaultEngine.ini`까지 원본 그대로임을 보증하지 않는다.

## 2. 서버 시작과 상태 확인

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Start
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Status
```

[carla_server.ps1](../scripts/carla_server.ps1)은 고정된 프로젝트 폴더의 launcher와 shipping executable을 사용하고, 화면을 숨겨 off-screen으로 실행한다. 실행 상태는 `results/local/carla_server_state.json`, 표준 출력과 오류는 같은 폴더의 `carla_server_stdout.log`와 `carla_server_stderr.log`에 남긴다. 엔진 자체 로그는 배포 폴더의 `CarlaUE4/Saved/Logs/`도 확인한다.

이번 Windows Shipping 배포의 첫 실행에서는 Town01 positional map argument가 실제 map에 반영되지 않아 `Town10HD_Opt`가 로드됐다. 따라서 실행 argument만 보고 작은 map으로 시작했다고 판정하지 않는다. 시작 script는 배포 폴더의 `CarlaUE4/Config/DefaultEngine.ini`에서 `GameDefaultMap`을 `/Game/Carla/Maps/Town01.Town01`로 고정하는 방법을 추가했다. Unreal의 project config는 설정 계층의 일부이며 실제 적용 결과는 서버 RPC의 map 값으로 확인한다. [Epic 설정 파일 계층](https://dev.epicgames.com/documentation/en-us/unreal-engine/configuration-files?application_version=4.27).

이 변경은 `GameDefaultMap`이 정확히 한 개이고 공식 기본값 `/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt` 또는 이미 고정한 Town01인 경우에만 진행한다. 처음 변경할 때 원본을 `results/local/carla_default_engine_original.ini`에 보관한다. 다른 값이나 모호한 설정은 보존하고 시작을 거부하며, AppData·사용자 profile의 설정은 수정하지 않는다. **고정 후 실제 RPC와 RGB 시험에서 Town01을 확인했다.** [실제 연결 결과](evidence/carla_windows_connection_2026-10-06.json).

시작 script는 두 TCP listener가 생길 때까지 최대 **45초** 기다린다. `Start`의 성공이나 process 존재는 RPC 준비 완료를 뜻하지 않는다. 첫 시작은 map과 shader 준비로 60초 이상 걸릴 수도 있다. 시작 대기나 첫 연결이 timeout이면 보존한 상태·로그를 확인하고 초기화 시간을 준 뒤 상태 및 연결 검사를 다시 실행한다. Boot 직후의 짧은 timeout 하나로 환경 실패를 확정하지 않는다. 최종 재실행에서는 `Start`와 실제 연결 검사까지 합쳐 약 5.84초에 통과했으며, 이 값은 해당 재실행의 측정치다.

이미 이 script가 시작한 서버가 실행 중이면 중복 서버를 만들지 않는다. Port 2000 또는 2001이 다른 process에 사용 중이면 시작을 중단하며 해당 process를 임의로 종료하지 않는다.

## 3. 실제 API 연결과 tick 검증

```powershell
.\.venv\Scripts\python.exe scripts/check_carla_connection.py --timeout 60 --ticks 3 --output results/local/carla_connection.json
```

[check_carla_connection.py](../scripts/check_carla_connection.py)는 CARLA API를 실제 import하고 RPC를 호출한다. Package·client·server 버전, map, snapshot 및 world 설정을 읽은 뒤 서버가 스스로 진행하는 tick 3회를 관찰한다. Map을 바꾸거나 actor를 생성하지 않고, world 설정 변경 및 `world.tick()`도 호출하지 않는다.

| 출력 | 의미와 다음 작업 |
| --- | --- |
| `PASS`, 종료 코드 0 | 버전 일치 및 tick 진행 검증 성공. 설치·실행 기록과 함께 D2 gate 확인 |
| `FAIL`, 종료 코드 1 | JSON의 `error.stage`와 `error.code` 확인; 초기화·설치·버전·연결 문제를 해결하고 재검증 |
| `PENDING`, 종료 코드 1 | 다른 client가 synchronous world를 관리 중. 소유자의 작업을 보존하고, 별도의 asynchronous 검증 시점에 다시 실행 |

`PENDING`에서 성공을 표시하거나 동기 world를 임의로 진행시키지 않는다. Connection `PASS`도 RGB가 렌더링되거나 waypoint가 올바르다는 의미는 아니다.

## 4. 선택 RGB smoke test

기존 RGB 진단 도구를 프로젝트 환경에서 재사용할 수 있다. 서버가 준비되어 있고 다른 수집·제어 client가 world를 사용하지 않는 상태에서 실행한다. 이 도구는 시험 차량·센서를 만들고 world를 임시 동기 모드로 설정하므로 읽기 전용 연결 검사와 다르다.

```powershell
.\.venv\Scripts\python.exe scripts/carla_rgb_probe.py --output-dir results/local/carla-rgb-smoke --timeout 60 --warmup 5 --frames 20 --reuse-current-world
```

[carla_rgb_probe.py](../scripts/carla_rgb_probe.py)는 640×360 RGB, 0.05초 step, warmup 5장 및 측정 20장을 사용한다. `--reuse-current-world`는 시작 script로 준비한 현재 Town01을 사용하고, 실제 map이 다르면 실패한다. 이번 로컬 실행에서는 map 재로드를 피하기 위해 이 옵션을 사용한다. 각 tick의 frame과 이미지 frame이 일치하는지, BGRA 길이와 RGB 픽셀 변화 및 PNG 출력을 검사한다. 종료 시 시험 actor를 정리하고 이전 world 설정을 복구한다. 결과는 출력 폴더의 `carla_rgb_report.json`과 `carla_rgb_preview.png`다. 실패 시 JSON에 기록된 stage와 정리 결과를 확인한다.

이 시험은 정지 차량의 짧은 렌더링·수집 검사다. 차량 자세와 waypoint 라벨의 정합, 장시간 수집, controller 또는 모델의 주행 성능은 D3 이후 해당 gate에서 확인한다.

## 5. 서버 종료

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Stop
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Status
```

종료 script는 기록한 PID, executable 경로와 생성 시각이 모두 일치하는 이 실행의 process만 종료한다. PID가 재사용되거나 process 정보를 확인할 수 없으면 종료를 거부하고 상태 기록을 보존한다. 실행 중인 다른 simulator의 소유권을 가져오지 않는다.

## 실패 복구와 로컬 검증

불완전한 `.zip.part`, 예상과 다른 ZIP 길이, marker 없이 내용이 남은 설치 폴더 및 잘못된 기존 `.venv`는 자동 덮어쓰기하지 않는다. 먼저 오류에 표시된 정확한 파일·폴더와 로그를 확인한다. 복구가 필요하면 해당 프로젝트 내부의 대상만 처리하고, 정상 ZIP 및 다른 설치·가상환경을 보존한 뒤 설치 명령을 재실행한다. 포괄적인 process 종료나 폴더 삭제를 복구 절차로 사용하지 않는다.

로컬 단위 검증은 simulator 없이 실패 판정과 설치 방어 로직을 확인한다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

현재 로컬 단위 검증 **70개가 모두 통과했고 skip은 0개**다. 단위 검증이 통과해도 실제 RPC 시험을 실행해야 한다. 현재 계정의 GitHub Actions 사용 제한으로 원격 CI는 통과로 판정하지 않는다. 로컬 실행 결과와 원격 workflow 결과를 구분한다.

Kaggle는 [기존 검증](KAGGLE_CARLA_VERIFICATION.md)과 [AGENTS.md](../AGENTS.md)의 재사용 원칙을 유지한다. 로컬 D2 gate가 통과하면 확인만을 위해 추가 Kaggle 작업을 실행할 필요가 없다. 원격 진단이 필요해도 아래 기존 ID·제목을 유지하고, 제출 전에 접근 가능 여부를 읽기 전용으로 확인한다. 접근 실패 시 새 노트북 생성으로 대체하지 않는다.

- CARLA: `dongmyungpark/vlf-carla-rgb-smoke-20261005`
- GPU·Vulkan: `dongmyungpark/vlf-carla-vulkan-preflight-20261005`
- 다운로드 진단: `dongmyungpark/vlf-carla-download-probe-20261005`

## 실제 결과와 다음 단계

**D2 gate: PASS.** 로컬 Windows 서버와 프로젝트 가상환경을 실제 연결하고, RGB 렌더링까지 확인했다. [D2 검증 요약](evidence/carla_windows_d2_2026-10-06.json)에 버전, 설정, 종료 결과 및 적용 범위를 기록했다.

| 검증 항목 | 실제 결과·증거 |
| --- | --- |
| Windows 설치 | ZIP 7,812,252,810 bytes, hash 일치, 추출 완료; [설치 기록](evidence/carla_windows_install_2026-10-06.json) |
| Python 및 버전 | CPython 3.12.7 x64, package·client·server 모두 0.9.16; [실제 RPC](evidence/carla_windows_connection_2026-10-06.json) |
| Map 및 tick | 실제 Town01, asynchronous tick 3회, frame·simulation time 증가; RGB 종료 후에도 재확인 |
| RGB | 640×360 측정 20장, world·sensor frame 22094~22113 모두 일치, 유효 픽셀 및 PNG 저장; [RGB 기록](evidence/carla_windows_rgb_2026-10-06.json), [실제 이미지](evidence/carla_windows_rgb_2026-10-06.png) |
| 수집 루프 | 0.345313초에 20프레임, 57.919 FPS; tick부터 RGB 수신까지 평균 6.473ms, 최대 7.257ms |
| GPU 메모리 | 전체 장치의 수집 전후 snapshot 3,156 / 3,172MiB, total 6,144MiB; process 사용량이나 peak 측정은 아님 |
| RGB 정리 | `cleanup_complete=true`, 시험 actor 정리·이전 asynchronous world 복구 후 RPC 재검증 통과 |
| 서버 종료 | 이 실행의 process 종료, 소유 상태 파일 제거, port 2000·2001 listener 없음 확인 |
| 로컬 품질 검사 | 단위 검증 70개 통과·0 skip, PowerShell syntax, strict 환경 사전 검사 및 `pip check` 통과 |

RGB PNG를 열어 실제 도로 장면임을 확인했다. 수집 처리율은 **정지 차량의 1초 simulation 구간**에서 RGB 검사를 포함한 loop의 측정값이다. Warmup, PNG 저장 및 GPU snapshot 시간은 제외되며, 모델 추론·제어 또는 장시간 주행 성능을 나타내지 않는다.

성공 실행에서는 새로 시작한 Town01을 재사용했다. 이전 시험의 `load_world("Town01")` 중 서버 process가 종료됐고 `world-load` 단계에서 실패했다. [실패 기록](evidence/carla_windows_reload_failure_2026-10-06.json)을 보존했으며 원인은 확인하지 못했다. GPU 메모리 부족이나 DX12 결함으로 단정하지 않는다. Map 재로드와 장시간 안정성은 D3 이후 추가 검증 항목이다.

공식 소스와 배포 설정의 DX12 기본값은 확인했지만 실제 engine RHI 로그는 확보하지 못했다. 또한 서버 종료 검증은 소유 process와 listener가 사라졌다는 결과이며, `Stop`의 강제 종료 fallback을 포함하므로 graceful engine shutdown을 검증한 것으로 표시하지 않는다. 원격 CI는 계정에서 비활성화되어 실행하지 못했고, 이번 작업에서 새 Kaggle 작업이나 노트북을 만들지 않았다.

최종 구성의 재현·검증 순서는 다음과 같다. 연결 검사는 RGB 전후에 수행하고 마지막에 서버를 종료한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_d2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Start
.\.venv\Scripts\python.exe scripts/check_carla_connection.py --timeout 30 --ticks 3 --output results/local/carla_connection.json
.\.venv\Scripts\python.exe scripts/carla_rgb_probe.py --output-dir results/local/carla-rgb-smoke --warmup 5 --frames 20 --reuse-current-world
.\.venv\Scripts\python.exe scripts/check_carla_connection.py --ticks 3 --output results/local/carla_connection.json
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Stop
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/carla_server.ps1 -Action Status
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/doctor.py --carla-root carla/CARLA_0.9.16 --strict
```

D3에서는 고정 차량·카메라와 재실행 가능한 동기 수집 config를 구현한다. D4~D7의 frame별 자세·라벨 검증이 끝나기 전에 본 수집이나 모델 학습으로 넘어가지 않는다.
