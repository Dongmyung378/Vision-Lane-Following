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
