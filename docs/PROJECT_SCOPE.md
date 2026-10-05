# 프로젝트 범위와 시작 계약

확정일: 2026-10-05 (D1). 전체 일정과 주간 gate는 [ROADMAP.md](ROADMAP.md)를 따른다. 아래 값은 첫 실행의 시작 설정이다. 성능을 측정한 뒤 개발 단계에서 변경할 수 있으며, 변경 이유와 설정 버전을 기록한다.

## 목표 시스템

```text
전방 RGB 영상 → 차량 좌표계의 차선 중심 waypoint 5개 → controller → steering
                                                        ↑
                                                   차량 속도 피드백
```

카메라 영상에서 경로를 예측하는 모델은 RGB만 입력받는다. 차량 속도는 저수준 속도 제어와 controller의 속도 의존 계산에 사용할 수 있다. 따라서 `camera-only`는 경로 예측기의 입력을 뜻한다. 시스템 전체의 모든 입력이 카메라라고 주장하지 않는다.

지도, 세계 좌표의 차량 자세, 정답 경로, 충돌 센서 및 차선 침범 센서는 라벨 생성과 평가에 사용한다. 이 정보가 B0 또는 M1의 경로 예측 입력으로 전달되지 않도록 데이터와 실행 경로를 분리한다. 정답 waypoint로 주행하는 실험은 별도의 **privileged reference**로 표시한다.

## 핵심 범위

| 포함 | 시작 조건 |
| --- | --- |
| Simulator | Windows의 CARLA, 로컬 수집 및 주행 평가 |
| 차량과 카메라 | 고정 차량 1종, 고정된 전방 RGB 카메라 1개 |
| Route | 분기 없는 직선, 좌커브 및 우커브 |
| 초기 환경 | 맑은 날씨, 다른 차량과 보행자가 없는 장면 |
| 학습 목표 | 현재 목표 차선의 미래 중심 경로를 예측 |
| Recovery | 같은 목표 차선을 유지하면서 작은 횡방향 및 yaw 편차에서 복귀 |
| 결과물 | 수집·학습·주행 코드, 데모, 공정한 평가, 실패 사례 및 재현 안내 |

교차로 경로 선택, 차선 변경, 객체 회피, 보행자·신호등 대응 및 실제 차량 적용은 핵심 범위 밖이다. waypoint의 15m 범위 안에 교차로, 갈림길 또는 차선 종료가 들어오는 sample은 초기 데이터에서 제외하고, 수집 episode도 이 조건에 닿기 전에 종료한다. Simulator 결과를 일반적인 실제 도로 자율주행 성능으로 해석하지 않는다.

## 수집과 출력의 시작 설정

| 항목 | 계약 |
| --- | --- |
| CARLA | 0.9.16을 우선 검토하고 D2에서 설치 패키지와 Python API의 동일 버전 연결을 검증한 뒤 고정 |
| 실행 | synchronous mode, fixed delta 0.05초 (20Hz) |
| RGB | 640×360, 수평 FOV 90도; mount 위치와 회전은 설정에 기록 |
| 차량 | Blueprint, actor origin, wheelbase 및 제어 기준점 offset을 실제 차량 선택 후 기록 |
| Waypoint | 목표 차선 중심선의 누적 길이를 따라 3·6·9·12·15m 앞의 점 |
| 출력 | `5 × 2`, 각 점은 `[x, y]`, 단위 m |
| 좌표계 | 해당 frame의 차량 actor transform이 원점; x는 전방, y는 우측 |
| 속도 | 동일한 목표 속도 25km/h; 모든 비교 방법에 공통 적용 |
| 초기 controller | Pure Pursuit; 속도 제어와 steering 제한을 공통 사용 |
| 첫 학습 모델 | 사전학습 ResNet18 encoder + waypoint regression head, Smooth L1 loss |

Waypoint의 3·6·9·12·15m는 곡선을 따른 거리이며 차량 x 좌표를 고정한 값이 아니다. 차량 actor 원점과 Pure Pursuit 제어 기준점을 혼동하지 않는다. Offset과 wheelbase는 D8 구현 전에 실제 차량에서 확인한다.

이미지, 차량 자세 및 waypoint는 **같은 simulator frame**에 대응해야 한다. 최근 이미지에 현재 자세를 임의로 붙이지 않는다. 차량을 이동·회전시킨 recovery sample은 같은 목표 차선의 세계 경로를 새 차량 자세로 다시 변환한다. 매 frame의 가장 가까운 차선으로 목표 차선을 바꾸지 않는다.

## 데이터와 평가 원칙

- 먼저 2,000~5,000프레임 pilot으로 수집 → 학습 → 주행 연결을 확인한다. 비용을 측정한 뒤 데이터 규모를 늘린다.
- Route, episode 및 scene group 단위로 train/dev/final test를 나눈다. 같은 장면의 central과 recovery sample은 같은 split에 둔다.
- B0 OpenCV와 M1 waypoint 모델은 같은 출력 형식, controller, 목표 속도, 주기, steering 제한과 smoothing을 사용한다.
- Privileged reference는 라벨과 controller의 검증 및 인지 오차 분석용이다. Camera-only 방법과 구분해 보고한다.
- D42까지 train/dev에서 선택을 완료하고, D43부터 동결된 설정으로 final test를 실행한다. Final test 결과로 모델이나 평가 조건을 다시 선택하지 않는다.
- 실패 episode까지 포함해 완주율, 횡방향 오차, lane invasion, collision, 복귀 성능 및 latency를 기록한다. Simulator의 동기 주행 성공만으로 실시간성을 주장하지 않는다.

데이터 라벨 schema, route 생성 규칙 및 종료 threshold는 D5 이후의 구현 단계에서 확정한다. 실제 데이터, 모델 성능 또는 주행 가능 여부는 D1에서 검증한 결과가 아니다.

## D1 완료 기준과 다음 단계

D1은 범위와 입력 계약을 명시하고 하드웨어·개발 환경을 조사하는 단계다. 측정한 환경과 확인하지 못한 항목은 [ENVIRONMENT_D1.md](ENVIRONMENT_D1.md)에 기록하며, 진행 증거는 [progress.md](progress.md)에 남긴다.

D2는 CARLA 설치 경로, simulator/API 버전 일치 및 실제 연결을 검증한다. 환경 조사에 Python 모듈 또는 executable이 존재하더라도 실제 simulator 연결·GPU 학습·수집 성공을 뜻하지 않는다. D3~D7에서 동기 수집과 frame 정합을 검증하고 주간 gate를 통과해야 이후 단계로 진행한다.
