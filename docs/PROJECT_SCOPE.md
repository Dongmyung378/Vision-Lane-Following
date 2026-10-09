# 프로젝트 범위와 시스템 계약

전방 RGB로 목표 차선의 미래 중심 경로를 예측하고, 공통 controller를 사용하는 방법들의 closed-loop 주행 성능을 비교합니다. 이 문서는 시스템의 입력·출력과 비교 원칙을 정의합니다.

## 목표 시스템과 정보 사용

```text
전방 RGB → 경로 예측기 → 차량 좌표계 waypoint 5개 → controller → steering
                                                        ↑
                                                   차량 속도 피드백
```

`camera-only`는 경로 예측기의 입력이 전방 RGB라는 의미입니다. 차량 속도는 저수준 속도 제어와 controller의 속도 의존 계산에 사용합니다.

지도, 세계 좌표의 차량 자세, 정답 경로, 충돌 센서와 차선 침범 센서는 라벨 생성과 평가에 사용합니다. Camera-only 경로 예측기에는 이 정보를 전달하지 않습니다. 정답 waypoint를 사용하는 주행은 **privileged reference**로 표시하고, 라벨·controller의 개발 QA와 인지 오차 분석에 사용합니다.

현재 구현된 주행 pipeline은 privileged reference를 실행합니다. OpenCV 경로 예측 baseline과 RGB waypoint 학습 모델은 별도의 구현 대상입니다.

## 실행 범위

| 항목 | 조건 |
| --- | --- |
| Simulator | CARLA 0.9.16과 동일 버전의 Python API |
| 개발 지도 | Town01 |
| 차량과 카메라 | 고정 차량 1종, 전방 RGB 카메라 1개 |
| Route | 분기 없는 직선, 좌곡선과 우곡선 |
| 환경 | 맑은 날씨, 다른 차량과 보행자가 없는 장면 |
| 예측 목표 | 처음 선택한 목표 차선의 미래 중심 경로 |
| Recovery | 같은 목표 차선을 유지하며 작은 횡방향·방향 편차에서 복귀 |
| 결과물 | 수집·라벨·제어·평가 도구와 camera-only 경로 예측 방법 |

교차로 경로 선택, 차선 변경, 객체 회피, 보행자·신호등 대응 및 실제 차량 적용은 포함하지 않습니다. 15m waypoint horizon 안에 교차로, 갈림길이나 차선 종료가 들어오는 sample은 초기 데이터에서 제외합니다. 수집 episode도 같은 조건에 닿기 전에 종료합니다.

## 수집과 출력 계약

| 항목 | 계약 |
| --- | --- |
| 실행 주기 | Synchronous mode, fixed delta 0.05초 (20Hz) |
| RGB | 640×360, 수평 FOV 90도; mount 위치·회전은 설정에 기록 |
| 차량 기하 | Blueprint, actor origin, wheelbase·rear axle offset·전륜 간격·최대 조향각 기록 |
| Waypoint 거리 | 목표 차선 중심선의 누적 길이를 따라 3·6·9·12·15m |
| 출력 | `5 × 2`, 각 점 `[x, y]`, 단위 m |
| 좌표계 | 해당 frame의 차량 actor transform이 원점; x는 전방, y는 우측 |
| Controller | Pure Pursuit, 공통 조향 제한·PI 속도 제어·곡률 preview 속도 계획 |
| 속도 정책 | Nominal 상한 25km/h; 같은 곡률 기반 감속 정책을 모든 비교 방법에 적용 |

Waypoint 거리는 곡선을 따른 누적 거리이며, 차량 x 좌표를 고정한 값이 아닙니다. 차량 actor 원점과 controller의 rear axle 기준점을 구분하고, 실제 native wheel 좌표에서 측정한 기하를 사용합니다.

RGB, 차량 자세와 waypoint는 **같은 simulator frame**에 대응합니다. Frame의 예측으로 만든 명령은 다음 frame에 적용하고 readback으로 확인합니다. 초기 편차를 준 차량의 라벨은 고정된 목표 차선의 세계 경로를 해당 차량 자세로 변환합니다. 매 frame 가장 가까운 차선으로 목표를 변경하지 않습니다.

파일 schema, route provenance와 라벨 sidecar 규칙은 [데이터 계약](data-contract.md), 좌표 변환과 투영은 [Waypoint 라벨](waypoint-labels.md)에 설명합니다.

## 비교와 평가 원칙

- OpenCV baseline과 학습 모델은 같은 waypoint 출력 형식, controller, 속도 정책, 주기와 조향 제한을 사용합니다.
- Route·episode·scene group 단위로 train/dev/final test를 나눕니다. 동일 장면의 중심 상태와 recovery sample은 같은 split에 둡니다.
- 먼저 작은 주행 pilot으로 수집·학습·제어 연결과 비용을 확인하고, 필요한 데이터 규모를 결정합니다.
- 모델과 설정 선택은 train/dev에서 완료합니다. Final test에는 동결된 설정을 적용하고 결과로 모델이나 평가 조건을 다시 선택하지 않습니다.
- 실패 episode를 포함해 완주율, 횡방향 오차, 차선 침범, 충돌, 복귀 성능과 latency를 계산합니다.
- 동기 simulator 주행과 처리 시간의 측정을 구분합니다. 동기 실행의 성공만으로 실시간성을 판단하지 않습니다.

Town02는 별도의 final test 지도 후보이며, 학습·개발 경로와 분리합니다. 평가 지표·종료 규칙과 적용 상태는 [평가 명세](evaluation.md), privileged 개발 QA의 세부 판정은 [주행과 복귀](driving.md)에 설명합니다.
