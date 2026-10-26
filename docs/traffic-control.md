# RGB 차선·교통 제어

`TrafficPolicy`는 주입받은 차선 경로 예측기, RGB 교통 인식기, 조향 제어기를 결합합니다. 차선 경로는 조향에 사용하고, 교통 감독기는 차량·보행자·신호 관측에 따라 목표 속도와 제동을 정합니다. 조향 값은 교통 감독기가 변경하지 않습니다.

정책 입력은 RGB8 영상, 영상 크기, 프레임 번호, 시각, 자차 속도입니다. 지도, 경로 정답, 차량의 세계 좌표, 시뮬레이터 객체, 실제 신호 상태는 정책에 전달하지 않습니다. 카메라 장착 위치와 내부 행렬, 차량 크기는 고정 보정값으로 별도 설정합니다.

## 구성

- `traffic_perception`: 객체 탐지, RGB 신호 색 판별, 평면 투영 거리 추정과 객체 추적.
- `traffic_tracking`: 차량 접근 속도의 일시적 큰 값은 완화하면서 최신 위험 증가에는 즉시 반응하는 선택 가능한 추적기.
- `traffic_memory`: 일시적으로 사라진 관측의 보수적 전파, 도로 옆 차량의 제한된 경로 밖 증거, 원거리 차량의 비대칭 거리 구간.
- `traffic_signal`: 미확인 신호와 색이 확인된 정지 신호의 상태 분리.
- `traffic_behavior`: 차간거리·제동거리·충돌까지의 예상 시간에 따른 목표 속도와 PI 속도 제어.
- `traffic_policy`: 위 구성 요소와 차선 제어기의 연결.
- `waypoint_mpc`: 예측 경로와 측정 차량 형상을 사용하는 선택 가능한 조향 제어기.

## 사용법

차선 예측기는 `timer()`와 `predict_profiled(rgb, timer, width=..., height=...)`를 제공합니다. 교통 인식기는 `process(RGBObservation)`, `reset()`, 고정 `calibration`과 `config`를 제공합니다. 조향 제어기는 `step(waypoints, speed, dt_seconds=.05)`와 `reset()`을 제공합니다. 모델과 장치는 호출자가 선택하고 로드합니다.

```python
from carla_tools.traffic_perception import RGBObservation
from carla_tools.traffic_policy import TrafficPolicy
from carla_tools.traffic_tracking import AdaptiveVehicleTracker

# 모델과 고정 보정값으로 생성한 교통 인식기에 추적기를 명시적으로 주입합니다.
traffic_perception.tracker = AdaptiveVehicleTracker(traffic_perception.config)
policy = TrafficPolicy(path_predictor, traffic_perception, lane_controller)
policy.reset()

# 카메라 프레임마다 같은 policy 인스턴스를 호출합니다.
observation = RGBObservation(
    rgb=rgb_bytes,
    width=640,
    height=360,
    frame=frame_number,
    timestamp=simulation_seconds,
    ego_speed_mps=measured_speed_mps,
)
control, diagnostics = policy.command(observation)
```

추적기 선택은 호출자가 명시적으로 수행합니다. 위 추적기는 차량의 유효한 속도 표본이 두 개일 때 `max(최신 접근 속도, 두 표본 평균)`을 사용합니다. 최신 접근 속도가 증가하면 즉시 반영하고, 이전의 큰 값이 남아 있을 때 그 영향을 줄입니다. 표본이 세 개일 때와 보행자는 기존 추적과 같습니다. 탐지 박스와 객체 ID를 합치거나 삭제하지 않습니다.

기본 속도 상한은 3m/s이며 제어 간격은 0.05초입니다. 영상, 속도, 시각은 같은 프레임에 대응해야 합니다. 카메라 데이터가 BGRA 형식이면 호출 전에 RGB로 변환합니다. 에피소드가 바뀌면 `reset()`으로 객체·신호·속도 제어 상태를 초기화합니다. 같은 프레임에서 차선 제어기를 별도로 다시 실행하면 제어기 상태가 두 번 진행되므로 `TrafficPolicy.command()`를 한 번 호출합니다.

출력의 `steer`, `throttle`, `brake`, `hand_brake`, `reverse`를 시뮬레이터 연결 계층에서 적용합니다. 이 정책은 시뮬레이터나 GPU 작업을 직접 시작하지 않습니다. 잘못된 입력 형식과 프레임 불일치는 예외로 거부하며, 경로 예측 실패는 정지 요청으로 처리합니다. 연결 계층은 예외를 받으면 이전 가속 명령을 반복하지 않고 정지 처리해야 합니다.

기본 증거 어댑터를 사용할 때는 `TrafficPerceptionConfig`와 `TrafficConfig`를 제공해야 합니다. 기본 어댑터는 발점 오차 3px, 카메라 높이 오차 0.15m, 피치 오차 2도, 경로 여유 0.35m, 신호 색 신뢰도 0.55, 추적 유효 시간 0.5초를 전제로 합니다. 제어기의 정지 간격 2m, 감속도 2m/s², 반응 시간 0.2초, 최소 신뢰도 0.5, 제어 간격 0.05초와 정지 표지 상태 초기화 거리 5m도 일치해야 합니다. 이 값이 다르면 생성 시 오류로 거부합니다. 생성 뒤 설정을 바꾸려면 정책을 새로 생성합니다. 차량 형상과 탐지 점수 임계값은 별도로 설정할 수 있습니다.

다른 추정·제어 모델을 연결하려면 `bridge=`로 사용자 어댑터를 명시적으로 전달합니다. 이 경우 어댑터와 인식·감독기 설정의 호환성을 호출자가 검증해야 합니다.

## 거리와 신호의 의미

`distance_m`과 기존 불확실성은 명목 거리 추정을 나타냅니다. `conservative_nearest_distance_m`은 별도로 계산한 근거리 쪽 하한입니다. 이 하한은 고정된 카메라 장착 오차와 발점 픽셀 오차, 평평한 도로 가정에 조건부로 계산합니다. 거리 구간의 `far_m: null`은 원거리 쪽 상한을 정할 수 없다는 뜻이며, 장애물이 없다는 뜻이 아닙니다. 하한은 신뢰도가 충분하고 제동·관측 누락 여유가 확보된 원거리 차량에만 제어 근거로 사용합니다. 보행자나 잘린 발점에 이 예외를 적용하지 않습니다.

미확인 신호가 보이는 동안은 정지 요청을 유지합니다. 미확인 신호가 사라지면 짧은 관측 부재 대기 후 그 불확실성만 해제합니다. 빨간색·노란색이 확인된 신호는 시간 경과만으로 해제하지 않으며, 같은 신호등과 연결되는 초록색 관측을 요구합니다. 어느 차로와 접근 방향을 제어하는 신호인지 판단하는 기능은 포함하지 않습니다.

명목 평면 거리, 탐지 누락, 가림, 신호 연결의 한계는 시뮬레이션 시나리오에서 별도로 평가해야 합니다. 완주, 충돌, 차선 유지, 필요한 정지와 재출발을 함께 확인합니다.
