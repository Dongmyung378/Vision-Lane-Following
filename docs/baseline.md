# RGB 차선 검출 기준 모델

이 기준 모델은 RGB 영상 한 장으로 차량 기준 3·6·9·12·15m 앞의 차선 중심점을 예측한다. GPU나 CARLA Python API를 불러오지 않는 CPU OpenCV 모듈이다. 예측 입력은 `uint8` RGB 배열 또는 RGB 바이트이며, 차량 위치·실제 카메라 자세·지도·경로·정답 waypoint를 받지 않는다. 속도는 예측 모듈의 입력에 포함되지 않으며 별도 공통 controller에서 사용한다.

## 설치와 실행

```powershell
python -m pip install -r requirements-baseline.txt
python scripts/predict_baseline.py --image results/local/input.png
```

JSON과 overlay 기본 경로는 `results/local/baseline/`이다. CLI의 출력은 `results/local/` 또는 `docs/local/`에만 저장한다. 입력 원본이나 설정 파일을 출력으로 덮어쓸 수 없다. 기존 출력도 덮어쓰지 않으므로 다시 실행할 때는 `--output`과 `--overlay`로 새로운 경로를 지정한다.

```python
from carla_tools.baseline import OpenCVLanePredictor, load_baseline_config

predictor = OpenCVLanePredictor(load_baseline_config())
prediction = predictor.predict(rgb, width=640, height=360)
if prediction["status"] == "PASS":
    waypoints = prediction["waypoints_ego_m"]  # shape (5, 2), metres
else:
    reason = prediction["error"]
```

`PASS`는 해당 영상에서 설정된 제약을 만족하는 예측을 만들었다는 뜻이다. 주행 성공·정답 오차·안전 성능을 보장하는 결과로 해석하지 않는다. 실제 주행 평가는 별도로 수행한다. 예측 실패 시 caller는 감속·정지하고 실패 원인을 집계해야 한다.

## 고정 카메라와 좌표 계약

`configs/baseline.json`은 640×360, 수평 FOV 90°, 차량 actor 기준 카메라 위치 `(1.5, 0, 2.4)m`, pitch −12°를 사용한다. CARLA 카메라의 전방·오른쪽·위 좌표를 optical 오른쪽·아래·전방으로 변환한다. 역투영은 actor의 `z=0`이 도로 평면이라는 고정 가정으로 계산한다. 실제 simulator pose 또는 프레임별 카메라 calibration으로 이를 보정하지 않는다.

출력 원점은 `vehicle_actor_transform`이며 x는 전방, y는 오른쪽, 단위는 m이다. 곡선을 따라 누적한 거리에서 3·6·9·12·15m 지점을 보간한다. 곡선에서 x를 각각 3·6·9·12·15m로 고정하지 않는다. 역투영과 overlay 모두 동일한 명시적 nominal camera를 사용한다.

## 차선 추정

HSV에서 노란색과 흰색 도색 후보를 추출하고, 도로 평면으로 역투영한 후보만 유지한다. 노란색 차선은 영상 연결 요소별 metric 중심과 PCA 방향을 구한다. 작은 색 잡음과 둥근 물체를 제외하고, 차량 앞 왼쪽의 도색부터 전방의 일관된 요소를 연결한다. 가까운 경계 후보들이 경쟁하면 예측을 거부한다. 후속 요소의 색상은 가장 가까운 관측 도색의 HSV hue와 비교하며 특정 지도나 도색의 정확한 색을 조회하지 않는다. Hue 차이가 7 OpenCV 단위보다 크면 다른 색 물체로 제외한다. 밝기보다 hue 일관성을 사용하므로 동일 색 도색의 그림자를 허용한다.

요소 사이를 도색 방향으로 제한한 cubic Hermite 곡선으로 연결한다. 영상 하단에 잘려 방향을 측정하기 어려운 가까운 도색은 두 개의 관측 중심과 먼 도색에서 측정한 접선으로 국소 원의 가까운 접선을 구한다. 원의 두 접선은 두 중심을 연결한 chord에 대해 대칭이라는 성질을 사용하며 직선 극한도 동일한 식으로 처리한다. 후보의 전방 거리·추정 접선·곡률·색 일관성을 먼저 확인하며 임의의 가까운 배경 색 물체를 추가 기준점으로 삼지 않는다. 별도의 세 번째 도색이나 simulator 정답을 필요로 하지 않는다. 이 모델과 추정 곡률은 `diagnostics`에 명시한다.

근거리 카메라 사각지대는 가장 가까운 도색의 관측 방향을 뒤로 연장하는 가정을 사용한다. 먼 영역은 설정된 짧은 거리까지만 모델 연장을 허용한다. 이 가정도 `diagnostics`에 남는다. 이미지에서 중심선 전체를 직접 관측했다는 의미는 아니다.

노란 경계 한 개가 왼쪽에서 관측되면 lane-width prior의 절반을 곡선의 오른쪽 법선으로 이동한다. 기본 prior는 4m다. 외측 경계는 직접 관측한 것으로 표현하지 않으며, diagnostics에 `inferred_boundary="right"`, `lane_width_source="fixed-nominal-prior"`, `confidence="single-boundary-prior"`를 기록한다. 이 폭을 실제 지도에서 읽지 않는다.

노란색 경계 추정은 서로 분리된 도색 요소가 최소 두 개 있고 전방 관측 범위가 7m 이상인 점선을 대상으로 한다. 하나의 연결 요소가 되는 노란 실선은 충분한 지지를 얻지 못해 실패할 수 있다. 노란색 후보가 우선 선택된 뒤 이 검사를 통과하지 못하면 흰색 경계로 다시 시도하지 않는다.

노란색 지지가 없으면 좌우 흰 도색 쌍을 사용한다. 거리 구간별 median, 고정 seed의 RANSAC, 반복 재적합으로 cubic 또는 quadratic 곡선을 구한다. 관측 폭이 설정 범위를 벗어나거나 충분한 두 경계를 찾지 못하면 실패한다. 정답이나 마지막 성공 출력을 반환하는 fallback은 없다.

## 실패와 재현성

가까운 경계 부재, 짧은 관측 거리, 색 잡음과 분기, 반대편 차선, 비현실적인 폭, 과도한 곡률·방향, 15m 지지 부족을 실패로 구분한다. 급한 곡선에서 한 개의 멀리 있는 도색만 보이는 영상도 실패할 수 있다. 이 모델은 평탄한 도로와 설정된 카메라를 가정하므로 경사·차체 pitch/roll·가려짐·다른 차선 폭에 한계가 있다.

예측은 영상 사이의 상태를 유지하지 않는다. OpenCV는 CPU 한 스레드로 실행하고 OpenCL을 끈다. 같은 영상과 같은 설정은 동일한 waypoint·diagnostics를 생성하고 `timing_ms`만 달라질 수 있다. 설정은 필수 키·중복 키·비유한 수·잘못된 자료형을 거부하고 SHA-256으로 식별한다. RGB, 설정과 예측을 로컬에 저장하면 GPU 없이 독립적으로 다시 예측할 수 있다.

`scripts/run_baseline_pipeline.py`의 현재 개발 프로토콜은 직선·좌회전·우회전 route의 중앙 시작 상태에서 B0와 privileged reference를 비교한다. 횡방향·yaw 편차의 recovery 계산은 평가기에 구현되어 있으나, 이 실행기는 해당 초기 편차 주행을 포함하지 않는다.

synthetic test는 독립적인 물리 도색 rasterizer로 직선·좌우 곡선·횡오프셋·yaw·그림자·색 잡음·흰색 경계 폭·빈 도로·반대편 차선과 배열/바이트 replay를 검사한다.

```powershell
python -m unittest tests.test_baseline -v
```
