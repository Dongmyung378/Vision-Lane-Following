# Vision Lane Following 프로젝트 로드맵

전방 RGB 카메라 영상에서 미래 차선 중심 경로를 예측하고, controller로 차량을 조향하는 CARLA 프로젝트를 완성한다. 핵심 일정은 **D1부터 D56까지 8주**다. D28까지 첫 모델 기반 주행을 연결하고, D42까지 개발과 선택을 마친 뒤, D43부터 독립적인 최종 평가를 수행한다. 선택 연구 확장은 D57부터 D84까지 4주이며 하나의 주제만 진행한다.

작성일은 2026년 10월 5일이다. D1은 실제 착수일을 뜻하며 특정 날짜에 고정하지 않는다. 일정은 하루 2~3시간 집중 작업, 매주 6일 구현과 실험 및 1일 검토와 보완을 기준으로 한다. 자동 수집과 학습의 대기 시간은 별도로 확보한다. 단계의 필수 검증을 통과하지 못하면 해당 단계를 연장한다. 56일은 계획 기간이며 성능이나 완료 날짜를 보장하는 수치가 아니다.

## 프로젝트 목표와 범위

목표 시스템은 `전방 RGB 영상 -> 차량 좌표계의 차선 중심 waypoint 5개 -> controller -> steering`이다. 최종 산출물은 주행 데모, 수집과 학습 코드, 독립적인 주행 평가, 실패 사례, 재현 안내다.

차선 경로를 예측하는 모델의 입력은 RGB 영상뿐이다. 지도, 세계 좌표의 차량 위치, 정답 차선 경로, 충돌 센서와 차선 침범 센서는 라벨 생성 및 평가에만 사용한다. 저수준 속도 제어와 controller의 속도 의존 계산에는 차량 속도 피드백을 사용할 수 있다. 이 범위를 README에 명시하고, 시스템 전체의 모든 입력이 카메라뿐이라고 표현하지 않는다.

초기 버전은 고정 차량과 고정 카메라를 사용하고, 분기 없는 직선과 곡선 구간을 주행한다. 교차로 의사결정, 차선 변경, 객체 회피, 보행자와 신호등 대응, 실제 차량 적용은 핵심 범위에 포함하지 않는다. 다른 주행 차량과 보행자가 없는 조건에서 시작한다. 실제 도로의 일반 자율주행 성능으로 결과를 확대 해석하지 않는다.

## 핵심 설계와 완료 기준

| 항목 | 시작 설정과 원칙 |
| --- | --- |
| Simulator | CARLA 0.9.16 Windows 패키지와 동일 버전 Python API를 우선 검토하고, 호환 확인 후 버전을 고정한다 |
| 수집 | synchronous mode, fixed delta 0.05초를 시작 설정으로 사용한다. 센서 frame과 해당 frame의 차량 자세를 연결한다 |
| 카메라 | 전방 RGB 1개, 640×360, 수평 FOV 90도. 위치와 회전을 설정 파일에 기록한다 |
| Waypoint | 목표 차선 중심을 따라 3, 6, 9, 12, 15m 앞의 점. 거리는 차선 중심선의 누적 길이 기준이며 차량 x 좌표값을 뜻하지 않는다 |
| 좌표계 | 해당 frame의 차량 actor transform을 원점으로 사용한다. x는 전방, y는 우측, 단위는 m. 모델 출력은 5×2 좌표다 |
| Controller | Pure Pursuit로 시작한다. 차량 원점과 제어 기준점의 offset 및 wheelbase를 명시한다. Stanley는 개발 단계의 비교 후보다 |
| 속도 | 모든 방법에 동일한 25km/h 목표 속도 제어를 적용한다. 35km/h 실험은 여유가 있을 때만 추가한다 |
| 메인 모델 | 사전학습 ResNet18 encoder와 waypoint regression head, Smooth L1 loss로 시작한다 |
| 학습 실행 | 로컬에서 작은 실행을 먼저 검증하고 Kaggle로 옮긴다. 지원 Python과 PyTorch 버전, config, seed, checkpoint를 기록한다 |
| 필수 비교 | B0 OpenCV, M1 waypoint 모델. 정답 waypoint 주행은 privileged reference로 별도 표시한다 |
| 선택 작업 | B1 segmentation, 외부 데이터, temporal 모델, 추가 backbone, DDP를 핵심 일정의 필수 항목으로 두지 않는다 |

**필수 검증:** 영상과 라벨의 정합, 목표 차선 유지, 정답 waypoint controller의 안정적인 주행, 데이터 분할, 평가기 동작을 확인한다. 이 항목이 실패하면 모델 확대나 최종 평가로 넘어가지 않는다.

**성능 목표:** 개발 route의 높은 완주율, 낮은 횡방향 오차, 적은 차선 침범과 충돌, baseline 대비 개선을 목표로 한다. 완주율 90%는 개발 목표로 사용할 수 있으나, 달성하지 못했다고 결과를 숨기거나 테스트 조건을 사후 변경하지 않는다. 프로젝트 완료 여부는 실행 가능한 시스템과 신뢰할 수 있는 평가 산출물로 판단한다.

## 데이터 구성과 라벨 계약

메인 데이터는 CARLA에서 직접 수집한다. 먼저 2,000~5,000프레임의 pilot 데이터로 수집, 학습, 주행 연결을 확인한다. 이후 수집 속도, 저장 용량, epoch 시간과 route coverage를 측정해 규모를 늘린다. 학습용 40,000~60,000프레임은 확장 목표다. 프레임 수를 채우기 위해 유사한 직선 영상만 반복 수집하지 않는다.

각 sample에는 RGB 경로, frame ID, timestamp, episode ID, route ID, 목표 차선 ID, 차량 자세, camera 설정, weather, waypoint, perturbation 설정을 기록한다. 학습 입력에는 RGB만 전달하고 다른 필드는 라벨 생성, 분할 및 분석에 사용한다. 평가용 충돌과 차선 침범 event도 별도로 보존한다.

Recovery 데이터는 차량의 lateral offset과 yaw offset이 있는 상태를 포함한다. 작은 편차에서 시작하고 차량 폭과 복귀 가능 범위를 확인한 뒤 확장한다. 차량을 이동하거나 회전시킨 뒤에는 같은 목표 차선의 세계 좌표 경로를 변경된 차량 자세 기준으로 다시 변환한다. 중앙 주행의 local waypoint를 그대로 재사용하지 않는다. 매 frame에서 가장 가까운 차선으로 정답을 바꾸지 않고, 원래 route의 목표 차선을 유지한다.

Waypoint의 15m 범위 안에 교차로, 갈림길, 차선 종료가 들어오는 샘플은 초기 데이터에서 제외한다. 수집 episode도 이 조건에 닿기 전에 종료한다. 이미지의 waypoint overlay로 직선, 양방향 곡선, off-center 상태를 검증한다. 처음에는 밝기, 대비, 약한 blur 등 영상 기하를 바꾸지 않는 augmentation을 사용한다. Perspective warp와 flip은 라벨 변환 규칙을 검증한 후에만 추가한다.

| 분할 | 용도 | 사용 규칙 |
| --- | --- | --- |
| Train | 모델 파라미터 학습 | 여러 route, 곡률과 recovery 상태를 포함한다 |
| Dev | checkpoint, controller, augmentation 선택 | 학습 route와 분리된 개발 route 및 개발용 날씨와 camera shift를 사용한다 |
| Final test | 마지막 성능 측정 | 개발에 사용하지 않은 route, 별도 town과 별도 perturbation 조건을 남겨둔다 |

프레임 무작위 분할을 사용하지 않는다. Route와 episode를 묶고, 같은 장면에서 만든 central 및 recovery sample도 같은 split에 둔다. 시간적으로 인접하거나 사실상 같은 경로의 샘플이 경계를 넘지 않는지 확인한다. `split_manifest.csv`에 dataset 버전, sample, route, episode, scene group, split을 기록한다.

초기 pilot은 한 town으로 시작할 수 있다. 확장 시 학습용 3개 town과 최종 테스트용 별도 town을 목표로 하되, 설치된 맵과 수집 비용을 확인해 구성한다. 별도 town을 확보하지 못하면 unseen-route 평가로 범위를 좁혀 명시한다. Development robustness와 final robustness는 route와 조건 조합이 겹치지 않도록 사전에 정한다. 최종 테스트 영상, 예측과 결과는 D42의 동결 이후에 처음 확인한다.

TuSimple은 선택적인 2D 차선 검출 실험용이며 CULane은 이후 실제 영상의 인지 검증용이다. 기본 라벨은 이미지 안의 차선 표시선 좌표이므로, 메인 모델에 필요한 미터 단위 중심 waypoint의 직접적인 대체 데이터로 사용하지 않는다.

## 공정한 비교와 평가 프로토콜

B0는 OpenCV 차선 검출과 nominal camera calibration을 통해 차량 좌표계의 중심 waypoint를 생성한다. M1은 RGB에서 같은 형식의 waypoint를 예측한다. 두 방법은 동일한 최종 controller, 목표 속도, steering 제한, smoothing, 주행 주기를 사용한다. Controller 비교 실험은 별도 표로 남긴다. Controller나 전처리를 바꾸면 영향받는 개발 비교를 다시 수행한다.

정답 waypoint 주행은 라벨과 제어를 검증하고 인지 오차의 영향을 분석하기 위한 기준 실험이다. 지도와 차량 자세를 사용하므로 camera-only 방법과 구분해 보고한다. Camera shift 시험에서 B0에 변경된 실제 camera extrinsic을 제공하지 않고 nominal calibration을 유지한다. M1에도 camera 설정을 입력으로 추가하지 않는다. 실제 camera 설정은 정답과 평가 기록에만 사용한다.

평가 정의는 D5부터 작성하고 D14까지 구현한다. Route progress는 원래 목표 route의 누적 길이로 계산하고, 횡방향 오차도 같은 목표 차선을 기준으로 측정한다. 충돌, 차선 이탈, timeout, 장시간 정지, route 종료에 대한 episode 종료 조건과 event 중복 집계 규칙을 개발 단계에서 검증한다. 구체 threshold는 개발 실험으로 결정하고 D42에 고정한다.

| 지표 | 기록과 집계 원칙 |
| --- | --- |
| Route completion | 목표 route 길이 대비 진행 비율. 실패 episode도 포함한다 |
| Mean 및 p95 lateral error | 목표 차선에 대한 거리 m. 모든 episode를 포함하고 완주 여부와 함께 제시한다 |
| Lane invasions 및 collisions | 원본 event와 중복 제거 규칙을 저장한다. km당 비율은 전체 event 수를 전체 주행 거리로 나눠 계산한다 |
| Intervention-free distance | 개입이나 종료 이전 거리와 종료 원인을 기록한다 |
| Recovery success | 초기 편차, 복귀 허용 오차와 시간 한도를 사전에 고정한다 |
| Steering smoothness | 동일 simulation 주기에서 steering rate와 변화량을 측정한다. 정의가 불명확한 jerk라는 이름을 쓰지 않는다 |
| Latency | 동일 하드웨어에서 전처리, 모델 추론, controller 및 전체 처리의 median과 p95를 측정한다 |

동기 모드에서는 simulator가 추론을 기다리므로 주행 성공만으로 실시간성을 주장하지 않는다. 20Hz를 목표로 하면 전체 처리에 50ms 예산이 필요하다. 목표를 넘으면 실제 측정값을 보고하고, 개발 단계에서 주기를 바꾼 경우 모든 비교 방법에 동일하게 적용한 뒤 D42에 고정한다.

최종 테스트는 아래 5개 그룹을 기본안으로 한다. 각 그룹에서 사전에 지정한 2개 route와 3개 seed 또는 초기 조건을 사용하면 방법당 30개 episode다. B0, M1, privileged reference를 모두 실행하면 총 90개 episode다. Seed가 다른 실제 초기 상태나 장면이 되도록 하고, 동일 seed의 결정적 재실행 3회를 독립 반복으로 취급하지 않는다. 같은 route와 seed 조합을 모든 방법에 공통 적용한다. 확정 규모는 pilot에서 측정한 실행 비용을 보고 D42 이전에 정한다.

| 그룹 | 조건 | 평가 목적 |
| --- | --- | --- |
| A | 학습 town의 새로운 route와 맑은 날씨 | Route generalization |
| B | 별도 town의 route와 맑은 날씨 | Town generalization |
| C | 남겨둔 route의 강한 비 조건 | Weather shift |
| D | 남겨둔 route의 sunset 조건 | Illumination shift |
| E | 남겨둔 route의 camera pitch 변화 | Mounting shift |

Camera height와 35km/h 시험은 선택 항목이다. 개발용 shift와 최종 시험 shift의 설정을 구분한다. 최종 결과를 본 뒤 모델, route, 속도 또는 threshold를 다시 선택하지 않는다. 기술적인 실행 오류로 재실행한 경우 이유와 영향 범위를 기록한다. 평가 코드를 수정해 의미가 바뀌면 영향받는 모든 방법을 다시 평가한다. 결과 기반 모델 개선은 새로운 버전의 개발로 전환하고 새로운 held-out 평가를 계획한다.

## D1부터 D56까지 일별 일정

### 1주 차 환경과 라벨 검증

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D1 | 범위, camera-only 정의, 하드웨어와 작업 환경 확인 | 프로젝트 범위와 환경 조사 기록 |
| D2 | CARLA 패키지와 동일 버전 Python API 연결 및 고정 | 짧은 simulator 연결 실행과 버전 목록 |
| D3 | 고정 차량, RGB camera, synchronous mode와 fixed timestep 설정 | 재실행 가능한 수집 config |
| D4 | 센서 queue, frame별 차량 snapshot, episode logger 구현 | 이미지와 자세가 대응하는 짧은 episode |
| D5 | 목표 route와 waypoint 계약, split 및 평가 정의 초안 작성 | 라벨 schema, split manifest, 평가 명세 |
| D6 | 직선과 곡선의 waypoint 생성, 차량 좌표 변환과 이미지 overlay | 단위와 좌표축을 확인한 라벨 시각화 |
| D7 | 영상과 라벨 정합, 누락 frame, 카메라 색 채널 검토 및 보완 | 1주 차 gate 기록과 수정 목록 |

**D7 gate:** 직선과 곡선의 영상, 자세, 라벨이 같은 frame에 대응한다. Sample 누락과 중복을 검출할 수 있고, 수집 설정을 다시 실행할 수 있다.

### 2주 차 정답 경로 주행과 작은 데이터

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D8 | Pure Pursuit와 차량 제어 기준점, wheelbase, 속도 제어 구현 | 정답 waypoint를 입력받는 controller |
| D9 | 정답 경로로 직선과 곡선 주행, 조향 방향과 gain 확인 | Privileged reference 주행 기록 |
| D10 | 작은 lateral 및 yaw 편차에서 목표 차선 복귀 검증 | Recovery 라벨과 controller 검증 |
| D11 | OpenCV ROI, 차선 검출, nominal camera 보정과 중심 경로 생성 | B0의 차량 좌표 waypoint overlay |
| D12 | B0를 공통 controller에 연결하고 평가기의 종료와 event 집계 구현 | 첫 B0 주행과 자동 metric CSV |
| D13 | 2,000~5,000프레임 pilot 수집, 분할 및 비용 측정 | Pilot dataset, manifest, 수집 속도와 저장량 |
| D14 | 개발 route 3개에서 정답 경로와 B0 실행 및 평가기 검토 | 2주 차 gate와 baseline 결과 |

**D14 gate:** 정답 waypoint controller가 직선과 좌우 곡선 개발 route에서 반복 완주한다. B0와 평가기가 실행되고 실패 원인을 기록한다. Pilot의 분할 누수가 없다. B0 완주 실패는 분석 대상으로 남길 수 있으나, 정답 경로 주행 실패는 해결 후 진행한다.

### 3주 차 첫 waypoint 모델 학습

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D15 | Dataset loader, preprocessing, waypoint tensor 형식 구현 | 실제 sample batch와 입력 라벨 확인 |
| D16 | ResNet18과 5×2 regression head, loss 구현 | Forward와 backward가 동작하는 작은 실행 |
| D17 | 작은 batch를 과적합시켜 학습과 좌표 scaling 검증 | 학습 경로가 정상임을 확인한 loss 기록 |
| D18 | Kaggle 실행 진입점, AMP와 checkpoint resume 구성 | 중단 후 이어서 실행한 기록 |
| D19 | Pilot 전체 학습, photometric augmentation 시작 | 초기 checkpoint와 history CSV |
| D20 | Dev의 near 및 far 오차, 곡선과 편차별 prediction overlay 분석 | Dev 오차표와 예측 경로 이미지 |
| D21 | 모델 선택 기준과 학습 시간을 검토하고 오류 보완 | 3주 차 gate와 첫 선택 checkpoint |

**D21 gate:** 데이터와 모델의 입력 및 출력이 일치하고, 작은 batch에서 loss가 감소한다. Checkpoint 재개가 가능하며, 고정된 dev 데이터의 오차와 시각화가 남아 있다. 최종 테스트 결과를 사용하지 않는다.

### 4주 차 모델을 실제 주행에 연결

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D22 | Checkpoint를 로컬에 옮기고 preprocessing과 출력 decoding 확인 | 로컬 offline 예측과 환경 기록 |
| D23 | RGB부터 모델, waypoint, controller까지 연결 | M1의 첫 closed-loop 실행 |
| D24 | 한 개의 단순 개발 route 주행과 B0 비교 | 동일 조건의 초기 비교 결과 |
| D25 | 좌우 곡선 및 여러 시작점의 실패 episode 저장 | 실패 영상, waypoint와 steering 로그 |
| D26 | 실패를 라벨, 인지, 제어, 실행 환경 문제로 분류하고 개발 조건에서 수정 | 원인별 분석과 수정 전후 기록 |
| D27 | 전처리와 추론 및 제어 latency 측정, 전체 pipeline 재실행 | Latency 표와 새 실행 기록 |
| D28 | 초기 M1 주행 데모, 개발 지표와 4주 차 gate 검토 | 첫 주행 데모와 다음 데이터 요구사항 |

**D28 gate:** 학습된 모델이 RGB에서 waypoint와 steering을 만들어 차량을 주행시킨다. 적어도 한 개의 개발 route에서 지속적인 차선 추종을 확인하고, 실패 원인과 처리 latency를 측정한다. 진행이 불가능하면 데이터 확대에 앞서 입력, 라벨 및 controller 연결을 보완한다.

### 5주 차 데이터 확대와 recovery 비교

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D29 | Pilot 비용과 실패 분석에 따라 town, route, 곡률별 확대 계획 확정 | Dataset v1 구성과 프레임 예산 N |
| D30 | 개발 및 학습 조건에서 central과 recovery 장면을 짝지어 수집 | 같은 route 장면의 paired sample |
| D31 | 라벨 시각화, 이상 sample 제거, scene group별 분할과 snapshot 고정 | Dataset v1과 확정 manifest |
| D32 | 총 N개 central sample로 조건 A 학습 | Central 조건의 checkpoint와 config |
| D33 | 같은 총 N개 안에서 일부 central을 recovery로 대체한 조건 B 학습 | Recovery 조건의 checkpoint와 config |
| D34 | 공통 dev route, controller와 초기 조건으로 A와 B 비교 | 동등한 데이터 예산의 recovery 실험 결과 |
| D35 | 효과와 실패를 분석하고 개발 기준으로 데이터와 후보 모델 선택 | 5주 차 gate와 선택 근거 |

**D35 gate:** 동일 프레임 예산, backbone, augmentation, 학습 update 수와 seed 정책에서 central 및 recovery 비교를 완료한다. 가능한 같은 장면을 짝지으며 scene group을 split 경계에서 분리하지 않는다. 최소 1개 학습 seed로 예비 비교하고, 실행 예산이 허용하면 2~3개로 확인한다. 1개 seed 결과는 탐색적인 결과로 표시한다. 효과가 없더라도 결과와 실패 사례를 보존한다.

40,000~60,000프레임 목표가 주행 통합이나 비교 실험을 밀어내면 실제 확보한 규모로 진행한다. 개발 중 모델이 실패한 상태를 추가 수집할 경우 해당 데이터도 개발 route에서만 얻고, 실험 조건과 dataset 버전을 구분한다.

### 6주 차 개발용 강건성 검증과 최종 설정 고정

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D36 | 고정 후보 waypoint 모델로 Pure Pursuit와 Stanley 비교 | Controller별 개발 결과표 |
| D37 | Dev에서 gain, lookahead 및 공통 smoothing 선택 | 최종 controller config |
| D38 | 개발용 날씨, illumination 및 camera shift에서 B0와 M1 비교 | Development robustness 결과 |
| D39 | 개발 결과에 근거한 필요한 개선 1회 및 재학습 | 개선 전후의 개발 비교 |
| D40 | Checkpoint 선택, 최종 B0와 M1 개발 비교 및 회귀 확인 | 모델과 전처리 선택 근거 |
| D41 | 평가 비용, route 및 seed 목록, 종료 조건, metric 정의 점검 | 최종 evaluation manifest와 실행 예산 |
| D42 | 모델, controller, 전처리, 속도, threshold, 코드와 데이터 버전 동결 | Freeze manifest와 6주 차 gate |

**D42 gate:** 최종 비교 방법과 설정을 재현할 수 있다. 모든 선택은 train과 dev에만 근거한다. 최종 테스트 결과와 영상은 아직 열람하지 않았다. 성능이 부족하면 개발 단계의 범위를 줄이거나 기간을 연장하고, 테스트를 시작한 뒤 조건을 바꾸지 않는다.

### 7주 차 독립적인 최종 평가

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D43 | 동결 설정으로 새 route와 별도 town의 clean 조건 실행 | A와 B 그룹의 paired episode 기록 |
| D44 | 남겨둔 강한 비 조건에서 모든 방법 실행 | C 그룹의 결과와 종료 원인 |
| D45 | 남겨둔 sunset 조건에서 모든 방법 실행 | D 그룹의 결과와 종료 원인 |
| D46 | 남겨둔 camera pitch 조건에서 모든 방법 실행 | E 그룹의 결과와 nominal calibration 확인 |
| D47 | 누락 실행과 기술적 실행 오류 확인, 정해진 paired 반복 완료 | 완전한 episode 목록과 재실행 사유 |
| D48 | 그룹별 완주율, 오차, event 비율, latency 및 변동 집계 | 자동 생성한 최종 CSV와 비교표 |
| D49 | 결과와 실패 사례 검토, 테스트 독립성과 집계 검증 | 7주 차 gate와 최종 성능 해석 |

**D49 gate:** 미리 정한 route와 초기 조건에서 B0, M1 및 privileged reference 결과가 있다. 실패 episode를 제외하지 않고 집계하며, 테스트 결과를 보고 모델이나 조건을 바꾸지 않았다. 반복 수, 학습 seed 수와 평가 범위를 정확히 표시한다.

### 8주 차 재현과 공개 패키징

| 일차 | 핵심 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D50 | Simulator와 Python 설정, 데이터 준비 및 실행 명령 정리 | 설치와 실행 안내 |
| D51 | 결과표, 아키텍처, 모델 및 데이터 선택 이유 작성 | README와 실험 설명 |
| D52 | 대표 성공 장면 및 실패 장면 영상 제작 | Demo 영상과 설명이 있는 실패 사례 |
| D53 | 깨끗한 checkout 또는 별도 환경에서 설치와 짧은 실행 재현 | 재현 검증 로그 |
| D54 | Checkpoint, config, manifest와 결과 파일 연결 점검 | 버전과 checksum이 있는 artifact 목록 |
| D55 | 누락 안내, 깨진 경로, 결과 집계 및 문서 표현 보완 | 공개 가능한 문서와 패키지 |
| D56 | 최종 체크리스트와 완료 기준 검토, v1.0 후보 준비 | 핵심 프로젝트 완료와 공개 후보 |

**D56 gate:** 설치, 데이터 준비, 학습 및 평가의 진입점이 문서화되어 있다. 짧은 수집, checkpoint 추론, 평가 집계가 새 실행에서 재현된다. 결과표와 실패 사례가 있으며, camera-only 범위와 privileged reference를 구분한다. 데모와 코드의 버전이 일치한다. 외부 저장소 push나 release 공개는 실제 공개를 진행할 때 별도 작업으로 수행한다.

## D57부터 D84까지 선택 연구 확장

핵심 결과를 먼저 정리한 뒤 아래 중 하나만 선택한다. 기본 추천은 **고정된 데이터 예산에서 domain randomization이 환경 변화에 대한 주행 성능을 개선하는지 검증하는 실험**이다. 데이터와 controller를 이미 갖추고 있어 비교 조건을 명확하게 만들기 좋다. 기존 최종 테스트 결과를 이미 확인했으므로 해당 테스트는 새 확장의 개발 참고 자료로 취급하고, 새 최종 평가 조건을 남겨둔다.

| 선택 주제 | 연구 질문 | 추가로 필요한 작업 |
| --- | --- | --- |
| Domain randomization | 다양한 학습 날씨와 camera 설정이 같은 데이터 예산에서 held-out 조건의 성능을 개선하는가 | 물리적으로 일관된 수집, 동등한 데이터와 학습 예산, 새로운 테스트 조건 |
| Temporal 모델 | 과거 3~5 frame이 끊긴 차선이나 부분 가림에서 주행을 개선하는가 | 시간 순서와 결측 처리, temporal split, 추가 latency 측정 |
| B1 segmentation | 차선 mask를 거치는 구조가 waypoint 직접 회귀와 어떤 실패 차이를 보이는가 | Mask 라벨, segmentation 학습, geometry 변환과 공통 controller |

| 일차 | 선택한 한 주제의 작업 | 그날 남길 결과 |
| --- | --- | --- |
| D57 | 핵심 결과와 남은 실패를 검토 | 확장 대상 실패 목록 |
| D58 | 연구 질문과 비교 가설 확정 | 한 문장 연구 질문과 검증 계획 |
| D59 | 추가 요소를 제외한 대조군 정의 | 통제할 데이터와 모델 설정 |
| D60 | 새로운 dev 및 final 조건 분리 | 확장용 평가 manifest |
| D61 | 추가 수집 또는 라벨 schema 설계 | 데이터 계약 |
| D62 | 필요한 구현의 작은 실행 확인 | 최소 실행 결과 |
| D63 | 9주 차 검토와 보완 | 범위와 의존성 확인 |
| D64 | 추가 데이터 또는 sequence 준비 | 확장 dataset 초안 |
| D65 | 분할과 라벨 시각화 검증 | 확장 dataset snapshot |
| D66 | 대조군의 학습 실행 | 대조군 checkpoint |
| D67 | 확장 방법의 학습 실행 | 확장 checkpoint |
| D68 | Dev 오차와 예측 분석 | 초기 개발 비교 |
| D69 | 입력, 출력과 계산 비용 확인 | Latency와 오류 기록 |
| D70 | 10주 차 검토와 보완 | 비교 실험 동작 확인 |
| D71 | 동일 데이터 및 update 예산으로 개발 비교 | 공정한 비교표 |
| D72 | 추가 요소 하나의 ablation 실행 | 요소별 효과 기록 |
| D73 | 가능하면 추가 학습 seed 실행 | Seed별 결과 |
| D74 | 개발 실패 사례 분석 | 개선 및 악화 조건 |
| D75 | Dev 근거로 최종 후보 선택 | 선택 근거 |
| D76 | 새 평가 프로토콜과 실행 규모 확인 | 최종 실험 manifest |
| D77 | 확장 모델과 설정 동결 | 확장 freeze manifest |
| D78 | 새로운 clean 평가 실행 | Clean 결과 |
| D79 | 새로운 환경 변화 평가 실행 | Shift 결과 |
| D80 | 남은 paired 반복과 기술적 누락 보완 | 전체 평가 기록 |
| D81 | 평균, 변동, 성능 저하와 latency 집계 | 확장 결과표 |
| D82 | 효과가 있었던 조건과 한계 작성 | 연구 결과 해석 |
| D83 | 재현 명령, 모델과 문서 점검 | 확장 재현 확인 |
| D84 | 선택 연구 마무리와 버전 후보 준비 | 연구 확장 결과와 공개 후보 |

## 작업 파일과 진행 기록

현재 저장소에서 향후 구현할 파일의 권장 구조다. 디렉터리와 실행 명령은 각 단계에서 실제 코드에 맞춰 확정한다.

```text
Vision-Lane-Following/
  configs/             수집 학습 제어 평가 설정
  carla_tools/         수집 route 라벨 센서
  src/
    datasets/
    models/
    controllers/
    training/
    evaluation/
  scripts/             수집 학습 평가 실행 진입점
  tests/               좌표 변환 동기화 split metric의 핵심 검증
  docs/
    ROADMAP.md
    data_contract.md
    evaluation_protocol.md
    progress.md
  results/             episode CSV와 요약 결과
  assets/              작은 데모 이미지와 영상 참조
  README.md
```

큰 데이터, CARLA 설치 파일과 checkpoint 원본은 Git에 넣지 않고 저장 위치, dataset 버전 및 checksum을 기록한다. Notebook은 분석과 시각화에 사용하고 주요 수집, 학습과 평가는 script로 실행한다.

진행 기록에는 계획 일차, 실제 날짜, 완료 작업, 산출물 경로, gate 결과 및 다음 작업을 남긴다. 하루를 넘겼다고 완료 표시하지 않고 산출물과 검증을 기준으로 표시한다. 매주 마지막 날은 밀린 작업과 실패 분석을 처리하고 다음 단계의 의존성을 점검한다.

## 최종 확인 목록

- [ ] 카메라 입력 범위와 속도 피드백 사용이 문서에 명시되어 있다.
- [ ] 수집 영상과 해당 frame의 자세 및 라벨 정합을 검증했다.
- [ ] Recovery 라벨과 lateral error가 원래 목표 차선을 기준으로 한다.
- [ ] 정답 waypoint로 controller의 직선과 곡선 주행을 검증했다.
- [ ] Train, dev와 final test가 route 및 scene group 단위로 분리되어 있다.
- [ ] B0와 M1이 같은 controller와 평가 조건을 사용한다.
- [ ] 동일 데이터 예산의 central 및 recovery 비교 결과가 있다.
- [ ] 최종 모델과 설정을 테스트 결과 열람 전에 고정했다.
- [ ] 실패 episode를 포함한 결과와 실제 반복 및 seed 수를 보고한다.
- [ ] Latency를 별도로 측정하고 simulator 동기 주행과 실시간성을 구분한다.
- [ ] 결과표, 실패 사례, 데모와 재현 안내가 있다.
- [ ] 사용한 데이터, 모델, config와 코드 버전을 추적할 수 있다.

## 공식 자료

일정과 규모는 이 프로젝트를 위한 계획 선택이다. 아래 자료는 simulator의 기능과 데이터 형식 및 순차 학습의 배경을 확인하는 근거다.

- [CARLA 0.9.16 릴리스](https://github.com/carla-simulator/carla/releases/tag/0.9.16): Windows 패키지와 camera sensor의 frame, timestamp, transform 정합 관련 수정.
- [CARLA 동기화와 timestep](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/): 동기 모드, fixed timestep 및 카메라 처리 지연.
- [CARLA Maps와 Waypoint](https://carla.readthedocs.io/en/0.9.16/core_map/): 차선 중심 waypoint, route 연결과 분기.
- [CARLA 센서](https://carla.readthedocs.io/en/0.9.16/ref_sensors/): RGB, collision, lane invasion과 센서 좌표계.
- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks): 실행 환경과 세션 제한은 착수 시 다시 확인한다.
- [TuSimple 데이터 형식](https://github.com/TuSimple/tusimple-benchmark/blob/master/doc/lane_detection/readme.md): 영상 좌표의 차선 라벨.
- [CULane 데이터셋](https://xingangpan.github.io/projects/CULane.html): 차선 검출 데이터와 annotation.
- [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410): 카메라 영상만으로 결정되지 않는 목표 경로와 주행 명령의 역할.
- [DAgger 원 논문](https://proceedings.mlr.press/v15/ross11a.html): 학습 데이터와 모델이 실제로 방문하는 상태 사이의 분포 차이.
