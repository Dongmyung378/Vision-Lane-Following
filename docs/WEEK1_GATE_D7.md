# D7 Week 1 RGB·pose·waypoint 정합 게이트

작성일: 2026-10-08 (Asia/Seoul). Gate version은 `vlf-week1-alignment-gate-d7-v1`이다. 구현은 [week1.py](../carla_tools/week1.py), 실행·재검증 entrypoint는 [run_week1_pipeline.py](../scripts/run_week1_pipeline.py), [verify_week1.py](../scripts/verify_week1.py)다. **D7 완료: 기존 Kaggle notebook version 4의 native 게이트와 전체 다운로드 후 최종 로컬 CPU 게이트가 모두 PASS했다.** 실제 수집은 3 episode, 총 180 frame이며, 최종 negative QA 9개도 기대한 실패를 검출했다.

## 통과 기준과 기술 범위

CARLA 0.9.16 Town01에서 `straight`, `left_curve`, `right_curve`의 정지 episode를 각각 60 frame 수집했다. Vehicle은 throttle/steer 0, brake 1, hand brake를 적용하고, 같은 수집 config와 nominal camera calibration을 사용한다. Spawn index만 각 고정 route에 맞게 바꾼다. 3개의 다른 scene에서 얻은 정지 영상은 학습 pilot이나 주행 성능 평가의 표본 수로 사용하지 않는다.

Case 이름만 보고 통과시키지 않는다. Actual route의 waypoint yaw와 XYZ chord heading을 함께 확인한다. 현재 geometry policy는 15m 안의 straight 총 heading change ≤3°, curve의 `abs(net yaw change) ≥20°` 및 `abs(net chord heading change) ≥15°`, yaw 방향 일관성 ≥0.9다. Curve의 yaw와 chord net change 부호도 같아야 한다. CARLA positive yaw는 우회전으로 분류한다. 기준과 실제 route 증거는 `geometry_routes.json`에 남긴다. Curve arc는 [D6 문서](WAYPOINT_LABELS_D6.md)의 dense XYZ polyline 근사이며 analytic arc가 아니다.

| 검사 | D7 요구 조건 |
| --- | --- |
| Raw episode | `COMPLETE`, 디스크 PNG·JSONL·metadata 무결성과 capture cleanup PASS |
| Frame/time | RGB sensor/world/snapshot frame이 동일하고 timestamp가 맞으며 연속 frame이 빠지지 않음 |
| Label | 독립 `validate_labels` PASS, 원본 hash 연결, raw는 미변경, 고정 route의 finite `5 × 2` actor-origin m 좌표 |
| Projection | 같은 frame camera transform과 nominal pinhole calibration 사용, inverse projection roundtrip 오차 ≤0.001m |
| Visibility | **매 frame 다섯 점 모두 camera 전방**, 최소 세 점이 영상 안에 존재 |
| Color | 저장 PNG를 RGB8로 decode하여 같은 frame BGRA witness와 RGB hash 확인; R/B를 구분하는 pixel 존재 |
| Case 간 정합 | 서로 다른 episode ID/path, 하나의 dataset version·town/OpenDRIVE source·calibration·spawn 제외 config |

영상 밖의 점은 `OUTSIDE_IMAGE`로 원래 UV를 기록한다. 특히 mount/FOV에 따라 3m 점이 밖에 있는 것 자체는 실패가 아니다. 다섯 점의 전방 조건과 최소 세 점의 가시 조건을 만족해야 한다. 좌표 clamp, near-plane/behind-camera sign flip으로 이 조건을 맞추지 않는다.

`verify_week1_cases`는 세 case의 모든 frame을 검사하며, overlay 한 장의 PASS만으로 전체 게이트를 통과시키지 않는다. Gate PASS의 의미는 정지 상태의 RGB·pose·label·projection·색상 정합이다. Controller 동작, 학습 정확도, lane following, recovery 또는 final benchmark 성능을 인증하지 않는다.

## 새 capture의 BGRA → RGB 증거

[color.py](../carla_tools/color.py)의 `color_provenance`는 같은 sensor callback의 BGRA8 buffer SHA-256, 변환 RGB8 SHA-256와 최대 4개 source pixel의 BGRA 값을 기록한다. 첫/중간/끝 pixel과 R/B 값 차이가 가장 큰 pixel을 선택하며 중복 index는 합친다. D7은 저장 PNG의 decoded RGB hash와 해당 pixel의 `[R,G,B]=[BGRA[2],BGRA[1],BGRA[0]]`를 확인하고, B≠R witness를 요구한다.

Raw BGRA 전체 buffer는 보존하지 않으므로 source buffer hash를 offline에서 다시 계산하는 증거는 아니다. 저장 RGB의 hash와 source pixel witness를 함께 확인하여 converter와 같은 R/B swap을 단순히 RGB hash 재계산으로 통과시키지 않게 한다. Witness는 **새 capture 때만 기록**한다. 이를 갖지 않은 D4/D5 legacy raw는 D6 기하 라벨/overlay 확인에는 쓸 수 있어도 D7 색상 게이트는 통과하지 못한다. 기존 파일에 witness를 사후 추가하지 않는다.

## 기존 Kaggle notebook 재사용

[AGENTS.md](../AGENTS.md)에 따라 CARLA notebook ID `dongmyungpark/vlf-carla-rgb-smoke-20261005`와 기존 제목 `VLF CARLA RGB smoke 20261005`를 유지한다. 새 날짜·일차·실험 suffix, Copy/Fork 또는 새 notebook으로 대체하지 않는다. GPU/Vulkan 및 download 진단도 각각 지정된 기존 ID를 재사용한다.

원격 제출 전에 대상 notebook의 존재·접근과 현재 metadata를 **읽기 전용으로 확인**한다. 접근 실패 시 기존 연결을 복구하며 자동 신규 생성으로 우회하지 않는다. [prepare_kaggle_week1.py](../scripts/prepare_kaggle_week1.py)는 이미 얻은 metadata의 ID/title을 확인해 public project source와 SHA manifest를 묶고 GPU/T4 workload를 준비한다. 이 script 자체가 원격 접근 확인이나 제출을 수행하지 않으며, `submitted=false`, `creates_new_notebook=false`를 기록한다. 인증정보는 bundle·코드·문서·Git에 저장하지 않는다.

```powershell
.\.venv-kaggle\Scripts\kaggle.exe kernels status dongmyungpark/vlf-carla-rgb-smoke-20261005
.\.venv-kaggle\Scripts\kaggle.exe kernels pull dongmyungpark/vlf-carla-rgb-smoke-20261005 -p results/local/week1-existing-notebook -m
.\.venv\Scripts\python.exe scripts\prepare_kaggle_week1.py --existing-metadata results/local/week1-existing-notebook/kernel-metadata.json --output-dir results/local/week1-submit-bundle
.\.venv-kaggle\Scripts\kaggle.exe kernels push -p results/local/week1-submit-bundle --accelerator NvidiaTeslaT4 -t 1500
.\.venv-kaggle\Scripts\kaggle.exe kernels status dongmyungpark/vlf-carla-rgb-smoke-20261005
.\.venv-kaggle\Scripts\kaggle.exe kernels output dongmyungpark/vlf-carla-rgb-smoke-20261005 -p results/local/week1-download
```

경로는 작업 예시이며 기존 notebook에서 읽은 metadata와 아직 없는 bundle directory로 바꾼다. Native CARLA rendering은 GPU가 있는 기존 Kaggle notebook에서 실행한다. CPU 재검증만 로컬에서 수행한다.

## Native pipeline과 산출물

[kaggle_carla_bootstrap.py](../scripts/kaggle_carla_bootstrap.py)의 `WORKLOAD=week1`은 unpacked server package의 startup map을 Town01로 맞추고 CARLA를 실행한다. 실행 중 다른 map으로 reload하여 수집하지 않는다. Bootstrap은 자신의 server process를 종료하며, pipeline은 이미 실행 중인 server를 사용한다. `run_week1_pipeline.py`만 실행한다고 server가 새로 시작되지 않는다.

Pipeline은 read-only geometry survey → live route 재확인과 RGB/pose/color capture → 별도 라벨/overlay 생성 → 독립 D7 게이트 → 복사본 기반 negative QA → capture 이후 asynchronous RPC 확인을 수행한다. 어느 단계든 실패하면 해당 stage와 FAIL을 보고한다. Version 4가 생성한 artifact 구조는 다음과 같다.

```text
/kaggle/working/
  kaggle_carla_bootstrap.json
  week1/
    geometry_routes.json
    raw/
      geometry_collection_report.json
      geometry_straight_01/episode.json, frames.jsonl, images/
      geometry_left_curve_01/episode.json, frames.jsonl, images/
      geometry_right_curve_01/episode.json, frames.jsonl, images/
      geometry_*_01_capture_report.json
    labels/straight.json, left_curve.json, right_curve.json
    overlays/<case>/overlay.png, report.json
    week1_manifest.json
    week1_report.json
```

Version 4 bootstrap은 위 pipeline의 PASS와 `server_stopped=true`를 기록했다. 전체 PNG까지 다운로드한 뒤 최종 코드의 CPU 게이트와 negative QA도 통과했다. Raw의 `NOT_GENERATED`/`null`과 derived sidecar의 `GENERATED`를 유지한다. 신규 native 수집의 dataset version은 `d6-geometry-v1`이며 기존 D5 train/dev manifest와 섞어 학습 split을 만들지 않는다.

Negative QA는 wrong ego frame, units, ego target, duplicate/missing label frame, source hash, PNG corruption, dropped raw frame, RGB hash를 다시 계산한 R/B swap이 실패하는지 확인한다. 복사본·임시 sidecar만 바꾸고 원본을 유지한다. Pipeline report와 bootstrap report의 PASS, bootstrap의 server 종료 상태 및 실제 artifact 다운로드를 함께 확인한다. Notebook 실행 종료만으로 통과 처리하지 않는다.

## 다운로드 후 CPU 재검증

Kaggle의 `week1/` 전체를 PNG까지 다운로드하고 상대 directory 구조를 보존한다. Version 4의 다운로드 위치는 `results/local/kaggle-week1-output-v4/week1`이다. `week1_manifest.json`의 episode/labels 경로는 manifest 위치를 기준으로 해석하므로 원격 절대 경로로 고치지 않는다.

```powershell
$week1Artifact = 'results/local/kaggle-week1-output-v4/week1'
.\.venv\Scripts\python.exe scripts\verify_week1.py --manifest "$week1Artifact/week1_manifest.json" --output results/local/week1-cpu-recheck.json
```

이 명령은 CARLA/GPU 없이 디스크에서 raw와 라벨을 새로 읽고 color witness·projection을 다시 검사한다. 출력 보고서는 원본 밖의 경로에 저장한다. 정상 종료 코드 0과 fresh report의 `status=PASS`를 모두 확인한다. 실패한 재검증에 과거 remote PASS를 대신 사용하지 않는다.

개별 case를 점검하려면 [D6 CPU 명령](WAYPOINT_LABELS_D6.md#cpu에서-생성과-재검증)을 사용한다. Unit fixture 검증은 다음처럼 실행한다. 이 test 결과는 native 증거와 별도로 기록한다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_color.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_week1.py -v
```

## 실제 결과와 다음 단계

**실제 D7 native 결과: PASS.** 기존 notebook `dongmyungpark/vlf-carla-rgb-smoke-20261005`의 version 4는 `COMPLETE`로 종료했고, 다운로드한 bootstrap은 `status=PASS`, `week1_verified=true`, `server_stopped=true`를 기록했다. CARLA rendering은 Kaggle Tesla T4에서 수행했다. 로컬 GPU나 CARLA server는 사용하지 않았으며, 로컬 작업은 저장 artifact의 CPU 검증이다. 원격 제출 code SHA-256는 `a02117c5c8beea7f9c849b9beb04bd97bfac0a486be763e7645476c9761afd5d`다.

| Case | Spawn / 실제 frame | 수집 수 | Route net yaw / chord heading | 최소 visible 점 | 색상 검증 frame / pixel witness | Camera↔world 최대 roundtrip 오차 |
| --- | --- | --- | --- | --- | --- | --- |
| `straight` | 0 / 1142–1201 | 60 | 0.000° / −0.000004° | 4 | 60 / 240 | 7.281 × 10⁻⁷ m |
| `left_curve` | 108 / 1323–1382 | 60 | −27.189° / −26.664° | 4 | 60 / 240 | 7.156 × 10⁻⁷ m |
| `right_curve` | 174 / 1478–1537 | 60 | +30.994° / +29.884° | 4 | 60 / 240 | 7.194 × 10⁻⁷ m |

총 180 frame에 대해 raw 무결성, RGB/world/snapshot frame·시간, label/route/pose, 색상 witness, finite projection, 좌표 미클램핑 및 capture cleanup을 검사했다. 모든 frame에서 3m 타깃은 `OUTSIDE_IMAGE`, 6/9/12/15m 타깃은 `VISIBLE`이었다. 다섯 점은 모두 camera 전방이므로 이 결과는 visibility 기준을 만족한다. OpenDRIVE source hash는 `97a7f6ac67812567e5c8ee0599cd823b23f80f30f3f97c502212e38b72e2b709`이며 세 case의 calibration과 spawn 제외 config hash가 같았다.

원격·로컬 결과, 코드와 source/label/config hash는 [실행 증거 요약](evidence/carla_week1_d6_d7_2026-10-08_summary.json)에 기록한다. 실제 native overlay는 [직선](evidence/carla_week1_d6_d7_2026-10-08_straight_overlay.png), [좌곡선](evidence/carla_week1_d6_d7_2026-10-08_left_curve_overlay.png), [우곡선](evidence/carla_week1_d6_d7_2026-10-08_right_curve_overlay.png)에서 확인한다. Overlay renderer 자체는 색상 인증을 하지 않으며, 위 색상 결과는 전체 frame을 검사한 별도 D7 게이트의 결과다.

**최종 로컬 CPU 게이트: PASS.** 실제 PNG 180개까지 다운로드한 뒤 최종 코드로 세 case 전체를 다시 검증했고 `errors=[]`, 3 episode, 180 frame을 확인했다. 원격 version 4와 로컬 audit의 source identity·hash 및 label SHA-256도 모두 일치했다. 결과 보고서는 `results/local/week1-v4-cpu-gate.json`에 있으며 Git에는 [증거 요약](evidence/carla_week1_d6_d7_2026-10-08_summary.json)을 저장했다.

**최종 negative QA: 9개 PASS.** 다운로드한 좌곡선 episode의 복사본에서 잘못된 ego frame·단위·ego 타깃, 중복/누락 label frame, 잘못된 source hash, PNG 손상, raw frame 누락, RGB hash를 다시 계산한 R/B swap을 각각 주입했다. 9개 모두 기대한 `FAIL`을 반환했으며 원본은 수정하지 않았다. 결과는 `results/local/week1-v4-cpu-negative.json`과 위 증거 요약에 있다. 정상 데이터의 게이트 PASS와 손상 데이터의 거부 결과를 각각 확인했다.

외부 `jsonschema 4.26.0` 검사도 raw frame 180개와 derived label 180개에 대해 오류 0건으로 PASS했다. 이 검사는 schema 적합성을 확인하며 geometry/color 검사를 대신하지 않는다. 최종 전체 unit suite는 309개 PASS, skip 0개이며 native artifact 검증과 별개의 결과다. Capture 뒤 RPC 확인도 version 0.9.16/Town01에서 asynchronous world의 세 후속 snapshot을 관찰해 PASS했고, bootstrap은 자신의 server 종료를 확인했다.

D8의 waypoint controller 구현·검증이 다음 단계다. D13에는 실제 moving training pilot과 고정 목표 lane의 recovery 데이터를 수집하고, D14에는 [평가 명세 초안](EVALUATION_SPEC_D5.md)에 따른 주행 metric·종료 규칙을 구현한다. 정지 gate에 route completion이나 inference latency 성능을 부여하지 않는다. Town02 final test는 `UNCOLLECTED`/`UNVERIFIED`로 남아 있으며 D42 freeze 전 개발 선택에 사용하지 않는다.

## Weather readback 실패와 복구

Version 3에서는 직선·좌곡선 각 60 frame 이후 우곡선 setup이 `weather-readback-mismatch`로 실패했다. [실패 증거](evidence/carla_week1_d6_d7_2026-10-08_weather_failure.json)를 보존했으며 어떤 weather 필드가 달랐는지는 당시 보고서에 없어 특정하지 않았다.

CARLA 0.9.16 공식 [Client.cpp](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/client/detail/Client.cpp#L239-L245)는 weather setter에 `AsyncCall`, getter에 `CallAndWait`를 사용한다. Setter 반환은 적용 완료 확인이 아니다. 공식 [CarlaEngine.cpp](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Game/CarlaEngine.cpp#L276-L281)는 synchronous tick을 기다리는 중에도 RPC를 처리한다. 이 소스에서 적용 전 즉시 readback의 가능성을 추론했고, setup과 원래 weather 복원에 bounded readback polling을 적용했다. Version 4의 PASS만으로 실패 당시의 특정 필드나 단일 원인을 확정하지 않는다.

[session.py](../carla_tools/session.py)는 setter를 한 번 호출한 뒤 요구 field set과 기존 값 검사를 유지하며, mismatch이면 monotonic 기준 최대 `min(2초, connection.timeout_seconds)` grace에서 20ms 이하 간격으로 재조회한다. Weather 확인을 위한 world tick은 추가하지 않는다. **각 synchronous RPC에는 별도의 configured client timeout이 적용되므로 전체 wall time이 항상 2초 이하라는 뜻은 아니다.** 시간 내 불일치가 지속되면 여전히 실패하고, 이후 `verify_applied()`도 엄격한 readback을 유지한다. 최종 로컬 코드는 이미 일치하는 경우 clock/sleep 없이 반환하는 fast path를 추가했다. 이는 즉시 성공·지연 적용·영구 불일치·복원 실패를 검사하는 [7개 CPU 회귀 test](../tests/test_weather_readback.py)로 확인했다.
