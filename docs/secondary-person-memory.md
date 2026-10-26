# Optional secondary person confirmation

This experimental simulation adapter adds a conservative memory veto to the
RGB hypothesis strategy. It does not select a default policy or establish
collision avoidance, completion, reliable physical range, or public-road safety.
Two detectors can share the same false positive and cause a persistent stop.

## Construction

Use the existing Python environment and compatible pinned detector dependencies:
PyTorch 2.7.1 and TorchVision 0.22.1. Importing these optional modules loads no
model or GPU runtime. The auxiliary model is the official TorchVision
Faster R-CNN ResNet-50 FPN v2 COCO_V1 model. Obtain its weights separately;
the backend never downloads weights or silently falls back to another model.

```python
from carla_tools.traffic_policy import TrafficPolicy
from carla_tools.traffic_secondary_confirmation import (
    ResNet50V2Backend, SecondaryPersonConfirmationScheduler, WEIGHTS_SHA256,
)
from carla_tools.traffic_secondary_veto import SecondaryHypothesisTrafficBridge

secondary = ResNet50V2Backend(
    weights_path,
    expected_sha256=WEIGHTS_SHA256,
    device="cpu",  # Choose the execution device explicitly.
)
scheduler = SecondaryPersonConfirmationScheduler(
    secondary, nominal_calibration,
    front_bumper_x_m=front_bumper_x_m,
    primary_source_identity=primary_detector.identity,
)
bridge = SecondaryHypothesisTrafficBridge(
    nominal_calibration=nominal_calibration,
    front_bumper_x_m=front_bumper_x_m,
    ego_half_width_m=ego_half_width_m,
    secondary_scheduler=scheduler,
)
policy = TrafficPolicy(path_predictor, traffic_perception, lane_controller,
    supervisor=supervisor, bridge=bridge)
```

The primary detector must retain the existing MobileNet Faster R-CNN architecture,
COCO label mapping and exact weight identity accepted by the scheduler. Nominal
calibration, vehicle geometry, perception settings and supervisor must share the
existing hypothesis strategy's fixed-camera, 20 Hz and 3 m/s contracts. Stable
primary track IDs must not be reused until reset. Call the policy once per
consecutive 0.05-second observation, including frames with no detections. The
surrounding application must stop on invalid input or policy exceptions.

## Scheduling and decisions

Only primary person hypotheses with confidence `0.5 <= score < 0.85`, in the
nominal path corridor, with fewer than three person sightings, and a finite
fixed-mount footpoint near bound beyond 8 m are eligible. This interval is a
scheduling heuristic; a partially occluded person's bounding-box bottom may not
be a ground contact. It is not a certified distance bound.

There are at most two auxiliary calls per track and one shared call per eligible
frame. An earlier positive agreement stops repeated calls for that track.
Sightings count even when the person was outside the path or already strong.
The secondary person must have confidence at least 0.5 and box IoU at least 0.3
with the current primary person. All other secondary objects, vehicles and
signals stay in the audit receipt and never enter the control observations.

A positive receipt protects that existing weak hypothesis from both old-image
road-patch retirement and weak-memory time-to-live deletion. It does not refresh
the last observation, create a track, delete a track, or authorize clearance.
The parent's current same-ID valid lateral-clearance evidence still releases
memory after 0.2 seconds. A negative result or missing confirmation preserves
parent behavior from reset; it does not undo an earlier positive veto. Reset
clears both scheduler and veto state.

With `secondary_scheduler=None`, this bridge equals the existing experimental
hypothesis strategy. That strategy is distinct from the default evidence bridge.

## Receipts and CPU replay

Packets bind the original RGB SHA-256, frame, timestamp, primary box, score and
observation count to matched secondary person boxes and recomputed IoU. They
also include the full secondary detections and timing for audit. GT inputs are
forbidden. The official auxiliary weights have SHA-256
`dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`.
`implementation_source_sha256()` identifies the installed confirmation module's
exact bytes; editing or formatting that module changes its receipt identity.
This is a provenance check for a trusted injected producer, not authentication
of an untrusted process or an independent safety proof.

`RecordedSecondaryBackend(receipts, identity, timestamp_by_frame=...)` supports
CPU replay without constructing a model. Receipts are keyed by frame and must
match RGB, time and weights. Full native receipts must also match the declared
implementation identity and RGB-only contract. Legacy raw box caches are
wrapped with their original receipt digest and the explicitly supplied replay
implementation identity; this transformation is metadata, not new inference.
Missing evidence raises `NOOBS`; it is never interpreted as absence of a person.

For a deliberately pinned historical producer, the veto bridge accepts an
explicit `secondary_backend_source_sha256` override. Preserve the historical
receipt and source; do not relabel its implementation as the installed module.
Record model calls separately from cached replays when reporting evaluations.
