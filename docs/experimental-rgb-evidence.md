# Optional RGB hypothesis memory and signal confirmation

`HypothesisTrafficBridge` is an explicitly selected simulation experiment for a
fixed forward camera, level road, 20 Hz observations and a maximum 3 m/s control
envelope. It consumes camera detections, lane predictions, nominal calibration
and ego speed. Simulator actor positions, semantic images and map annotations
are excluded from its decision inputs.

The existing evidence bridge remains the default. Select this experiment only
when its narrower assumptions and known failure modes are acceptable:

```python
from carla_tools.traffic_experimental import make_evidence_bridge
from carla_tools.traffic_policy import TrafficPolicy

bridge = make_evidence_bridge(
    nominal_calibration=traffic_perception.calibration,
    front_bumper_x_m=traffic_perception.config.front_bumper_x_m,
    ego_half_width_m=traffic_perception.config.ego_half_width_m,
    strategy="hypothesis",
)
policy = TrafficPolicy(path_predictor, traffic_perception, lane_controller,
                       bridge=bridge)
```

The injected bridge owns compatibility with the perception and supervisor
configuration. Keep the standard 50 ms update period, 2 m/s² comfortable braking,
2 m standstill gap and 0.2 s reaction budget. The experiment does not modify the
model, current detection boxes, current track identifiers or lane controller.

## Vehicle identity memory

An old vehicle identity can be retired only when a currently observed alternative
has repeated simultaneous cross-label RGB support: closely overlapping boxes,
matching footpoints and consistent ground range. At least three observations
over 0.1 s must support the pair, the surviving identity must be fresh, and both
range bounds must exceed the stopping budget. Close, unknown-range, unknown-speed
and urgent approaching old hazards retain their original conservative memory.
Two distinct vehicles or a person and a vehicle cannot provide this certificate.

## Tentative far-person existence

Current person detections always retain the original emergency behavior. High
confidence, repeated, initially near, uncertain-ground or supervisor-urgent
person observations cannot use tentative retirement. A weak, never-confirmed,
initially far hypothesis may lose existence belief after four consecutive valid
RGB observations show an unobstructed grey road contact patch while ego speed is
at most 0.2 m/s. The image anchor must remain stable; a detected low-speed camera
jump blocks retirement for the rest of the episode. Any overlapping current
object detection, invalid perception or changed nominal calibration also blocks
the evidence.

This is an existence hypothesis update, not proof that a person has physically
cleared the road. A real far person seen briefly can move away from its old
contact patch and still be missed by the detector. Grey clothing, undetected
occluders, slopes, motion and errors in monocular geometry remain limitations.
The synthetic tests preserve this counterexample explicitly. Use a complete
reachable-region observer or additional sensing when this tradeoff is unacceptable.

## Signal confirmation

Detector confidence and colour confidence remain separate. A vertical signal
head with readable red or yellow requests an immediate stop. Confirmed stop
memory requires either repeated colour evidence, a head already persistent in
earlier observations, or a strong detector observation supported independently
by pixels showing a compact lamp in a dark vertical housing. The latter checks
the expected lamp position within the head. An isolated weak or poorly shaped
box requests a conservative stop while visible and during the short absence
hold, without manufacturing a permanent confirmed-red latch.

Confirmed stop memory has no timeout release. Only sufficiently persistent green
associated with every confirmed head can release it; green at another head
cannot. An unreadable current head remains uncertain and requests a stop. Lane
tangent projection reports only camera bearing: it does not determine which
approach a signal legally controls. Brief weak true signals, horizontal heads,
colour errors, occlusion and head-association errors remain unresolved limits.

## Validation

Use synthetic counterexamples for close hazards, missing detections, different
vehicle footpoints, urgent person occlusion, a shifted camera, unrelated green,
single-frame strong true red and weak false colour. Compare complete adapted
observations, diagnostics and control commands with a reference implementation
on fixed recorded RGB inputs. Such replay verifies implementation consistency;
new closed-loop driving trials are still required to assess route completion,
collisions, lane invasion, unnecessary stops and restart delay.
