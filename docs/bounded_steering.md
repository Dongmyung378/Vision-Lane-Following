# Bounded steering feedback

`carla_tools.bounded_steering.BoundedSteeringFeedback` combines a nominal
steering request with an auxiliary request. Both inputs use normalized
steering units and must be taken **before** smoothing and slew limiting.
The caller retains the nominal controller's throttle and brake policy.

The order of operations is:

1. Compute `gain * (proposal - nominal)` when the auxiliary estimate is usable.
2. Clip that correction to `[-correction_limit, correction_limit]`.
3. Add the correction to the nominal request and enforce the steering limit.
4. Smooth once using the previous actual output, then apply the slew limit.

```python
from carla_tools.bounded_steering import BoundedSteeringFeedback

feedback = BoundedSteeringFeedback()
feedback.reset()  # Reset at the beginning of an independent episode.

adjusted = feedback.step(
    nominal_request=0.12,
    proposal_request=0.18,
    enabled=True,
    dt_seconds=0.05,
)
steer = adjusted.steer
```

The defaults are gain `0.35`, correction limit `0.004`, normalized steering
limit `1.0`, smoothing alpha `1.0`, and slew `2.0` units per second. They are
configurable through `BoundedSteeringConfig`.

Set `enabled=False` when the auxiliary estimate is unavailable. In that case
the proposal may be absent or nonfinite, and its correction is zero. The
actual output history remains continuous. Invalid nominal requests,
timesteps, or enabled proposals raise `ValueError` before changing state.

The returned `SteeringCorrection` distinguishes the requested correction,
bounded correction, filtered request, and actual output. The correction bound
describes steering requests; lane clearance is measured separately in driving
evaluation using the vehicle footprint throughout driving and stopping.
