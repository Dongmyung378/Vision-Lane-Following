# Exclusive secondary person evidence

`ExclusiveSecondaryPersonConfirmationScheduler` is an optional extension of the
secondary person confirmation scheduler. Construct the same explicit backend,
calibration and primary identity described in `secondary-person-memory.md`, then
select the ownership scheduler when building the injected bridge:

```python
from carla_tools.traffic_secondary_ownership import (
    ExclusiveSecondaryPersonConfirmationScheduler,
)

scheduler = ExclusiveSecondaryPersonConfirmationScheduler(
    secondary_backend,
    nominal_calibration,
    front_bumper_x_m=front_bumper_x_m,
    primary_source_identity=primary_detector.identity,
)
```

Pass this scheduler to `SecondaryHypothesisTrafficBridge`, then use that bridge
explicitly in `TrafficPolicy`. Existing factories and defaults do not change.
Importing this module loads no model or device runtime. The backend still needs
an explicit device and verified weight file before inference.

For each secondary person box with score at least 0.5, all current primary
persons with score at least 0.5 compete by box IoU. Strong, near and out-of-path
primaries remain competitors even when they cannot receive an additional veto.
The unique greatest IoU at or above 0.3 owns the box. Equal best overlaps within
1e-6 are ambiguous. A box is never reassigned to a weaker primary. Several
secondary boxes may support one primary; a secondary box supports at most one
primary. Nonperson secondary detections remain outside policy inputs.

The original scheduler still bounds calls and selects which weak, far primary
tracks may receive additional memory protection. Ownership only changes new
positive evidence assignment. It never removes a primary, grants retirement,
refreshes last observation, changes TTL, or undoes an earlier positive veto.
Current same-ID positive lateral clearance keeps its original hold requirement.

Receipts retain competing primary IDs, scores, overlaps, ownership and withheld
confirmations. Primary boxes remain in the original same-frame perception packet.
Receipts identify both the installed base module and this ownership
module by source SHA-256. Preserve exact producer sources for replay. Source
hashes establish provenance for a trusted producer, not authentication.

Two real people can overlap and share one secondary box. One person may lose
additional confirmation even though the primary observation remains. This
adapter does not establish physical identity, safe absence, collision avoidance,
completion or public-road readiness. A separate scenario evaluation is required.
