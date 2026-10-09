# Moving RGB waypoint pilot

The pilot uses native CARLA RGB frames from a moving privileged reference
controller. Labels describe the original pinned lane in the same frame's actor
coordinates: five points at future centreline arcs 3, 6, 9, 12 and 15 metres,
with x forward and y right. The prediction model receives RGB only. Route,
pose, calibration, weather, speed and perturbation metadata support data
validation and analysis.

`configs/pilot.json` defines a 2,000–5,000 sample budget. Native Town01 spawn
points are surveyed before any collection. Straight, left curve and right
curve routes appear in both train and dev. Whole 35 metre route corridors and
measured actor positions must stay more than 75 metres apart across splits.
Central and recovery episodes from a nearby scene remain in the same split.
Town02 is reserved and is neither loaded nor inspected by this pipeline.

Each native episode records every RGB frame, same-frame snapshot, original
route label, acknowledged controller command, event sensor callbacks,
controlled stop and owned-actor cleanup. Dataset samples exclude startup and
stationary frames using measured speed and consecutive position displacement.
Repeated RGB hashes are excluded within a split and rejected across splits.
No repeated stationary images or deterministic reruns pad the sample count.

`pilot_manifest.json` freezes relative PNG paths, SHA256 hashes, labels,
episode/route/scene/split identities and source metadata/trace hashes. The
CPU gate independently replays every original route label and native
controller command, verifies PNG content and colour provenance, checks
termination and cleanup, rebuilds the moving sample selection, and validates
all cross-split spatial boundaries. Missing conditions, changed labels,
duplicate samples and overwritten source bytes fail validation.

With an already running CARLA server:

```powershell
.venv\Scripts\python.exe scripts/run_pilot_pipeline.py --output-dir results/local/pilot-example
.venv\Scripts\python.exe scripts/verify_pilot.py --manifest results/local/pilot-example/pilot_manifest.json --output results/local/pilot-example-gate.json
```

The collector refuses existing output directories. Its artifact archive also
contains a fresh development comparison of the RGB baseline and privileged
reference. Those comparison episodes are excluded from training samples.
Reference driving failures block the collection gate; valid baseline detection
or driving failures remain measured outcomes. Collection throughput, byte
costs, failed attempts and execution reports stay with the local artifacts.
