# Freeze an independent driving evaluation

The final evaluation workflow binds a selected RGB pipeline, a static route
catalog, initial states, metrics and the exact source and data versions before
any final image, prediction or driving outcome is observed. Preparing and
verifying a freeze uses CPU file checks; it does not start CARLA.

## Paired evaluation plan

The default plan contains five groups. Each group uses two routes and three
explicitly different starting poses. B0, M1 and the privileged reference run
each state, giving 30 episodes per method and 90 episodes in total.

| Group | Map and physical condition |
| --- | --- |
| A | Unseen Town01 routes, ClearNoon |
| B | Town02 routes, ClearNoon |
| C | The reserved Town01 routes, HardRainNoon |
| D | The reserved Town01 routes, ClearSunset |
| E | The reserved Town01 routes, ClearNoon with camera pitch increased by 2 degrees |

The starting states use reference positions 0, 2 and 4 m. The latter two also
apply lateral/yaw offsets of +0.2 m/+2 degrees and -0.2 m/-2 degrees. Seeds
42, 43 and 44 identify these fixed states; they are not three independent
training seeds or randomized traffic scenes. All methods use the same states.
Method order rotates between states.

Each drive measures 15 m after initialization, with a 10 s simulation timeout,
a 25 km/h speed cap and a 0.05 s simulation step. A route includes at least
35 m of continuous nonbranching geometry, preserving the future 15 m label
horizon after the latest planned start. These short routes measure local lane
following and should not be described as long-distance navigation.

M1 uses the checkpoint, RGB preprocessing and controller from a
[selected pipeline](selected-pipeline.md). B0 uses its fixed nominal camera
calibration. The actual pitch-shift calibration is not given to either
perception method. The reference receives the original route and same-frame
vehicle pose, and is explicitly privileged. All three methods use the selected
controller and measured native vehicle geometry.

## Static topology and independence

The catalog contains two route contracts for each map. Offline CARLA
`Map` parsing of OpenDrive can establish lane geometry without rendering or
collecting final images. Bind each map's SHA-256, lane identifiers, longitudinal
coordinates, world anchors and headings.

Offline OpenDrive does not establish packaged-map spawn-point indices. Static
routes therefore store a null spawn index. At execution, the adapter checks the
live map and every anchor, then selects the actual spawn point closest in XY to
the frozen route start, breaking ties by index. The native drive loop resets to
the fixed route and initial state before measurement. An unavailable or
incompatible map, anchor or spawn is a technical problem; it does not authorize
selecting an easier replacement route.

`audit_route_exclusion` checks route and scene identities and continuous XY
segment separation against the supplied training and development inventory.
The minimum separation is 75 m, including opposite lane directions. The
inventory must cover every dataset and prior development route, not just the
latest experiment. Bind that inventory and its provenance. Geometry-only checks
do not establish that a model can drive a route.

## Prepare and freeze

Create a private JSON array of artifact bindings. Each entry contains a
project-relative path, file SHA-256 and one or more roles:
`dataset`, `split`, `selection`, `topology`, `budget`. The full list must cover
all five roles. A dataset manifest that also contains split assignments can
serve both roles. Keep actual results, paths, budgets and reports outside
tracked documentation.

```powershell
python scripts/prepare_final_evaluation.py `
  --routes results/local/final-plan/routes.json `
  --selection results/local/selected-pipeline.json `
  --artifact-bindings results/local/final-plan/artifact-bindings.json `
  --output results/local/final-plan/evaluation-manifest.json

python scripts/freeze_final_evaluation.py create `
  --manifest results/local/final-plan/evaluation-manifest.json `
  --routes results/local/final-plan/routes.json `
  --selection results/local/selected-pipeline.json `
  --development-review results/local/final-plan/development-review.json `
  --artifact results/local/final-plan/dataset-manifest.json `
  --artifact results/local/final-plan/selection-evidence.json `
  --artifact results/local/final-plan/topology-evidence.json `
  --artifact results/local/final-plan/budget-evidence.json `
  --output results/local/final-plan/freeze.json
```

Supply every artifact named in the manifest, repeating `--artifact` as needed.
The example dataset manifest must cover both dataset and split roles. The
output parent must exist, and outputs must use new filenames.

The freeze binds the raw manifest, routes, selection pointer, model checkpoint
and development review, all declared artifacts, and the public source tree.
The source tree includes code, tests, configurations, schemas, requirements and
public documentation. Private notes and generated caches are excluded.
Verification detects file edits, removed bindings and newly added source files.
The Git commit is contextual information; the file hashes identify the exact
execution snapshot, including uncommitted implementation files.

```powershell
python scripts/freeze_final_evaluation.py verify `
  --freeze results/local/final-plan/freeze.json `
  --expected-freeze-id <independently-retained-freeze-id>
```

Retain the returned freeze ID separately. Recomputing an editable self-hash
does not establish external authenticity. A freeze is published once and is
never replaced in place. Keep mutable completion notes outside its bound files.

## Execution and failure handling

The native final runner verifies the freeze before importing CARLA or loading
the selected predictor. Its isolated adapter reuses the existing capture,
same-frame driving, event handling and cleanup functions while substituting
the explicitly supported final map/weather parser and static-topology verifier.
It does not mutate the historical modules' global namespaces. The same
geometric route tracker additionally measures reference-label construction
time; privileged-reference timing is reported separately from RGB perception.

Native driving requires an approved GPU simulator environment. Reuse an
existing Kaggle notebook and verify its identity before submission. An offline
CPU planning or mock test result does not establish native execution success.

Only after the freeze has passed verification and final evaluation is intended,
run the declared groups against a dedicated CARLA server:

```bash
python scripts/run_final_evaluation.py \
  --freeze results/local/final-plan/freeze.json \
  --expected-freeze-id <independently-retained-freeze-id> \
  --groups A C D E --device cuda --execute-heldout \
  --output-dir results/local/final-town01-run
```

Run group B separately for Town02 if the measured runtime budget calls for two
jobs. A group subset retains every route, initial state and method in that group.
It is not a complete final comparison until all five groups are accounted for.
The explicit execution flag distinguishes the native command from CPU planning;
it does not replace the user's GPU authorization.

All failed trials remain in the planned denominator. Preserve technical
failures and every attempt. A documented infrastructure failure permits at most
one additional attempt with the identical freeze and trial; collision, lane
departure, stall, timeout, prediction failure and controller rejection are
method outcomes, not retry reasons. Changes to evaluation meaning invalidate
the freeze and require reevaluating all affected methods.

## Metrics and budget

The manifest fixes [evaluation definitions](evaluation.md), termination
precedence, event deduplication, recovery criteria and latency scope. Report
completion and clean completion alongside pooled frame error and episode
summaries. A method that fails early can have a small observed error; include
its failed outcome and traveled distance.

RGB processing latency includes BGRA conversion, perception and controller
computation. It excludes rendering, sensor waiting, ground-truth generation,
disk writes and video encoding. Compare median and p95 to the 50 ms processing
budget, with any steady-state warmup exclusion reported separately.

Estimate execution time and storage from existing development measurements,
including failed episodes, setup/download costs and a contingency allowance.
Keep Kaggle queue delays outside a numerical runtime guarantee. Planning
90 episodes is a workload estimate, not evidence that those episodes have run.
