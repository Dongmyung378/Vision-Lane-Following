# Paired RGB waypoint datasets

An expanded dataset records moving CARLA RGB frames and same-frame waypoint
labels for central driving and small recovery perturbations. The model input
is RGB. Route, actor pose, camera calibration, speed, weather and perturbation
records support source validation and analysis; they are not additional model
inputs.

## Scene boundaries

Assign a route to train or dev before collecting its variants. Keep every
weather, route-start, central and recovery variant of a physical corridor in
that assignment. A different spawn index, route identifier or travel direction
does not establish a different scene. Compare complete surveyed corridors and
measured actor positions across splits, preserving the configured spatial
exclusion buffer and the previous development boundary. The reserved final
town remains unopened.

## Source and label checks

Each source episode retains its original RGB PNGs, same-frame transforms,
acknowledged controls, event callbacks, controlled stopping observations and
owned-actor cleanup. Reconstruct labels from the original target lane and the
recorded actor pose. Recovery labels must use the perturbed pose, rather than
copying central-driving actor coordinates or switching to whichever lane is
nearest.

Bind the initial actor heading to the tangent at the declared start on the
original route, plus the configured yaw perturbation. Audit recorded requested
headings against that same source independently. The measured actor tolerance
still applies; a nearest-segment change at a polyline vertex must not redefine
the heading that was requested.

The target is five points at future centreline arc distances 3, 6, 9, 12 and
15 metres. Coordinates are metres from the actor origin, with x forward and
y right. Camera projection overlays are a label inspection tool; their
calibration and pose must describe that same frame. An overlay does not
replace source replay or prove a learned model's driving performance.

Exclude stationary and startup frames using measured motion. Validate every
retained image's content hash, dimensions, RGB colour provenance and labels.
Reject duplicate sample identities and cross-split RGB duplicates; repeated
images must not fill a sample budget. Preserve failed attempts and their
rejection reasons with private collection artifacts.

## Equal-budget comparison

Freeze a central-only training selection A and a selection B with the same
total number of unique samples. B replaces some central samples with recovery
samples from matched scenes. Match within a fixed route, weather and start
setting using recorded reference arc distance and a predeclared maximum gap.
Do not match images by apparent similarity or choose samples using model
errors. Keep the shared central portion and development selection fixed.

Recovery episode membership alone is insufficient: a frame near the end of a
recovery can already show central driving. Use the measured lateral or heading
difference between the matched frames to identify active recovery examples.
For heading, subtract the continuously interpolated original-route anchor yaw
at each frame's reference arc from the actual actor yaw, then compare the
wrapped residuals. This avoids treating a change of polyline segment or normal
progress around a curve as an active yaw recovery. Preserve the recorded
tracking heading error separately for controller and evaluation replay.
Each selected recovery replaces its own central anchor in A; do not add scenes
that are absent from A or replace an unrelated central example. Record the
collection's total sample budget separately from the per-arm training budget.

Save selection identities, ordered sample identifiers, pairing distances and
hashes in the dataset snapshot before training. Compare A and B with the same
backbone, preprocessing, augmentation, number of updates and seed policy.
Holding the total budget equal does not imply equal recovery coverage; report
the number of samples available and selected in each case and condition.

## Immutable snapshot

A complete manifest binds the collection configuration, route assignments,
episode metadata and trace hashes, source images, labels, filter decisions,
pairs and training selections. Independently reconstruct these from unchanged
source bytes. Rewriting a manifest hash cannot substitute for reconstructing
its contents. Refuse incomplete coverage, changed sources, unsafe paths,
split conflicts or insufficient matched data instead of silently reducing a
frozen budget.

Dataset collection and validation prepare a training input. Any accuracy or
closed-loop improvement from that input requires a separate training and
evaluation experiment. Store generated manifests, images, overlays, logs and
actual execution reports in private artifact directories.

Inspect a collection plan without importing CARLA or starting a simulator:

```powershell
.venv\Scripts\python.exe scripts/run_dataset_pipeline.py --dry-run --config configs/dataset_v1.json
```

The native collector requires an already running, explicitly authorized CARLA
server. Its original collection interface still uses ClearNoon by default;
the expanded collector explicitly enables the configured development weather
presets and verifies their native numeric readbacks. Generated source
artifacts remain outside version control.

With that server already running, collect into new private directories using
a previously reviewed route catalog to preserve its train/dev assignments:

```powershell
.venv\Scripts\python.exe scripts/run_dataset_pipeline.py --config configs/dataset_v1.json --frozen-route-catalog results/local/inputs/route-catalog.json --output-dir results/local/dataset-example/raw --artifact-dir results/local/dataset-example/published
```

For the existing Kaggle runtime, prepare its execution bundle locally instead:

```powershell
.venv\Scripts\python.exe scripts/prepare_kaggle_dataset.py --existing-metadata results/local/inputs/existing-notebook-metadata.json --route-catalog results/local/inputs/route-catalog.json --output-dir results/local/dataset-bundle --config configs/dataset_v1.json
```

Packaging is offline and starts no simulator or GPU. Supply metadata for the
existing private notebook; its identity and title are retained. Before any
remote submission, perform a fresh read-only check that this same notebook
exists and is accessible. An access failure requires investigation rather
than creating or forking another notebook. Packaging itself submits nothing.

Reconstruct an extracted snapshot on CPU, including all original source
episodes and excluded frames:

```powershell
.venv\Scripts\python.exe scripts/verify_dataset.py --manifest <extracted-directory>/dataset_v1_manifest.json --output <new-private-report.json>
```

The CARLA client package is needed to compare saved weather values against its
preset definitions. This validation does not connect to CARLA or render
images. Existing output files are refused.

The collector also exports `qa/split_manifest.csv`, ordered central-only and
central/recovery training tables, a common dev table and
`training_selections.json`. Image paths in these references are relative to
the source manifest's parent directory. Selection indices preserve the frozen
sample order; the two training arms use the same dev identifiers.

## RGB-only selection loader

`DatasetV1WaypointDataset` consumes the original dataset manifest, a fixed
selection name and the PASS report from full native validation. The CSV files
are ordered export references for inspection and downstream integrations; the
loader reads the manifest's source-bound selections. Existing pilot training
and evaluation CLI commands retain their pilot-manifest interface and do not
accept these CSV files or a Dataset v1 manifest.

```python
from pathlib import Path
from carla_tools.dataset_learning import DatasetV1WaypointDataset

dataset_root = Path("path/to/private/extracted-dataset")
manifest = dataset_root / "dataset_v1_manifest.json"
validation_report = dataset_root / "dataset_validation_report.json"

train = DatasetV1WaypointDataset(
    manifest, "train", selection="central_recovery",
    validation_report_path=validation_report, augment=True,
)
dev = DatasetV1WaypointDataset(
    manifest, "dev", selection="central_recovery",
    validation_report_path=validation_report, augment=False,
)
```

Use `selection="central_only"` to consume the other fixed training arm. Both
arms expose the same development sample identifiers. Items contain the RGB
tensor, actor-coordinate waypoint target and evaluation identifiers; pose,
speed and calibration remain outside the image tensor.

A complete training integration must record the selection name, hashes of
the ordered train and dev sample identifiers, the dataset manifest and
snapshot hashes, and the bound native validation report hash in its reports
and checkpoints. The dataset manifest alone contains both arms, so its hash
does not identify which arm trained a checkpoint. Constructing this loader
prepares an input dataset; it does not execute or certify a training run.

`qa/index.html` links to contact sheets and full-size label overlays. Inspection
frames are chosen deterministically at the earliest and middle reference arc
of each split, geometry, weather and role stratum. Each overlay has a JSON
sidecar containing its original PNG, frame, route, actor and camera matrices,
calibration hashes, projected points and explicit visibility. Points outside
the image remain outside; they are not moved to the image border.

To regenerate these derived views separately, call
`carla_tools.dataset_artifacts.export_dataset_artifacts(manifest_path, validated_gate, output_dir)`
with the already completed native validation report and a new private output
directory. The function checks the report's manifest binding and rechecks the
inspected sources. It does not repeat the full native audit or run a model.
`validate_dataset_artifacts` independently reconstructs the exported views and
rejects altered tables, overlays and unbound extra files.
