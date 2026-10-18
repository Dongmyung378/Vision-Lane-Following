# Central and recovery learning comparison

This workflow measures the effect of recovery examples while keeping the
architecture, initialization and training budget equal. It uses the source-bound
[expanded dataset](dataset.md) and the `hybrid_geometry` model: a ResNet18 RGB
encoder, spatial attention and a decoder that accumulates learned path segments.
The prediction input remains RGB. Pose, reference paths and measured camera
calibration support labels and auditing; they are not concatenated to the model
input.

## Frozen dataset and comparison arms

`DatasetV1WaypointDataset` requires a complete dataset manifest and its bound
`PASS` source-validation report. It verifies the manifest snapshot, dataset
configuration, implementation hashes and reconstructed sample selections.
Original PNGs are hashed and decoded when consumed. Manifest and validation
report hashes are checked again at training boundaries.

The two training selections are fixed by the dataset manifest:

| Selection | Training examples |
| --- | --- |
| `central_only` | Central examples, including anchors for the paired recovery examples. |
| `central_recovery` | The shared central examples plus recovery examples replacing their corresponding anchors. |

Both selections contain the same number of examples. Recovery pairs come from
the same physical scene, weather and collection start, matched by reference arc
distance. The dataset validator checks the maximum arc gap and minimum active
lateral or heading difference. Every variation of a physical scene shares its
train/dev assignment. Both arms use the same complete development selection.

Keep the original sources, validation report and manifest unchanged. A compact
report transported to a training environment may bind a previously completed
native replay through its original report and episode-metric hashes. Such a
summary records that it reuses earlier CPU validation; it does not represent a
new native replay in the training environment.

## Equal training budget

The protocol lives in `configs/recovery_training.json`. Its defaults specify
5,000 training examples per arm, seeds 42 and 43, 32 epochs and batch size 64.
Each arm therefore performs 79 updates per epoch, including the final partial
batch, for 2,528 updates per seed. All configured epochs run; arm-specific early
stopping is disabled.

Each arm starts from the same complete `hybrid_geometry` checkpoint. The loader
checks the existing refined-model contract and pretrained-weight provenance.
The comparison records both the checkpoint file hash and a canonical hash of
its named tensor values, including scalar buffers. Optimizer and AMP scaler
states start fresh for every arm and seed.

Both arms use the same per-epoch index permutation, learning-rate schedule,
loss, optimizer groups and gradient clipping. Photometric augmentation and the
half-probability horizontal reflection use a key derived from training seed,
epoch and selection index. Corresponding central/recovery slots receive the
same augmentation draws. Reflection negates waypoint y and exchanges the left
and right case labels used for reporting. It reflects the original camera grid
about its fixed principal point before the common crop and resize.

Preprocessing keeps the full native image width, crops rows `[144, 360)`, resizes
to `320 × 180` with OpenCV `INTER_AREA`, and applies fixed ImageNet channel
normalization. Labels retain the vehicle actor origin, metres and shape `5 × 2`.
Training uses the existing horizon-weighted SmoothL1 loss. Encoder BatchNorm
running statistics remain frozen, while its affine parameters and encoder
weights receive gradients. The optimizer configuration and warmup/cosine
schedule are included in the recorded protocol.

## Checkpoint and seed selection

Before any updates, the driver fixes a development subset by round-robin
sampling case, weather and condition strata. Ranking uses sample identity and
a fixed seed, without inspecting predictions. The default subset contains
2,048 images and is identical for every arm and training seed. A dataset with
fewer available development images is rejected for this protocol.

The lowest loss on this subset selects each arm's checkpoint. Ties retain the
earlier epoch. The representative pair uses one common training seed: minimize
the mean of the two arms' best subset losses, breaking a tie by numeric seed.
`native_pair_selection.json` records this decision before full-development
evaluation starts.

Every selected checkpoint, across both arms and both seeds, is then evaluated
on the entire common development set. The unchanged initial checkpoint is
evaluated on those same images. These scores are development evidence. Reserved
towns and final-evaluation conditions remain unopened.

Evaluation reports contain:

- Coverage, failed predictions, near/far errors and per-waypoint errors in metres.
- Case, condition, weather, case/condition and case/weather groups.
- Ordered common-development sample hash, case counts and inference-contract hash.
- Checkpoint, manifest, validation-report, configuration, protocol and source hashes.
- Per-image predictions and targets, and descriptive variation across training seeds.

The inference contract excludes training seed and optimizer settings, allowing
models with different training configurations to prove that they use the same
RGB representation and waypoint interpretation. Forward latency uses a separate
fixed, warmed batch-one sample set. Its scope includes model forward execution
and transfers and excludes image decoding, preprocessing and control.

## Run training

Prepare and validate the expanded dataset first. Supply an existing compatible
refined checkpoint as the shared initialization. Output directories must be new.
Run the GPU command inside the approved existing Kaggle training notebook:

```bash
python scripts/train_recovery_waypoints.py \
  --manifest /path/to/dataset/dataset_v1_manifest.json \
  --validation-report /path/to/dataset_validation.json \
  --initial-checkpoint /path/to/initial/checkpoint.pt \
  --protocol configs/recovery_training.json \
  --output-dir /path/to/recovery-experiment \
  --device cuda
```

The default CLI device is CPU. The Python API has the same explicit device
choice:

```python
from carla_tools.recovery_training import (
    load_recovery_protocol, run_recovery_comparison,
)

report = run_recovery_comparison(
    manifest_path, validation_report_path, initial_checkpoint_path, output_dir,
    protocol=load_recovery_protocol(), device="cpu",
)
```

`scripts/prepare_kaggle_recovery_training.py` prepares a source bundle for the
existing private training notebook. It does not submit it. The caller checks
that notebook's identity and access immediately before submission. The bundle
pins the existing capture archive and prior initialization version, verifies
their hashes, reconstructs the exact manifest and transports the prior CPU
validation proof. It records whether its training processes have stopped and
preserves partial artifacts if a run fails. Reuse the existing notebook and
keep credentials out of source and artifacts.

A large archive can also be attached from an existing completed CPU notebook
output through the prepared submission metadata. Verify that existing notebook's
identity, version and code before submission, and bind the metadata hash to the
experiment record. The resolver checks the entire mounted archive's SHA; its
filename or input reference alone is insufficient. Preserve the original capture
version and archive hash when using this staged copy, and reuse the existing
notebooks.

## Output and checkpoint loading

The output root contains `comparison.json`, the frozen `protocol.json` and
`native_pair_selection.json`. The original checkpoint bytes are copied to
`initial/checkpoint.pt`, with full-development metrics in `initial_evaluation/`.
Each `seed<seed>/<arm>/` directory contains its training report, selected
checkpoint and full-development `evaluation/metrics.json` and
`evaluation/predictions.jsonl` files.

`central_only/checkpoint.pt` and `central_recovery/checkpoint.pt` at the output
root are byte copies of the representative seed's checkpoints. Their
`evaluation/metrics.json` files copy that seed's full-development metrics.
These stable paths simplify consumers; all seed-specific outputs remain
available for inspection.

Load the source-bound recovery checkpoint through its own loader:

```python
from carla_tools.recovery_training import load_recovery_checkpoint

model, config, checkpoint = load_recovery_checkpoint(
    checkpoint_path, device="cpu", expected_sha256=checkpoint_sha256,
    manifest_sha256=manifest_sha256,
)
```

The checkpoint role is `selected-recovery-model`; its arm, training seed,
model-state hash and training provenance remain available in `checkpoint`.
Adjacent `refined_checkpoint.pt` exports support the existing refined-model
runtime interface. They preserve a `recovery_provenance` chain back to the
source-bound checkpoint. The paired driving adapter uses the stricter recovery
loader for its new models.

## Paired native driving and selection

`carla_tools.recovery_comparison.freeze_driving_protocol` binds development
routes, all checkpoint hashes, the training-comparison hash, source hashes and
selection policies before driving. The packaged native comparison uses both
default training seeds: four trained models plus the unchanged incumbent.
Every model runs the same route, start and perturbation matrix, with arm order
alternating between states. The controller, nominal RGB path correction and
termination rules are identical. Raw learned predictions remain in the
artifacts alongside corrected predictions.

`scripts.prepare_kaggle_recovery_driving.prepare_recovery_driving` prepares the
existing private CARLA notebook using a pinned training archive. It checks
selected checkpoint and evaluation-report hashes before execution. The
generated pipeline calls `scripts/run_recovery_driving.py` with `--routes`,
`--protocol`, `--models`, `--training-comparison`, `--output-dir` and an explicit
`--device`. Its outputs include `native_report.json`, per-arm metric CSVs and
episode artifacts for source replay.

Selection requires lower overall full-development waypoint error and lower
equal-episode mean lateral error than the incumbent. Each case must also meet
the frozen absolute offline and driving-error bounds, coverage, clean-completion,
collision, lane-invasion, body-margin and latency requirements. A case can worsen
relative to the incumbent while remaining within its bound. Among eligible
models in the preselected representative seed pair, the selector ranks driving
mean lateral error first and offline mean error second. It retains the incumbent
when neither representative qualifies. Results from the other training seed
remain part of the comparison and seed-effect summary.

`select_recovery_model` returns a decision and failed criteria without updating
a deployment pointer. Artifact integrity, successful execution and model
improvement are separate conclusions. Short, correlated development routes,
two training seeds and one simulator configuration do not establish independent
generalization or deterministic physics. Independent final evaluation remains
a separate procedure.

Store execution reports, images, logs, datasets and weights in ignored private
output directories. Public documentation describes the method and contracts;
actual execution evidence is kept outside source control.
