# RGB waypoint learning

The learned predictor receives one RGB image and returns five path waypoints at
3, 6, 9, 12 and 15 metres of future path arc length. Coordinates use the vehicle
actor origin, with x forward and y right, in metres. Arc distance is the label
sampling rule; it does not require x to equal the sampling distance on a curve.
Waypoint validation checks increasing distance from the actor origin rather
than increasing x: a turn past 90 degrees can move farther along the path while
forward x decreases. The downstream controller converts points to the measured
rear-axle origin and independently checks their order there. Actual axle
geometry is not an input to the RGB model.
Speed remains an input to the downstream controller and is absent from the
image model. Simulator pose, routes, lane identifiers and measured camera
calibration support data auditing and ground-truth labels, and never enter the
predictor input.

## Dataset and preprocessing

`carla_tools.learning_data.LaneWaypointDataset` reads a complete moving-pilot
manifest. It checks sample counts, finite ordered labels, original 640 × 360
RGB PNG dimensions and SHA256 hashes. Relative image paths must remain inside
the manifest directory and cannot use symlinks. Images are hashed again when
loaded; the manifest can be checked again at epoch and checkpoint boundaries.
PNG checks use the original full image before any preprocessing.

Train and development assignments apply to entire episodes, routes and scene
groups. The loader rejects cross-split group overlap, duplicate sample IDs and
duplicate RGB files or hashes. Spatial route separation and same-frame label
reconstruction are verified by the pilot source validator before training.
The reserved final-evaluation town is not a source for this pilot.

Preprocessing receives the full 640 × 360 RGB image and selects the fixed
full-width native-pixel rectangle `[x0, y0, x1, y1] = [0, 144, 640, 360]`.
Bounds are half-open: rows `[144, 360)` and columns `[0, 640)` form a 640 × 216
image, resized to 320 × 180 using OpenCV `INTER_AREA`. Training, development
and inference apply the same crop and resize. The pixel representation changes;
the five actor-coordinate waypoint labels remain in metres with shape `5 × 2`.
There is no horizontal flip, perspective warp or geometric jitter. Fixed
ImageNet channel means `(0.485, 0.456, 0.406)` and standard deviations
`(0.229, 0.224, 0.225)` normalize values divided by 255. No normalization
statistics are fitted to the development split.

Training augmentation changes brightness, contrast and saturation only, before
the fixed crop and resize. It preserves the source pixel grid and metre labels.
A seed derived from the configured seed,
epoch and sample ID makes augmentation independent of worker scheduling. The
development split is never augmented. DataLoader workers should be recreated
each epoch (`persistent_workers=False`) after `dataset.set_epoch(epoch)`.

## Model and loss

`carla_tools.waypoint_model.WaypointModel` uses the torchvision ResNet18 encoder
and a spatial regression head. The 320 × 180 input produces a 512-channel
6 × 10 feature map. Eight nonoverlapping bins preserve two vertical regions
and four horizontal regions before a `4096 → 256 → 10` head with ReLU and
dropout. Row boundaries are `(0, 3, 6)` and column boundaries are
`(0, 2, 5, 7, 10)`. Explicit slice means support deterministic backpropagation
without adaptive-pooling CUDA backward operations. The official
[ImageNet pretrained ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
are loaded strictly from the official download or an explicitly supplied file.
The official SHA256 prefix is checked and the complete file hash is recorded.
A download failure, missing file or incompatible state raises an error; it
never silently switches to random initialization. `pretrained=False` is an
explicit option for synthetic tests and checkpoint restoration.

The complete original encoder state is loaded before replacing its parameterless
global pool and classification layer. During training, encoder BatchNorm layers
use the stored pretrained running statistics. Their affine parameters and other
encoder weights still receive gradients, while head dropout remains active.
The separate, smaller encoder learning rate preserves pretrained features.
The fixed native-pixel crop and resize, spatial architecture, pooling
boundaries, BatchNorm behavior and head configuration are part of the
checkpoint contract. The current model version is
`vlf-resnet18-road-roi-waypoint-v3`; a configuration from another model version
is rejected rather than interpreted as this model.

The head predicts a residual from a declared straight-path prior
`[(3, 0), (6, 0), (9, 0), (12, 0), (15, 0)]`. Multiplying head outputs by a fixed
15-metre scale produces coordinates in metres. Zero initialization of the final
layer gives a known starting path; training learns both x and y corrections.
This prior is a fixed model choice, not a route or ground-truth input.

SmoothL1 compares both axes after dividing predictions and targets by 15. The
default beta is `0.03`, corresponding to a 0.45-metre transition. Reported
waypoint errors remain in metres and use Euclidean distance, separately by
horizon, geometry and initial condition. Loss values are not metre errors.
The fixed configuration is in `configs/learning.json` and has a canonical
SHA256. Its development targets are provisional offline accuracy goals; they
do not establish closed-loop driving performance.

## Installation and API

Install the matched PyTorch/torchvision wheels from the
[official PyTorch installation source](https://pytorch.org/get-started/previous-versions/).
For local CPU validation:

```powershell
.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-learning.txt
```

Kaggle training uses the corresponding CUDA wheels and records the actual
runtime versions and GPU. Model inference accepts the full 640 × 360 RGB image
and applies the same fixed crop and resize internally:

```python
from pathlib import Path
from carla_tools.training import load_checkpoint

model, config, checkpoint = load_checkpoint(
    Path("results/local/waypoint-training/checkpoint.pt"), device="cpu"
)
prediction = model.predict_rgb(rgb_uint8_640x360)
```

Inference returns the shared `status`, `waypoints_ego_m`, `error`, `diagnostics`
and `timing_ms` fields. Nonfinite, unordered or out-of-range predictions return
an explicit failure with no retained previous path. Public documentation
describes contracts and procedures; dataset artifacts, checkpoints and actual
execution reports belong under `results/local/`.

## Local checkpoint inference

`carla_tools.inference.OfflineWaypointPredictor` restores a selected model once
and predicts each original RGB frame independently. The loader checks the
checkpoint's configuration, model contract, pretrained provenance and training
source hashes. It rejects a `training-state` checkpoint: use `checkpoint.pt`
for prediction and `last_checkpoint.pt` for training continuation. Optional
SHA256 pins verify the selected checkpoint bytes and training dataset identity.

```python
from pathlib import Path
from carla_tools.inference import OfflineWaypointPredictor

predictor = OfflineWaypointPredictor(
    Path("results/local/waypoint-training/checkpoint.pt"), device="cpu"
)
prediction = predictor.predict_rgb(rgb_uint8_640x360)
```

The PNG command checks the original image format, dimensions and RGB8 channels
before preprocessing. Symlinks, grayscale, alpha channels and 16-bit images
are rejected. Supply a new output path; an existing report is never replaced.

```powershell
.venv\Scripts\python.exe scripts/predict_waypoints.py --checkpoint results/local/waypoint-training/checkpoint.pt --image results/local/example-rgb.png --output results/local/local-inference/prediction.json --device cpu --mode learned
```

The JSON records the model identity, runtime and input image hash alongside the
prediction. The default `learned` mode uses the CNN. Explicit `--mode assisted`
uses the OpenCV-first policy described below and records which predictor ran.
Both modes return five actor-origin points in metres. The model forward pass
already applies its residual scale; multiply neither output axis by 15 again.
No route, pose, speed or target label is supplied to this interface. A failed
frame returns no path, including after an earlier successful prediction.

## Selecting a development checkpoint

Within a training run, the lowest held-out development SmoothL1 selects the
checkpoint. Compare different runs using the same frozen development sample
IDs, original image hashes, metre labels and failure accounting. Check complete
coverage first, then compare near and far metre errors together with geometry,
initial-condition groups, p95 errors and prediction overlays. Record the
selection rule and every rejected candidate locally; changing a rule after
examining development results is a development decision that must be recorded.
Never use the reserved final town to choose a model.

Report pure CNN and OpenCV-assisted results separately. An assisted mean does
not replace the CNN's error or establish its performance on a difficult curve.
Selecting the best available development checkpoint does not imply that all
accuracy targets pass or that closed-loop driving has been validated. Preserve
such failures and inspect them before connecting the model to vehicle control.

Before handoff, verify the image/label contract, a real small-batch loss decrease,
checkpoint continuation, fixed development errors and visual review. Record
completed epochs, selected epoch, measured training time, device, AMP and the
reason training stopped. Keep execution duration separate from remote waiting
time and compare costs only alongside the actual configuration and data size.
Retain the selected model and the last continuation state as separate artifacts.
An inference summary distinguishes the checkpoint execution epoch from the
epoch that selected the best model. When continuation finds no better model,
the new selected file retains earlier best weights even though its training
history extends to a later execution epoch.

## Optional OpenCV-assisted inference

`carla_tools.rgb_predictor.BaselineAssistedWaypointPredictor` first runs the
OpenCV RGB baseline. If its result passes the shared finite, ordered actor
waypoint contract, that result is returned. Otherwise the learned predictor
processes the same original RGB image. Selection uses only prediction validity;
targets, geometry groups, routes, pose, speed and previous paths are absent from
the selection API. Both failures return an explicit failure with no path.

```python
from carla_tools.rgb_predictor import BaselineAssistedWaypointPredictor

predictor = BaselineAssistedWaypointPredictor(model)
prediction = predictor.predict_rgb(rgb_uint8_640x360)
selected = prediction["diagnostics"]["selected_predictor"]
```

This policy retains the baseline's fixed nominal camera, flat ground plane and
single-boundary lane-width assumptions. A valid prediction is not an accuracy guarantee. Keep
pure learned and assisted results separate when assessing improvements.

```powershell
.venv\Scripts\python.exe scripts/evaluate_rgb_predictor.py --manifest results/local/pilot/pilot_manifest.json --checkpoint results/local/waypoint-training/checkpoint.pt --output-dir results/local/assisted-evaluation --device cpu
```

The assisted evaluation executes actual RGB inference on every development
sample. It records predictor selection counts, failures and coverage, geometry
and condition groups, and the unchanged near/far accuracy targets. Its latency
includes baseline inference and optional learned preprocessing and inference;
PNG decoding is excluded. It does not measure closed-loop driving performance.

## Training and held-out evaluation

The training runner first checks that a fixed, distributed batch of training
images can be overfit with actual forward and backward passes. It then discards
those diagnostic weights, resets the seeds and starts ordinary training from
the verified pretrained encoder. AdamW uses separate encoder and regression
head learning rates, warmup and a deterministic cosine schedule. Mixed precision
is enabled on CUDA; local CPU validation uses full precision. Development
SmoothL1 selects the checkpoint and controls early stopping.

```powershell
.venv\Scripts\python.exe scripts/train_waypoints.py --manifest results/local/pilot/pilot_manifest.json --output-dir results/local/waypoint-training --device cpu --max-epochs 1
.venv\Scripts\python.exe scripts/train_waypoints.py --manifest results/local/pilot/pilot_manifest.json --output-dir results/local/waypoint-resumed --device cpu --resume results/local/waypoint-training/last_checkpoint.pt
.venv\Scripts\python.exe scripts/evaluate_waypoints.py --manifest results/local/pilot/pilot_manifest.json --checkpoint results/local/waypoint-resumed/checkpoint.pt --output-dir results/local/waypoint-evaluation --device cpu
```

Use a new output directory for every execution. `last_checkpoint.pt` contains
the coherent completed-epoch model, optimizer, scaler and random-generator state
for continuation. `checkpoint.pt` contains the selected model for inference and
development evaluation and is not accepted for training continuation. A resume
requires matching configuration, pilot
manifest, model contract and runtime sources. Restored optimizer tensors,
scaler, random-generator and sampler states are compared with their saved
values before training continues. Resume changes the execution duration, not the
frozen target, data split or total configured training schedule.

Evaluation opens the held-out development split, reports every attempted sample
and separates failed predictions from successful-sample errors. Near horizons
are 3 and 6 metres; far horizons are 9, 12 and 15 metres. Results include mean,
95th-percentile and maximum Euclidean error, x/y absolute errors and coverage,
with separate geometry and initial-condition groups. The learned model,
straight prior and OpenCV baseline use the same development RGB. Baseline
failures remain in its coverage denominator; comparison on its successfully
detected samples is reported as an explicitly identified common subset.

Saved overlays project the actor-coordinate metre waypoints onto the original
640 × 360 RGB image using the fixed nominal camera projection. Green targets
are privileged labels drawn after inference for QA only; red points are model
predictions. The measured forward latency excludes PNG decoding and image
preprocessing and should not be interpreted as end-to-end vehicle control
latency.

For GPU work, `scripts/prepare_kaggle_training.py` packages public runtime
sources and the exact private pilot archive/manifest hashes for the existing
training notebook. Before submission, read-only access to that notebook must
be verified. The runner normally mounts the existing collection notebook output.
The optional `--pilot-kernel-version` requests a positive numeric version of
that same notebook using the SDK's `owner/slug/version` attachment form. This
request does not establish which bytes the service will expose: the exact
archive and manifest hashes must still match before training starts.
If that attachment exposes an earlier successful output instead of the verified
pilot archive, the optional `--pilot-dataset` selects the configured stable
private pilot dataset. That dataset must be prepared and checked independently
before packaging; the packager creates no remote resources. Other dataset
attachments are preserved, and the collection-kernel attachment is omitted
for this explicit transport. Both transports require the same exact archive
and manifest SHA256, so an older or different pilot is refused.

The runner extracts the pinned pilot, installs matched CUDA dependencies and
preserves checkpoints and evaluation artifacts. It does not create a notebook
or use the reserved final town.

To collect and train in one execution of the existing CARLA notebook, use
`scripts/prepare_kaggle_pilot_training.py --existing-metadata <metadata.json>
--output-dir <new-local-bundle-directory>`. This packages the unchanged pilot
workload followed by a separate training runtime. The handoff requires a
successful native validation gate, an explicitly stopped CARLA server and
matching SHA256 values for the produced ZIP, on-disk manifest and archived
manifest. Those verified bytes become the immutable training input in the
same session. The native and training source manifests remain separate, and
collection and training have independent bounded execution budgets. The
packager performs no remote submission, upload or resource creation.

## Frozen ViT comparison

An optional DINOv3 ViT-B/16 candidate preserves the same RGB source, fixed road
ROI, actor-coordinate waypoint labels and route-separated development split.
Install `requirements-vit.txt` with the appropriate CPU or CUDA PyTorch wheels.
The verified pretrained artifact comes from the timm DINOv3 conversion, pinned
to an immutable revision and full SHA256. Loading never falls back to random
weights. The upstream DINOv3 license continues to apply.

The encoder remains frozen and in evaluation mode. After the original fixed
320 x 180 preprocessing, the candidate replicates the bottom row into twelve
additional rows so the encoder receives a 192 x 320 image without silently
cropping incomplete patches. Eight disjoint spatial patch groups feed a
regression head that predicts the same five metre-valued waypoints. CUDA uses
FP16 for feature extraction and FP32 for the regression head. PyTorch math
attention retains FP32 intermediate calculations to avoid FP16 attention-score
overflow; the runtime explicitly selects this backend.

Training caches RGB features without augmentation, verifies a small training
batch can be overfit, restores the fresh head, and selects the checkpoint by
development SmoothL1 loss. This protocol differs from the ResNet encoder
fine-tuning and photometric augmentation, so the comparison evaluates two
complete candidate pipelines rather than isolating architecture alone.

```powershell
.venv\Scripts\python.exe scripts/train_vit_waypoints.py --manifest <pilot-manifest.json> --weights <verified-pretrained.pth> --output-dir <new-private-output-directory> --device cuda
```

For remote execution, `scripts/prepare_kaggle_vit.py` packages the candidate for
the existing training notebook. It retrieves an explicitly numbered existing
collection output, checks the archive and manifest hashes, and downloads the
public hash-pinned pretrained weights. It requires a T4 CUDA runtime and has
bounded execution and owned-process cleanup. No new notebook or dataset is
created. Read-only access to the target notebook must be checked before
submission.

Evaluation reopens every development RGB image and executes the complete
encoder and head. It retains failed predictions in coverage, reports geometry
and condition groups, and measures synchronized forward latency separately
from decoding, preprocessing and vehicle control. Offline waypoint accuracy
does not establish closed-loop driving success. Keep experiment outputs,
checkpoints and actual comparison records in ignored local directories.

## Controlled CNN and attention comparison

The spatial comparison keeps the original pooled ResNet reference and adds two
RGB-only candidates. Both candidates retain the ResNet encoder's 6 x 10 feature
grid and project its 60 locations from 512 to 128 features. The spatial control
flattens these tokens into an MLP. The hybrid adds fixed two-dimensional position
encodings, two self-attention blocks, and five waypoint queries with query
self-attention and cross-attention. Each query predicts a two-axis residual
relative to the same straight-path prior. The experimental heads use float32;
CUDA training can use mixed precision in the CNN. Attention explicitly uses the
PyTorch math backend.

All three arms fine-tune the complete ImageNet-pretrained encoder with the same
route-separated samples, road crop, photometric augmentation, seed, AdamW rates,
warmup, epoch budget and early-stopping rule. Each arm discards its small-batch
diagnostic model and starts again from fresh initialization. The selected model
minimizes development SmoothL1. Candidate checkpoint contracts distinguish the
architecture and bind the configuration, source files and dataset manifest.
These inference checkpoints do not contain a resumable optimizer state.

The fixed-batch diagnostic caps its head learning rate at the regular head
learning rate. This diagnostic changes no retained model weights: the model
and seeds are reset before ordinary fine-tuning. A completed reference arm can
be supplied with `--reused-baseline` to avoid retraining it. Reuse requires the
same configuration, manifest, checkpoint binding and core prediction sources;
the report retains its original source hashes and execution provenance. Keep
the reference's original checkpoint and prediction archive alongside the new
candidate outputs. The reuse does not claim a new reference execution.

```powershell
.venv\Scripts\python.exe scripts/train_spatial_waypoints.py --manifest <pilot-manifest.json> --output-dir <new-private-output-directory> --device cpu
```

`scripts/prepare_kaggle_spatial.py` prepares the same comparison for the existing
private training notebook, preserving its ID and title. It creates only a local
bundle, embeds exact runtime source bytes and requires immutable collection
archive and manifest hashes. Verify access to the existing notebook before
submission. The remote runner requires a T4, enforces a total execution limit,
stops its owned process on timeout and exports all three checkpoints and full
development predictions without creating notebooks or datasets.

Evaluation retains every attempted development frame in coverage and reports
near/far errors by curve direction and initial condition. Batch-one forward
latency excludes decoding, preprocessing and vehicle control, and excludes
warmup frames. Current selection uses the mean error over all five waypoints,
complete coverage, absolute near/far mean budgets for every geometry case,
and a fixed forward latency budget. A case may be worse than the reference
while remaining within its budget. This produces an offline development model;
closed-loop validation and the reserved final test are separate requirements.
Keep actual scores, execution logs and selection records in ignored local
directories.

Use `SpatialCheckpointPredictor` for explicit RGB inference from one selected
spatial checkpoint. It defaults to CPU, verifies the checkpoint contract and
runtime sources, optionally requires an expected full checkpoint SHA256, and
reports the actual candidate variant and model version. Invalid RGB inputs
return a failed prediction. The existing default predictor is unchanged.

For a retained checkpoint whose training runner has changed, the optional
`--training-source-manifest` supplies explicitly trusted archived source
hashes. The current model implementation and shared core sources must still
match; historical training sources are never accepted automatically.

```powershell
.venv\Scripts\python.exe scripts/predict_spatial_waypoints.py --image <native-rgb.png> --checkpoint <selected-checkpoint.pt> --expected-checkpoint-sha256 <sha256> --output <new-private-prediction.json> --device cpu
```

## Mirrored and geometric attention refinements

The refinement experiment uses the same compact attention encoder with two
decoder choices. Both use train-only horizontal reflection about the fixed
camera principal point and negate the target's actor-coordinate lateral axis.
Development images remain unchanged. Deterministic sample/epoch seeds control
the reflection, and neither route identifiers nor geometry classes reach the
model. The normalized SmoothL1 horizon weights are `[1, 1, 2, 2, 3]`.

The mirrored control predicts ordinary Cartesian residuals. The geometric
decoder predicts five segment directions and lengths, then cumulatively sums
them into the same five actor-origin waypoints. Directions use `pi*tanh(raw)`.
Lengths use a 3m prior with learned bounded adjustments; the first segment has
a wider bound to account for actor displacement from the route. This is a
path-distance prior and may limit performance outside the training geometry.
It does not query a map or measured vehicle pose. Comparisons to the original
models change augmentation and loss as well as the decoder; the two refinement
arms share augmentation, loss and training budget.

```powershell
.venv\Scripts\python.exe scripts/train_refined_waypoints.py --manifest <pilot-manifest.json> --reused-baseline <trusted-frozen-reference.json> --output-dir <new-private-output-directory> --device cpu
```

Checkpoint selection minimizes the complete development weighted loss. The
separate current model selection requires improved overall mean error, complete
coverage, every geometry case within its absolute near/far mean budgets, and
acceptable forward latency. Tail errors remain reported; case-wise relative
improvement against the reference is not required. Model contracts and
source hashes bind the decoder, training transformation and retained weights.
Actual experiments and model selection records belong in ignored directories.

`carla_tools.model_selection` applies the current policy separately from saved
training/checkpoint source contracts. The policy in `configs/model_selection.json`
requires near mean at most 0.3m and far mean at most 1.0m for the overall, left,
right and straight groups, complete coverage, lower overall mean than the
reference, and forward p95 at most 50ms. These are group-mean budgets; individual
frame maxima and far-error p95 are separate reported statistics. Saved training
comparisons retain their historical embedded decisions for reproducibility.
The authoritative current decision is the separate selection artifact produced
by `select_waypoint_model.py`; future Kaggle refinement bundles run this step
after training. The selection change does not require retraining or rewriting
checkpoint files.

```powershell
.venv\Scripts\python.exe scripts/select_waypoint_model.py --reference <frozen-reference.json> --comparison <comparison.json> --output <new-private-selection.json>
.venv\Scripts\python.exe scripts/predict_selected_waypoints.py --selection <selection.json> --image <native-rgb.png> --output <new-private-prediction.json> --device cpu
```

The selected-model inference command verifies the checkpoint bytes, training
report and dataset identity. Its default selection path is the ignored local
`results/local/selected-waypoint-model.json`. Selecting a project development
model and verifying actual native driving remain separate steps.

For single-image inference, `predict_refined_waypoints.py` accepts a native RGB
PNG and requires the expected checkpoint SHA256. It verifies the model contract
and source files and writes a new prediction JSON without overwriting existing
artifacts. The default device is CPU.

`run_refinement_pipeline.py` connects explicit checkpoint predictors to the
shared native controller. It compares the reference and candidate on frozen
development routes alongside the fixed OpenCV baseline. The paired protocol
uses straight, left and right curves, original-route starts at 0/2/4m, and
center or lateral offsets of -0.3/+0.3m: 81 episodes across three methods.
The common original-route goal is 19m; each start has its own remaining
evaluation distance. Learned predictors receive only RGB; speed reaches the
controller separately. B0 uses RGB and its fixed nominal camera configuration.
Every original frame is independently inferred before replaying the recorded
controls, event handling, stopping, cleanup and evaluation metrics. Execution
integrity and driving success are reported separately. Failed driving episodes
remain in the success-rate denominator, and unexecuted trials are reported
separately; a partial run never becomes a complete protocol success rate.
Every verified episode exports CPU-encoded video and a synchronized CSV with
predicted waypoints, steering and termination reason. The video uses nominal
ground-plane projection for viewing; original PNGs and traces are retained for
audit. Partial artifacts are preserved after an execution timeout.
The frozen `driving_success` field records reaching the route goal under the
declared termination policy, including its sustained-departure hold time.
`carla_tools.driving_quality` additionally reports clean completion: a completed
route with zero collision or lane-invasion events and a nonnegative minimum
vehicle-body lane margin. This diagnostic keeps the original results intact
and exposes route completions with a terminal lane crossing.
GPU execution belongs
in the existing Kaggle notebooks; local commands default to CPU.

`prepare_kaggle_refinement.py` and `prepare_kaggle_refinement_driving.py` prepare
local bundles for the existing training and CARLA notebooks. The driving bundle
uses an existing training output and a hash-pinned archived reference. The
candidate can also be retrieved from an explicit saved version, with an archive
hash, checkpoint hash and size check, to avoid circular latest-output sources.
Both preserve notebook IDs and titles and require a fresh read-only access
check before submission. They create no notebooks or datasets.
