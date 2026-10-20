# Development appearance training

The appearance experiment compares an unchanged incumbent, a continuation
control, and a candidate with additional photometric augmentation. The model
architecture and RGB preprocessing stay fixed. The experiment creates
development candidates; it does not promote a model or claim independent
generalization.

## Matched experiment

Both training arms start from the same complete incumbent state with fresh
optimizers and gradient scalers. They use identical training sample IDs,
epoch budgets, batch sizes, learning-rate schedules, seed policies, shuffle
orders, base photometric augmentation and horizontal mirror draws. The
candidate can add brightness, contrast, Gaussian blur and sensor noise through a
separate deterministic RNG stream, with inactive strengths set to zero. These additions change image appearance
without modifying pixel geometry or waypoint labels.

Choose the appearance strengths and development views using completed
development evidence, then freeze them before training. A private protocol
file can override the example defaults without rewriting the public config:

```powershell
python scripts/train_development_waypoints.py `
  --manifest <dataset_v1_manifest.json> `
  --validation-report <source_validation_report.json> `
  --initial-checkpoint <incumbent_refined_checkpoint.pt> `
  --protocol <frozen_development_protocol.json> `
  --output-dir <new_private_output_directory> `
  --device cpu
```

Use `--device cuda` only in an approved GPU environment. The command does not
silently switch devices or start a simulator. Dataset loaders require a
source-bound validation report and the existing physical-scene split; the
reserved final town and conditions stay unopened.

## Checkpoint selection and evaluation

A fixed stratified development subset selects epochs using the equal-weight
loss across predeclared appearance views. Views include clean RGB and optional
fixed dim, bright, low-contrast, blur or noise transformations. These views
are development diagnostics, not evidence of unseen weather performance.

The common representative seed minimizes the paired control/candidate subset
loss before full development scoring or native driving. All training seeds
are retained for comparison. The unchanged incumbent and every trained model
then predict the complete common development set with identical clean input
preprocessing. Prediction records preserve sample IDs, original RGB hashes,
targets, waypoints and per-horizon errors for independent reconstruction.

Primary checkpoints bind the protocol, source files, dataset manifest,
validation report, initial complete tensor state and exact sample selections.
The compatible refined-model export retains development provenance and
identical named tensor values. Runtime compatibility alone does not establish
closed-loop eligibility. Compare candidates with the same controller and
conditions, apply predeclared safety and latency bounds, and keep the
incumbent when the development evidence does not justify replacement.

## Existing Kaggle notebook preparation

`scripts/prepare_kaggle_development_training.py` creates a local submission
bundle for the existing private training notebook. It requires the reviewed
dataset manifest, original source replay proof, prior training archive and
exact incumbent checkpoint member. The existing CPU notebook output provides
the immutable dataset archive through a pinned mount reference. The bootstrap
verifies its complete SHA-256, reconstructs the exact manifest and retrieves
the version-pinned incumbent archive using platform authentication.

The generated bundle embeds reviewed source bytes and the chosen runtime
protocol. Before submitting, verify both existing notebook identities, access,
source versions and code hashes through fresh read-only checks. Preparation
does not create a new notebook, publish data, submit a job or store credentials.

The bootstrap pins the learning dependencies, verifies actual T4 execution,
forwards bounded epoch progress, validates all declared arm budgets, and
archives checkpoints, reports, prediction records and runtime sources. Owned
child processes stop on completion or failure. Store generated bundles,
weights and actual execution records outside public source control.
