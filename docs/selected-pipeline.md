# Use a selected RGB and control pipeline

A pipeline pointer binds the selected geometric hybrid checkpoint, learned
preprocessing, RGB path fusion and controller configuration. Load
them together when reproducing a selected driving pipeline. The older waypoint
inference command continues to serve waypoint predictions alone.

The pipeline pointer is the execution selection for this adapter. Older
model-only inference commands use their own selection artifact; they do not
read a pipeline pointer or adopt its controller automatically. Keep historical
selection evidence unchanged and pass the current pipeline explicitly with
`--selection` when reproducing the complete selected configuration.

The pointer uses a relative checkpoint path within its directory. It records
the checkpoint and model contract hashes, learning configuration, correction
configuration, controller configuration and an explicit runtime source map.
Changing any runtime binding requires a new pipeline identity. Optional
provenance records the selection evidence; loading checks the runtime bindings.

The correction binding can select the original `path` correction or
`guarded_path` fusion. Guarded fusion compares the current learned near points
with the RGB correction proposal. A proposal outside its frozen displacement
budget leaves the current learned path in control. Both proposals and the guard
decision remain in diagnostics. This check uses the current RGB prediction and
fixed nominal camera assumptions; it receives no route, pose, measured camera
calibration or previous frame.

## Predict one frame and compute control

Supply an existing pointer, an original 640 × 360 RGB8 PNG and native vehicle
measurement evidence. The geometry input can be a captured `episode.json` or
a JSON object containing its complete `vehicle_geometry` and
`geometry_evidence` fields. The command reconstructs the axle positions,
wheelbase, front track and steering normalization from the measured wheels and
their measurement transform. Geometry and speed are controller inputs; they
are never passed to RGB perception.

```powershell
python scripts/predict_selected_pipeline.py `
  --selection pipeline.json `
  --geometry native_episode.json `
  --image rgb.png `
  --speed-mps 5.0 `
  --output prediction_and_control.json
```

CPU is the default device. The output contains raw and corrected waypoints,
the selected controller's command and diagnostics, source binding checks and
processing timings. A single image starts with fresh controller state. This
command computes control offline; it does not start CARLA or apply commands.

## Preserve controller history over a sequence

Use JSONL with one image and speed per frame. Image paths are relative to the
JSONL file's directory. Consecutive frames must advance by one frame and
0.05 seconds. An explicit `reset_before` starts a new controller episode and
permits a discontinuous frame or timestamp.

```jsonl
{"frame":100,"timestamp":5.0,"image":"images/00000100.png","speed_mps":5.0}
{"frame":101,"timestamp":5.05,"image":"images/00000101.png","speed_mps":5.1}
{"frame":200,"timestamp":10.0,"image":"images/00000200.png","speed_mps":0.0,"reset_before":true}
```

```powershell
python scripts/predict_selected_pipeline.py `
  --selection pipeline.json `
  --geometry native_episode.json `
  --frames frames.jsonl `
  --output sequence_prediction_and_control.json
```

The controller retains its PI integral, smoothing and steering rate history
within each episode. Perception uses only the current RGB frame. Processing
stops on the first prediction or typed controller rejection and retains that
failure, its original path and timing. The Python adapter requires an explicit
reset before another frame following a failure. It does not reuse the last path
or switch to a different controller.

## Python adapter

```python
from carla_tools.selected_pipeline import load_selected_pipeline

pipeline = load_selected_pipeline("pipeline.json", measured_vehicle_geometry)
result = pipeline.step(rgb, speed_mps=5.0)
pipeline.verify_bindings()  # Recheck weights, pointer and sources after a stream.
pipeline.reset()            # Start a separate controller episode.
```

`measured_vehicle_geometry` is a validated `VehicleGeometry` or its complete
dictionary, including both axle y coordinates. The adapter never inserts
defaults for missing measurements. The CLI additionally reconstructs native
measurement evidence before creating the adapter.

After reviewing selection evidence, create a pointer with
`build_pipeline_selection(checkpoint, controller_config, base_directory=...)`.
Pass `preprocessing_config=guarded_preprocessing_config()` from
`carla_tools.guarded_path_correction` to bind guarded fusion; omitting that
argument preserves the original path correction contract.
The builder checks the real checkpoint on CPU and returns a dictionary; it
does not publish a pointer. Write it within `base_directory` so the relative
checkpoint binding remains valid. Store personal experiment evidence and
generated predictions outside tracked project documentation.

The reported processing interval begins with learned RGB preprocessing and
ends after control. It excludes BGRA conversion, simulator and sensor waiting,
logging and video encoding. It does not establish real-time closed-loop
performance.
