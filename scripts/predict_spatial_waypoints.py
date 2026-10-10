"""Predict actor-coordinate waypoints from one native RGB PNG and a strict checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--training-source-manifest",type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    from carla_tools.inference import read_rgb_png, write_prediction_json
    from carla_tools.spatial_inference import SpatialCheckpointPredictor
    from carla_tools.contracts import read_strict_json
    source = None if args.training_source_manifest is None else read_strict_json(args.training_source_manifest)["source_sha256"]
    predictor = SpatialCheckpointPredictor(args.checkpoint, device=args.device,
                                           expected_checkpoint_sha256=args.expected_checkpoint_sha256,source_sha256=source)
    rgb, image = read_rgb_png(args.image)
    result = predictor.predict_rgb(rgb)
    write_prediction_json(args.output, dict(result, image=image, model=predictor.identity))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
