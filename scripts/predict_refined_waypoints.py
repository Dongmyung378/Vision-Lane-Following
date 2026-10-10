"""Predict from a native RGB PNG with an explicitly hash-bound refined checkpoint."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('image', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    from carla_tools.inference import read_rgb_png, write_prediction_json
    from carla_tools.refinement_driving import NativeCheckpointPredictor
    predictor = NativeCheckpointPredictor(args.checkpoint, kind='refined',
        expected_sha256=args.expected_checkpoint_sha256, device=args.device)
    rgb, image = read_rgb_png(args.image)
    result = predictor.predict(rgb)
    write_prediction_json(args.output, dict(result, image=image, model=predictor.identity))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
