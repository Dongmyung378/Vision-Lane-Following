"""Use the locally selected project checkpoint for native RGB inference."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,default=ROOT/'results/local/selected-waypoint-model.json')
    for name in ('image','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--near-correction', action='store_true', help='Explicit development RGB near-path correction for the selected geometric hybrid')
    args = parser.parse_args(argv)
    from carla_tools.contracts import read_strict_json
    from carla_tools.inference import _input_file, _sha256_file, read_rgb_png, write_prediction_json
    from carla_tools.refinement_driving import NativeCheckpointPredictor
    selection_path = _input_file(args.selection)
    selection = read_strict_json(selection_path)
    decision, binding = selection['decision'],selection['model']
    if selection['status'] != 'PASS' or not binding or decision['selected_checkpoint_sha256'] != binding['checkpoint_sha256']:
        raise ValueError('selected-inference-binding-mismatch')
    training = selection_path.parent / binding['training_report_path']
    checkpoint = selection_path.parent / binding['checkpoint_path']
    if _sha256_file(_input_file(training)) != binding['training_report_sha256']:
        raise ValueError('selected-inference-binding-mismatch')
    predictor = NativeCheckpointPredictor(checkpoint,kind=binding['kind'],
        expected_sha256=binding['checkpoint_sha256'],device=args.device,training_report=training)
    if predictor.identity['variant'] != decision['selected_candidate'] or predictor.identity['manifest_sha256'] != decision['manifest_sha256']:
        raise ValueError('selected-inference-model-or-dataset-mismatch')
    if args.near_correction:
        if predictor.identity['variant'] != 'hybrid_geometry':
            raise ValueError('selected-near-correction-requires-geometric-hybrid')
        from carla_tools.near_correction import NearCorrectedPredictor
        predictor = NearCorrectedPredictor(predictor)
    rgb,image = read_rgb_png(args.image)
    result = predictor.predict(rgb)
    write_prediction_json(args.output,dict(result,image=image,model=predictor.identity,selection=decision))
    return 0 if result['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
