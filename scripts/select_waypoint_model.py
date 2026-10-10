"""Apply the current absolute-budget selection policy to saved DEV reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import read_strict_json
from carla_tools.model_selection import DEFAULT_POLICY, load_selection_policy, select_waypoint_model


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','comparison','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--reference-training-report', type=Path)
    args = parser.parse_args(argv)
    reference = read_strict_json(args.reference)
    reference = reference.get('evaluation',reference)
    comparison = read_strict_json(args.comparison)
    if comparison.get('status') != 'PASS' or comparison.get('fixture_injection') is not False or comparison.get('final_town_used') is not False:
        raise ValueError('selection-invalid-comparison')
    candidates = comparison['evaluations']
    decision = select_waypoint_model(reference,candidates,policy=load_selection_policy(args.policy))
    root = (args.checkpoint_root or args.comparison.parent).resolve()
    model = None
    if decision['eligible_improvements']:
        name = decision['selected_candidate']
        if name not in ('hybrid_mirror','hybrid_geometry','hybrid','spatial'):
            raise ValueError('selection-unsupported-checkpoint-kind')
        checkpoint, training = root/name/'checkpoint.pt', root/name/'training_report.json'
        kind = 'refined' if name in ('hybrid_mirror','hybrid_geometry') else 'spatial'
    else:
        checkpoint, training, kind = args.reference_checkpoint,args.reference_training_report,'spatial'
    if checkpoint is not None:
        checkpoint = checkpoint.resolve(strict=True)
        if sha(checkpoint) != decision['selected_checkpoint_sha256']:
            raise ValueError('selection-checkpoint-bytes-mismatch')
        if training is None:
            raise ValueError('selection-training-binding-required')
        training = training.resolve(strict=True)
        binding = read_strict_json(training)
        if (binding.get('status') != 'PASS' or binding.get('variant') != decision['selected_candidate']
                or any(binding.get(k) != decision[k] for k in ('manifest_sha256','config_sha256'))
                or binding.get('checkpoint_sha256') != decision['selected_checkpoint_sha256']):
            raise ValueError('selection-training-binding-mismatch')
        model = {'kind':kind,'checkpoint_path':Path(os.path.relpath(checkpoint,args.output.resolve().parent)).as_posix(),
                 'checkpoint_sha256':sha(checkpoint),
                 'training_report_path':Path(os.path.relpath(training,args.output.resolve().parent)).as_posix(),
                 'training_report_sha256':sha(training)}
    report = {'status':'PASS','decision':decision,'model':model,
              'input_sha256':{'reference':sha(args.reference),'comparison':sha(args.comparison),'policy':sha(args.policy)},
              'local_gpu_used':False,'retrained':False}
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':'PASS','selected_candidate':decision['selected_candidate'],'decision':decision}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
