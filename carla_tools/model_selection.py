"""Select by overall DEV error subject to absolute geometry-case budgets.

Selection is separate from immutable training/checkpoint source contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .contracts import read_strict_json

DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'configs/model_selection.json'
CASES = ('all', 'left_curve', 'right_curve', 'straight')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('selection-invalid-numeric-metric')
    return value


def _validated_policy(value):
    if (set(value) != {'schema_version','policy_version','primary_metric','near_mean_max_m','far_mean_max_m','forward_p95_max_ms'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1 or value['policy_version'] != 'mean-and-absolute-case-bounds-v1'
            or value['primary_metric'] != 'mean_euclidean_m'):
        raise ValueError('selection-invalid-policy')
    for key in ('near_mean_max_m','far_mean_max_m','forward_p95_max_ms'):
        if _number(value[key]) <= 0:
            raise ValueError('selection-invalid-budget')
    return value


def load_selection_policy(path=DEFAULT_POLICY):
    return _validated_policy(read_strict_json(Path(path)))


def _groups(report):
    if (report.get('status') != 'PASS' or report.get('split') != 'dev'
            or report.get('fixture_injection') is not False or report.get('final_town_used') is not False
            or type(report.get('sample_count')) is not int or report['sample_count'] <= 0
            or any(not re.fullmatch('[0-9a-f]{64}', str(report.get(key, ''))) for key in
                   ('manifest_sha256','config_sha256','checkpoint_sha256'))):
        raise ValueError('selection-invalid-DEV-protocol')
    groups = {}
    for group in report['metrics']:
        if group['condition'] != 'all':
            continue
        if group['case'] in groups:
            raise ValueError('selection-duplicate-case')
        groups[group['case']] = group
    if set(groups) != set(CASES):
        raise ValueError('selection-required-case-missing')
    for group in groups.values():
        count, predicted, failed = (group[k] for k in ('sample_count','predicted_count','failed_prediction_count'))
        if (any(type(v) is not int for v in (count,predicted,failed)) or count <= 0
                or min(predicted,failed) < 0 or predicted + failed != count
                or not math.isclose(_number(group['coverage']), predicted/count, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('selection-invalid-coverage-counts')
        for key in ('near_mean_m','far_mean_m','far_p95_m','mean_euclidean_m'):
            _number(group[key])
        # The common contract has two near and three far waypoints.
        if not math.isclose(group['mean_euclidean_m'], (2*group['near_mean_m']+3*group['far_mean_m'])/5,
                            rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('selection-inconsistent-overall-mean')
    if (groups['all']['sample_count'] != report['sample_count']
            or sum(groups[case]['sample_count'] for case in CASES[1:]) != report['sample_count']
            or sum(groups[case]['predicted_count'] for case in CASES[1:]) != groups['all']['predicted_count']):
        raise ValueError('selection-inconsistent-group-counts')
    total = groups['all']['predicted_count']
    if total:
        for metric in ('near_mean_m','far_mean_m','mean_euclidean_m'):
            weighted = math.fsum(groups[case][metric]*groups[case]['predicted_count'] for case in CASES[1:])/total
            if not math.isclose(groups['all'][metric],weighted,rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('selection-inconsistent-case-weighted-mean')
    _number(report['latency_ms']['p95'])
    return groups


def select_waypoint_model(reference, candidates, *, policy=None):
    policy = load_selection_policy() if policy is None else _validated_policy(dict(policy))
    base = _groups(reference)
    signature = tuple(reference[k] for k in ('manifest_sha256','config_sha256','sample_count'))
    eligible, reasons, scores = [], {}, {}
    for name, report in candidates.items():
        groups = _groups(report)
        if (report.get('variant') != name or tuple(report[k] for k in ('manifest_sha256','config_sha256','sample_count')) != signature
                or any(groups[c]['sample_count'] != base[c]['sample_count'] for c in CASES)):
            raise ValueError('selection-dataset-or-case-count-mismatch')
        failed = []
        for case in CASES:
            if groups[case]['coverage'] != 1:
                failed.append(case + ':coverage')
            for metric, bound in (('near_mean_m','near_mean_max_m'),('far_mean_m','far_mean_max_m')):
                if groups[case][metric] > policy[bound]:
                    failed.append(case + ':' + metric)
        score = groups['all']['mean_euclidean_m']
        if score >= base['all']['mean_euclidean_m']:
            failed.append('overall-mean-improvement')
        if report['latency_ms']['p95'] > policy['forward_p95_max_ms']:
            failed.append('forward-latency')
        scores[name], reasons[name] = score, failed
        if not failed:
            eligible.append(name)
    selected = min(eligible, key=lambda name:(scores[name], name)) if eligible else reference['variant']
    chosen = candidates[selected] if eligible else reference
    return {'schema_version':1, 'selected_candidate':selected, 'selected_checkpoint_sha256':chosen['checkpoint_sha256'],
            'eligible_improvements':eligible, 'failed_criteria':reasons,
            'primary_metric':'mean_euclidean_m', 'selected_mean_m':_groups(chosen)['all']['mean_euclidean_m'],
            'reference_mean_m':base['all']['mean_euclidean_m'], 'manifest_sha256':signature[0], 'config_sha256':signature[1],
            'policy':policy, 'policy_sha256':hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'selection_scope':'DEV-offline-model-selection', 'closed_loop_verified':False, 'final_town_used':False}
