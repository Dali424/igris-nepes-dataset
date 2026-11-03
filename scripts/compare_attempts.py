#!/usr/bin/env python3
"""Compare the physics of two attempts of the same profile and seed for bit identity.

Each argument is an attempt_* directory (for example one collected alone and one
collected with --parallel 2). Compared: the sampled start, the outcome and its
verified evidence (state/action hashes, measured poses, failure reason/stage), the
episode result, and the startup and per-control-step physics traces line by line
after decompression (paths of the two attempts are normalised). Wall-clock files
(throughput.json, logs) are not compared. The Pink IK worker's process id (trace key
worker_pid) differs between any two runs and is always skipped; its request counter
(request_id) is compared. Attempts of two different package copies differ by
construction in config_sha256 (the config embeds the package path); pass
--ignore config_sha256 for such a comparison. Attempts solved by different IK backends
(the previous native-solver release vs this Pink release) are not bit-identical by
design (README, "Pink IK"); for such a pair only the checks section is meaningful.
Exit 0 when everything is identical, 1 when anything differs, 2 when the attempts are
not comparable.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

EPISODE = 'cycle_0000_crs_pickup_load_jig'
# Process ids differ between any two runs: the Pink IK worker's pid in every IK step of the trace.
ALWAYS_IGNORED = frozenset({'worker_pid'})


def load(path):
    return json.loads(Path(path).read_text())


def normalise(value, prefix):
    if isinstance(value, dict):
        return {key: normalise(item, prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [normalise(item, prefix) for item in value]
    return value.replace(prefix, '<ATTEMPT>') if isinstance(value, str) else value


def differences(a, b, path='', limit=10, ignore=frozenset()):
    """Up to limit (path, a, b) leaves that differ; keys named in ignore are skipped."""
    found = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if len(found) >= limit:
                break
            if key not in ignore:
                found += differences(a.get(key, '<missing>'), b.get(key, '<missing>'), f'{path}.{key}', limit - len(found), ignore)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            if len(found) >= limit:
                break
            found += differences(x, y, f'{path}[{index}]', limit - len(found), ignore)
    elif a != b or type(a) is not type(b):
        found.append([path or '.', a, b])
    return found


def lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return stream.read().splitlines()


def trace(attempt, relative):
    run = attempt / 'collection/runs/attempt_000000'
    for candidate in (run / relative, run / (relative + '.gz')):
        if candidate.is_file():
            return candidate
    return None


def compare_trace(a, b, relative, ignore=frozenset()):
    path_a, path_b = trace(a, relative), trace(b, relative)
    if path_a is None or path_b is None:
        return {'identical': path_a is None and path_b is None, 'missing': [str(p) for p, q in ((a, path_a), (b, path_b)) if q is None]}
    left, right = lines(path_a), lines(path_b)
    report = {'lines': [len(left), len(right)], 'identical': True}
    for index, (x, y) in enumerate(zip(left, right)):
        if x != y:
            parsed = differences(normalise(json.loads(x), str(a)), normalise(json.loads(y), str(b)), ignore=ignore)
            if parsed:
                report.update(identical=False, first_difference_line=index, differences=parsed)
                return report
    report['identical'] = len(left) == len(right)
    return report


def evidence(attempt):
    """(outcome, details) from the attempt's verification receipts, else its native report."""
    if (attempt / 'verification.json').is_file():
        e = load(attempt / 'verification.json')
        return 'success', {'motion_exact_sha256': e['motion_exact_sha256'], 'motion_quantized_sha256': e['motion_quantized_sha256'],
                           'frames': e.get('frames'), 'columns': e.get('columns'),
                           'measured_reset_pose_w': e['startup_reset']['measured_pose_w'],
                           'measured_initial_world_poses': e['initial_measurement']['poses']}
    if (attempt / 'failure_verification.json').is_file():
        e = load(attempt / 'failure_verification.json')
        return 'failure', {'reason': e.get('reason'), 'stage': e.get('stage')}
    trials = load(attempt / 'collection/collection.json').get('attempts', []) if (attempt / 'collection/collection.json').is_file() else []
    return 'unverified', {'trials': [{key: trial.get(key) for key in ('accepted', 'status', 'failure_reason', 'returncode')} for trial in trials]}


def compare(a, b, ignore=frozenset()):
    ignore = frozenset(ignore) | ALWAYS_IGNORED
    a, b = Path(a).resolve(strict=True), Path(b).resolve(strict=True)
    samples = [load(path / 'profile/sample.json') for path in (a, b)]
    identity = [{key: sample.get(key) for key in ('profile', 'pair', 'seed')} for sample in samples]
    if identity[0] != identity[1]:
        return 2, {'comparable': False, 'identity': identity}
    report = {'comparable': True, 'identity': identity[0], 'attempts': [str(a), str(b)], 'ignored_keys': sorted(ignore)}
    checks = {'sampled_offset_xy_m': [sample.get('applied_offset_xy_m') for sample in samples]}
    (kind_a, details_a), (kind_b, details_b) = evidence(a), evidence(b)
    checks['outcome'] = [kind_a, kind_b]
    for key in sorted(set(details_a) | set(details_b)):
        checks['evidence.' + key] = [details_a.get(key), details_b.get(key)]
    results = [a / 'collection/runs/attempt_000000' / EPISODE / 'result.json', b / 'collection/runs/attempt_000000' / EPISODE / 'result.json']
    if all(path.is_file() for path in results):
        values = [load(path) for path in results]
        for key in ('success', 'last_stage', 'reason', 'recorded_frame_count', 'recorded_duration_s', 'duration_s'):
            checks['result.' + key] = [value.get(key) for value in values]
    report['checks'] = {key: {'identical': pair[0] == pair[1], 'values': pair if pair[0] != pair[1] else pair[0]}
                        for key, pair in checks.items()}
    report['traces'] = {name: compare_trace(a, b, name, ignore) for name in ('startup_trace.jsonl', f'{EPISODE}/trace.jsonl')}
    report['bit_identical'] = (all(item['identical'] for item in report['checks'].values())
                               and all(item['identical'] for item in report['traces'].values()))
    return (0 if report['bit_identical'] else 1), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('attempt_a', type=Path)
    parser.add_argument('attempt_b', type=Path)
    parser.add_argument('--ignore', action='append', default=[], metavar='KEY',
                        help='Trace key to skip wherever it occurs, e.g. config_sha256 across two package copies '
                             '(worker_pid is always skipped)')
    args = parser.parse_args(argv)
    code, report = compare(args.attempt_a, args.attempt_b, frozenset(args.ignore))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f'Comparison failed: {exc}') from exc
