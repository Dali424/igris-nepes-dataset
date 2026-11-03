#!/usr/bin/env python3
"""Revalidate preserved original episodes and merge an exact mixed-revision quota.

Use the dataset Python. Imports use their pinned ORIGINAL verifier and bundle in
isolated subprocesses; legacy evidence is never upgraded or rewritten. Existing
MP4 full-decode receipts remain binding through complete inventory hashes. Final
aggregation fully decodes videos again and preserves all recorded numeric values.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROFILES = {'left2': 2, 'right4': 4}
RECEIPT = 'meta/nepes_storage_verification.json'
SOURCES = 'meta/nepes_storage_sources.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable_write(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == payload, f'Existing audit/manifest differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as f:
            temporary = Path(f.name); f.write(payload); f.flush(); os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == payload, f'Existing audit/manifest differs: {path}')
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def no_overlap(output, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                f'Output overlaps protected source: {source}')


def unique_compatible(entries, expected_counts=None):
    require(bool(entries), 'No verified sources')
    groups = {p: [] for p in PROFILES}
    attempts, archives, evidence_ids = set(), set(), set()
    capture = entries[0]['evidence']['capture_schema']
    features = entries[0]['features']
    for entry in entries:
        e = entry['evidence']; profile = e['profile']
        require(e.get('verified') is True and profile in PROFILES and e['pair'] == PROFILES[profile], 'Unverified/profile-mismatched source')
        require(e['capture_schema'] == capture and entry['features'] == features, 'Incompatible capture/features across original revisions')
        require(e['attempt'] not in attempts and e['archive'] not in archives and entry['verification_sha256'] not in evidence_ids, 'Duplicate source episode/archive/evidence')
        attempts.add(e['attempt']); archives.add(e['archive']); evidence_ids.add(entry['verification_sha256'])
        for old in groups[profile]:
            for key in ('motion_exact_sha256', 'motion_quantized_sha256'):
                require(old[key] != e[key], f'Duplicate {profile} {key}')
            for column in ('observation.state', 'action'):
                for key in ('exact_sha256', 'quantized_sha256'):
                    require(old['columns'][column][key] != e['columns'][column][key], f'Duplicate {profile} {column} {key}')
            for key in ('startup_reset', 'initial_measurement'):
                require(old[key]['xy_signature_sha256'] != e[key]['xy_signature_sha256'], f'Duplicate measured {profile} {key} XY')
        groups[profile].append(e)
    counts = {p: len(v) for p, v in groups.items()}
    if expected_counts is not None:
        require(counts == expected_counts, f'Exact source quota mismatch: {counts} != {expected_counts}')
    return counts


def bundle_integrity(bundle, expected_manifest_sha):
    bundle = Path(bundle).resolve(strict=True)
    require(sha(bundle / 'manifest.json') == expected_manifest_sha, 'Source bundle manifest changed')
    for relative, item in read(bundle / 'manifest.json')['files'].items():
        p = bundle / relative
        require(p.resolve().is_relative_to(bundle) and not p.is_symlink() and sha(p) == item['sha256'], f'Source bundle file changed: {p}')


def lock_sources(stack, rows):
    paths = {Path(row['batch']).resolve().parent / '.batch.lock' for row in rows}
    paths |= {Path(row['attempt']) / 'collection/.compact_collection.lock' for row in rows}
    for path in sorted(paths):
        stream = stack.enter_context(path.open('r'))
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)


def validate_source_revision(verifier, plan, record, sample, evidence):
    """Use the pinned source policy, keeping legacy verification bytes intact."""
    program = '''import importlib.util,json,pathlib,sys
path=pathlib.Path(sys.argv[1])
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location("original_campaign_verifier",path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
value=json.load(sys.stdin)
print(json.dumps(module.validate_record_revision(value["plan"],value["record"],value["sample"])))
'''
    result = subprocess.run([sys.executable, '-B', '-c', program, str(verifier)],
                            input=json.dumps({'plan': plan, 'record': record, 'sample': sample}),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, f'Original source revision policy rejected record: {result.stderr[-6000:]}')
    lineage = json.loads(result.stdout)
    require(isinstance(lineage, dict) and lineage.get('sample_schema') == sample['schema'], 'Invalid original source lineage')
    if evidence['schema'] == 'nepes_seeded_attempt_verification_v1':
        require(sample['schema'] == 'nepes_seeded_robot_initial_xy_v1'
                and lineage.get('recipe_id') == 'baseline' and 'source_revision' not in evidence,
                'Legacy evidence must remain unmodified baseline v1')
    else:
        require(evidence.get('source_revision') == lineage, 'Source evidence revision differs from original policy')
    return lineage


def verify_one(row, destination, source_cache):
    attempt, archive, batch, bundle, verifier, original = [Path(row[k]).resolve(strict=True)
        for k in ('attempt', 'archive', 'batch', 'bundle', 'verifier', 'verification')]
    require(batch.name == 'batch.json' and attempt.is_relative_to(batch.parent), 'Attempt is outside declared batch')
    require(original == attempt / 'verification.json' and archive.is_relative_to(attempt / 'collection'), 'Source evidence/archive path mismatch')
    no_overlap(destination, [attempt, bundle, verifier.parent])
    state = read(batch); plan = state['plan']
    require(Path(plan['bundle']).resolve() == bundle, 'Source batch bundle mismatch')
    cache_key = (str(bundle), plan['bundle_manifest_sha256'])
    if cache_key not in source_cache:
        bundle_integrity(bundle, plan['bundle_manifest_sha256']); source_cache.add(cache_key)
    tools = plan['tools_sha256']
    require(verifier.name == 'merge_seeded.py' and verifier.name in tools, 'Unpinned original verifier')
    for name, checksum in tools.items():
        require(Path(name).name == name and sha(verifier.with_name(name)) == checksum, f'Source verifier/helper changed: {name}')
    matches = [r for r in state['attempts'] if Path(r['directory']).resolve() == attempt]
    require(len(matches) == 1 and matches[0].get('status') == 'accepted', 'Source is not an accepted original batch record')
    record = matches[0]
    for key in ('profile', 'pair', 'seed'):
        require(row[key] == record[key], f'Ledger/batch {key} mismatch')
    require(sha(original) == row['verification_sha256'] == record['verification_sha256'], 'Original evidence changed')
    require(sha(archive / RECEIPT) == row['archive_receipt_sha256'], 'Original archive receipt changed')
    sample = read(attempt / 'profile/sample.json')
    require(sample['schema'] == row['sample_schema'], 'Original sample schema changed')
    command = [sys.executable, '-B', str(verifier), 'verify-attempt', '--bundle', str(bundle),
               '--attempt', str(attempt), '--output', str(destination)]
    # Each source uses a separate interpreter to avoid old/new sampler module caching.
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, f'Original source verifier rejected {attempt}: {result.stderr[-6000:]}')
    evidence = read(destination)
    require(evidence == read(original) and sha(destination) == row['verification_sha256'], 'Revalidation did not reproduce original evidence bytes')
    require(evidence['attempt'] == str(attempt) and evidence['archive'] == str(archive), 'Verifier returned another source')
    require(record['index'] == record['seed'] - plan['first_seed'], 'Source record seed/index differs from its plan')
    lineage = validate_source_revision(verifier, plan, record, sample, evidence)
    for key in ('profile', 'pair', 'seed', 'archive_receipt_sha256'):
        require(evidence[key] == row[key], f'Revalidated {key} differs from ledger')
    for key in ('config_sha256', 'config_file_sha256', 'sample_sha256', 'source_config_sha256',
                'sampler_sha256', 'motion_exact_sha256', 'motion_quantized_sha256', 'frames', 'columns'):
        if key in row: require(row[key] == evidence[key], f'Ledger evidence field changed: {key}')
    extension = read(archive / 'meta/nepes_collection.json')
    require(extension['episodes'][0]['provenance'].get('kind') == 'original', 'Only original physical episodes may be imported')
    import pyarrow.parquet as pq
    for parquet in (archive / 'data').rglob('*.parquet'):
        require(set(pq.read_table(parquet, columns=['provenance.is_augmented']).column(0).to_pylist()) == {0}, 'An augmented episode cannot count as an original')
    features = read(archive / 'meta/info.json')['features']
    for key, value in [('capture_schema_sha256', evidence['capture_schema']), ('features_sha256', features)]:
        if key in row: require(row[key] == digest(value), f'Ledger {key} mismatch')
    require(sha(original) == row['verification_sha256'] and sha(archive / RECEIPT) == row['archive_receipt_sha256'], 'Original changed during verification')
    return {'origin': row.get('origin', 'imported'), 'source_id': row.get('source_id', digest(row)),
            'batch': str(batch), 'bundle': str(bundle), 'verifier': str(verifier),
            'verifier_sha256': sha(verifier), 'source_bundle_manifest_sha256': plan['bundle_manifest_sha256'],
            'sample_schema': sample['schema'], 'verification': str(original),
            'verification_sha256': row['verification_sha256'], 'validated_source_revision': lineage,
            'evidence': evidence, 'features': features}


def verify_still_bound(entries):
    for entry in entries:
        e = entry['evidence']; attempt = Path(e['attempt'])
        checks = {Path(entry['verification']): entry['verification_sha256'],
                  Path(entry['verifier']): entry['verifier_sha256'],
                  Path(entry['bundle']) / 'manifest.json': entry['source_bundle_manifest_sha256'],
                  attempt / 'profile/config.json': e['config_file_sha256'],
                  attempt / 'profile/sample.json': e['sample_sha256'],
                  Path(e['archive']) / RECEIPT: e['archive_receipt_sha256'],
                  Path(e['run']) / 'startup_trace.jsonl': e['startup_reset']['startup_trace_sha256'],
                  Path(e['initial_measurement']['source']): e['initial_measurement']['sha256']}
        for path, checksum in checks.items():
            require(sha(path) == checksum, f'Original source changed during campaign verification: {path}')


def load_imports(path):
    document = read(path)
    require(document.get('schema') == 'nepes_campaign_import_candidates_v1' and isinstance(document.get('imports'), list), 'Invalid import ledger')
    require(set(document['expected_counts']) == set(PROFILES)
            and all(type(n) is int and n >= 0 for n in document['expected_counts'].values()), 'Invalid import quotas')
    require(len(document['imports']) == sum(document['expected_counts'].values()), 'Import count mismatch')
    require(len({r['source_id'] for r in document['imports']}) == len(document['imports']), 'Duplicate ledger source ID')
    return document


def verify_imports(path, evidence_dir, cache):
    ledger = load_imports(path); result = []
    for i, row in enumerate(ledger['imports']):
        result.append(verify_one(row, evidence_dir / f'import_{i:06d}.json', cache))
        print(f'Import {i + 1}/{len(ledger["imports"])} verified: {row["profile"]} seed {row["seed"]}', flush=True)
    unique_compatible(result, ledger['expected_counts'])
    return result


def new_rows(batch, bundle):
    batch = Path(batch).resolve(strict=True)
    state = read(batch / 'batch.json'); plan = state['plan']
    require(state.get('status') == 'complete', 'New collection must be complete before mixed merge')
    require(Path(plan['bundle']).resolve() == bundle, 'New batch uses another bundle')
    quotas = plan.get('quota_by_profile', {p: plan.get('quota_each') for p in PROFILES})
    require(set(quotas) == set(PROFILES) and all(type(n) is int and n > 0 for n in quotas.values()), 'Invalid new-batch quotas')
    accepted = [r for profile in PROFILES for r in state['attempts']
                if r.get('status') == 'accepted' and r['profile'] == profile]
    require({p: sum(r['profile'] == p for r in accepted) for p in PROFILES} == quotas, 'New batch accepted quota mismatch')
    require(state['counts'] == quotas, 'New batch counts disagree with accepted records')
    rows = []
    for record in accepted:
        attempt = Path(record['directory']).resolve(strict=True); e = read(attempt / 'verification.json')
        require(attempt.is_relative_to(batch), 'New attempt escaped batch')
        rows.append({'origin': 'new', 'attempt': str(attempt), 'archive': e['archive'],
                     'batch': str(batch / 'batch.json'), 'bundle': str(bundle),
                     'verifier': str(bundle / 'scripts/seeded/merge_seeded.py'),
                     'verification': str(attempt / 'verification.json'),
                     'verification_sha256': record['verification_sha256'],
                     'archive_receipt_sha256': e['archive_receipt_sha256'],
                     'profile': record['profile'], 'pair': record['pair'], 'seed': record['seed'],
                     'sample_schema': read(attempt / 'profile/sample.json')['schema']})
    return state, rows, quotas


def compact_module(bundle):
    sys.path.insert(0, str(bundle / 'runtime/left2/source/robros_motion_planning'))
    from robros_motion_planning import compact_dataset
    compact_dataset._official_version()
    return compact_dataset


def preflight(args):
    ledger = load_imports(args.imports); rows = ledger['imports']
    protected = [args.bundle, args.imports] + [r[k] for r in rows for k in ('attempt', 'bundle')]
    no_overlap(args.output, protected)
    evidence_dir = args.output.with_suffix('.evidence')
    no_overlap(evidence_dir, protected)
    with ExitStack() as stack:
        lock_sources(stack, rows); before = sha(args.imports)
        entries = verify_imports(args.imports, evidence_dir, set())
        verify_still_bound(entries)
        require(sha(args.imports) == before, 'Ledger changed during verification')
        report = {'schema': 'nepes_campaign_import_verification_v1', 'verified': True,
                  'imports': str(args.imports.resolve()), 'imports_sha256': before,
                  'counts': unique_compatible(entries, ledger['expected_counts']), 'total_episodes': len(entries),
                  'source_specific_evidence_preserved_exactly': True, 'mp4_full_decode_receipts_inventory_bound': True,
                  'mp4_decoded_again_at_preflight': False, 'entries': entries}
        stable_write(args.output, report)
    return {'verified': True, 'episodes': len(entries), 'counts': report['counts'], 'audit': str(args.output)}


def merge(args):
    bundle = args.bundle.resolve(strict=True); output = args.output.resolve(); audit = args.audit.resolve()
    require(type(args.expected_per_pair) is int and args.expected_per_pair > 0, 'Expected per-pair quota must be positive')
    require(output.exists() == args.verify_existing, 'Existing dataset requires --verify-existing; no overwrite')
    ledger = load_imports(args.imports)
    with ExitStack() as stack:
        # --imports-only supports bounded real-source merge validation, without a synthetic new batch.
        batch = args.new_batch.resolve(strict=True) if args.new_batch is not None else None
        if batch is not None:
            stream = stack.enter_context((batch / '.batch.lock').open('r')); fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            state, rows, quotas = new_rows(batch, bundle)
        else:
            state, rows, quotas = None, [], {p: 0 for p in PROFILES}
        all_rows = ledger['imports'] + rows
        protected = [bundle, args.imports] + ([batch] if batch is not None else []) + [r[k] for r in all_rows for k in ('attempt', 'bundle')]
        for target in (output, audit, audit.with_suffix('.evidence'), output.with_name(output.name + '.campaign_sources.json')):
            no_overlap(target, protected)
        require(not audit.is_relative_to(output) and not output.is_relative_to(audit.with_suffix('.evidence')), 'Audit overlaps destination')
        lock_sources(stack, all_rows)
        ledger_sha = sha(args.imports)
        batch_sha = sha(batch / 'batch.json') if batch is not None else None
        evidence_dir = audit.with_suffix('.evidence'); cache = set()
        entries = verify_imports(args.imports, evidence_dir, cache)
        for i, row in enumerate(rows):
            entries.append(verify_one(row, evidence_dir / f'new_{i:06d}.json', cache))
            print(f'New source {i + 1}/{len(rows)} verified', flush=True)
        expected = {p: args.expected_per_pair for p in PROFILES}
        counts = unique_compatible(entries, expected)
        compact = compact_module(bundle)
        if state is not None:
            dataset = Path(state['dataset']).resolve(strict=True)
            require(dataset.is_relative_to(batch), 'New merged dataset escaped batch')
            saved = read(dataset / RECEIPT)
            require(saved['verified'] and saved['kind'] == 'merged_dataset' and saved['total_episodes'] == len(rows), 'New merge receipt/quota mismatch')
            require(compact._inventory(dataset) == saved['files'], 'New merged dataset changed')
            require(read(dataset / SOURCES)['source_shards'] == [{'path': r['archive'], 'receipt_sha256': r['archive_receipt_sha256']} for r in rows], 'New merged sources/order differ from batch')
            for i, row in enumerate(rows):
                require(saved['episode_payload_sha256'][str(i)] == read(Path(row['archive']) / RECEIPT)['episode_payload_sha256']['0'], 'New merged payload/source mismatch')
        mapping = []
        for i, entry in enumerate(entries):
            e = entry['evidence']; group = 'seeded_motion_' + e['motion_exact_sha256']
            mapping.append({'episode_index': i, 'profile': e['profile'], 'pair': e['pair'], 'seed': e['seed'],
                'origin': entry['origin'], 'source_id': entry['source_id'], 'source_batch': entry['batch'],
                'source_bundle': entry['bundle'], 'source_verifier': entry['verifier'], 'source_verifier_sha256': entry['verifier_sha256'],
                'source_archive': e['archive'], 'archive_receipt_sha256': e['archive_receipt_sha256'],
                'verification_schema': e['schema'], 'verification_sha256': entry['verification_sha256'],
                'sample_schema': entry['sample_schema'], 'sample_sha256': e['sample_sha256'],
                'config_sha256': e['config_sha256'], 'config_file_sha256': e['config_file_sha256'],
                'source_revision': deepcopy(entry['validated_source_revision']), 'xy_half_range_m': e['xy_half_range_m'],
                'offset_xy_m': e['sampled_offset_xy_m'], 'motion_exact_sha256': e['motion_exact_sha256'],
                'motion_quantized_sha256': e['motion_quantized_sha256'], 'split_group': group, 'motion_family': group,
                'measured_startup_pose_w': e['startup_reset']['measured_pose_w'],
                'measured_initial_world_poses': e['initial_measurement']['poses']})
        document = {'schema': 'nepes_mixed_original_campaign_v1', 'expected_per_pair': args.expected_per_pair,
                    'total_episodes': len(entries), 'counts': counts, 'import_counts': ledger['expected_counts'],
                    'new_counts': quotas, 'imports_sha256': ledger_sha, 'new_batch_sha256': batch_sha,
                    'episodes': mapping, 'legacy_evidence_not_upgraded': True,
                    'split_policy': 'Original episode and every augmentation sibling share seeded_motion_<parent motion SHA>.'}
        manifest = output.with_name(output.name + '.campaign_sources.json'); stable_write(manifest, document)
        shards = [Path(e['evidence']['archive']) for e in entries]
        if args.verify_existing:
            existing = read(output / RECEIPT); sources = read(output / SOURCES)
            require(existing['verified'] and existing['kind'] == 'merged_dataset', 'Unverified existing mixed dataset')
            require(all(existing.get('checks', {}).get(k) is True for k in
                        ('video_bytes_preserved_without_reencoding', 'episode_numeric_payloads_preserved')), 'Missing original merge preservation checks')
            require(sources['source_shards'] == [{'path': str(s), 'receipt_sha256': sha(s / RECEIPT)} for s in shards]
                    and sources['collection_manifest'] == {'path': str(manifest), 'sha256': sha(manifest)}, 'Existing mixed source lineage differs')
            require(read(output / 'meta/nepes_stationary_collection.json') == document, 'Existing mixed manifest changed')
            receipt = compact.verify_dataset(output, expected_episodes=len(entries))
            require(receipt['files'] == existing['files'] and receipt['episode_payload_sha256'] == existing['episode_payload_sha256'], 'Existing mixed inventory/payload changed')
        else:
            receipt = compact.merge_verified_shards(shards, output, repo_id='local/igris-c-mixed-originals', collection_manifest=manifest)
        require(receipt['verified'] and receipt['total_episodes'] == 2 * args.expected_per_pair, 'Mixed output quota/verification mismatch')
        for i, s in enumerate(shards):
            require(receipt['episode_payload_sha256'][str(i)] == read(s / RECEIPT)['episode_payload_sha256']['0'], 'Mixed merge changed recorded numeric payload')
        require(sha(args.imports) == ledger_sha and (batch is None or sha(batch / 'batch.json') == batch_sha), 'Source ledger/batch changed during merge')
        verify_still_bound(entries)
        report = {'schema': 'nepes_campaign_merge_verification_v1', 'verified': True, 'dataset': str(output),
                  'counts': counts, 'episodes': len(entries), 'frames': receipt['total_frames'],
                  'receipt_sha256': sha(output / RECEIPT), 'source_manifest': str(manifest),
                  'source_manifest_sha256': sha(manifest), 'numeric_payloads_preserved': True,
                  'legacy_evidence_not_upgraded': True, 'duplicate_state_action_initial_xy_checked': True}
        stable_write(audit, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('preflight-import'); p.add_argument('--bundle', type=Path, required=True); p.add_argument('--imports', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    m = commands.add_parser('merge')
    for key in ('bundle', 'imports', 'output', 'audit'): m.add_argument('--' + key, type=Path, required=True)
    source = m.add_mutually_exclusive_group(required=True)
    source.add_argument('--new-batch', type=Path); source.add_argument('--imports-only', action='store_true')
    m.add_argument('--expected-per-pair', type=int, default=100); m.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args(); print(json.dumps(preflight(args) if args.command == 'preflight-import' else merge(args), indent=2))


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f'Campaign source validation stopped: {exc}') from exc
