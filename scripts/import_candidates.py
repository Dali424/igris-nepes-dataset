#!/usr/bin/env python3
"""Build an import ledger (import_candidates.json) for scripts/campaign.py --imports.

Rows of existing ledgers (--ledger) are copied verbatim. Every ACCEPTED record of
each seeded batch (--batch; per profile, in record order) becomes a row bound to
that batch's own bundle, pinned verifier, verification evidence and archive receipt.
Nothing is re-verified or rewritten here: the campaign's verify_imports stage and
merge_base re-run every row's ORIGINAL verifier against its unchanged bundle.

A batch is read under a shared lock on its .batch.lock; a running collector holds
that lock exclusively, so stop the collection or campaign first. The ledger is
written once and never overwritten (a campaign pins its SHA-256). The printed summary
gives the per-profile counts, the remaining deficit for --base-per-pair, the largest
balanced --augmentation-episodes and a --first-seed that repeats no imported seed and
stays outside every imported batch's reserved seed range.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile

TOOLS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('nepes_import_campaign_sources', TOOLS / 'campaign_sources.py')
sources = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sources)
require, read, sha, digest = sources.require, sources.read, sources.sha, sources.digest
PROFILES = tuple(sources.PROFILES)
SCHEMA = 'nepes_campaign_import_candidates_v1'
# Optional row fields that campaign_sources.verify_one compares with the re-verified evidence.
EVIDENCE_FIELDS = ('config_sha256', 'config_file_sha256', 'sample_sha256', 'source_config_sha256', 'sampler_sha256',
                   'motion_exact_sha256', 'motion_quantized_sha256', 'frames', 'columns')


def budget_end(plan):
    """First seed after the batch's reserved range [first_seed, first_seed + per-profile budget)."""
    budgets = plan.get('max_attempts_by_profile') or {name: plan.get('max_attempts_each') for name in PROFILES}
    require(all(type(value) is int and value > 0 for value in budgets.values()), 'Batch plan has no finite attempt budget')
    return plan['first_seed'] + max(budgets.values())


def batch_rows(batch, profiles):
    """(rows, summary) for the accepted records of one seeded batch, read under its shared batch lock."""
    batch = Path(batch).expanduser().resolve(strict=True)
    with (batch / '.batch.lock').open('r') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'{batch} is in use by a running collector; stop its collection or campaign first') from exc
        return accepted_rows(batch, profiles)


def accepted_rows(batch, profiles):
    """batch_rows() without the lock: the caller guarantees that no collector writes the batch."""
    state = read(batch / 'batch.json'); plan = state['plan']
    require(state.get('schema') == 'nepes_seeded_batch_v1', f'Not a seeded batch: {batch}')
    bundle = Path(plan['bundle']).resolve(strict=True)
    verifier = bundle / 'scripts/seeded/merge_seeded.py'
    tools = plan.get('tools_sha256', {})
    require('merge_seeded.py' in tools and all(sha(verifier.with_name(name)) == checksum for name, checksum in tools.items()),
            f'The frozen tools of {batch} are not unchanged in its bundle {bundle}; add its rows by hand')
    rows = []
    for profile in PROFILES:
        if profile not in profiles:
            continue
        for record in state['attempts']:
            if record['profile'] != profile or record.get('status') != 'accepted':
                continue
            attempt = Path(record['directory']).resolve(strict=True)
            require(attempt.is_relative_to(batch), f'Attempt outside its batch: {attempt}')
            verification = attempt / 'verification.json'
            require(sha(verification) == record['verification_sha256'], f'Accepted evidence changed after acceptance: {verification}')
            evidence = read(verification)
            require((evidence['profile'], evidence['pair'], evidence['seed']) == (profile, record['pair'], record['seed'])
                    and record['index'] == record['seed'] - plan['first_seed'], f'Evidence identity differs from its record: {attempt}')
            archive = Path(evidence['archive']).resolve(strict=True)
            require(archive.is_relative_to(attempt / 'collection'), f'Archive outside its attempt: {archive}')
            row = {'attempt': str(attempt), 'archive': str(archive), 'batch': str(batch / 'batch.json'), 'bundle': str(bundle),
                   'verifier': str(verifier), 'verification': str(verification),
                   'verification_sha256': record['verification_sha256'],
                   'archive_receipt_sha256': evidence['archive_receipt_sha256'],
                   'profile': profile, 'pair': record['pair'], 'seed': record['seed'],
                   'sample_schema': read(attempt / 'profile/sample.json')['schema'],
                   'verification_schema': evidence['schema']}
            row.update({key: evidence[key] for key in EVIDENCE_FIELDS if key in evidence})
            row['capture_schema_sha256'] = digest(evidence['capture_schema'])
            row['features_sha256'] = digest(read(archive / 'meta/info.json')['features'])
            row['source_id'] = digest(row)
            rows.append(row)
    summary = {'batch': str(batch / 'batch.json'), 'batch_sha256': sha(batch / 'batch.json'), 'status': state.get('status'),
               'accepted': {name: sum(row['profile'] == name for row in rows) for name in PROFILES},
               'first_seed': plan['first_seed'], 'reserved_seed_end': budget_end(plan)}
    return rows, summary


def build(ledgers, batches, profiles, excluded=()):
    rows, inputs = [], {'ledgers': [], 'batches': [], 'excluded_attempts': sorted(map(str, excluded))}
    excluded = {Path(path).expanduser().resolve() for path in excluded}
    for path in ledgers:
        path = Path(path).expanduser().resolve(strict=True)
        document = sources.load_imports(path)
        kept = [row for row in document['imports'] if Path(row['attempt']).expanduser().resolve() not in excluded]
        rows.extend(kept)
        inputs['ledgers'].append({'path': str(path), 'sha256': sha(path), 'counts': document['expected_counts'],
                                  'excluded': len(document['imports']) - len(kept)})
    for path in batches:
        batch_rows_, summary = batch_rows(path, profiles)
        rows.extend(row for row in batch_rows_ if Path(row['attempt']) not in excluded)
        inputs['batches'].append(summary)
    found = {Path(row['attempt']).expanduser().resolve() for row in rows}
    require(not excluded & found, 'Internal error: an excluded attempt was kept')
    attempts = [str(Path(row['attempt']).resolve()) for row in rows]
    require(len(attempts) == len(set(attempts)), 'The same attempt appears twice; check the --ledger/--batch inputs')
    require(len({row['source_id'] for row in rows}) == len(rows), 'Duplicate ledger source ID')
    evidence = [row['verification_sha256'] for row in rows if row.get('verification_sha256')]
    require(len(evidence) == len(set(evidence)), 'Duplicate verification evidence')
    require(all(row['profile'] in PROFILES for row in rows), 'Unknown imported profile')
    counts = {name: sum(row['profile'] == name for row in rows) for name in PROFILES}
    return {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(), 'expected_counts': counts,
            'total_episodes': len(rows), 'imports': rows, 'generated_by': 'scripts/import_candidates.py', 'inputs': inputs}


def advice(document, base_per_pair):
    """Deficits, the largest balanced augmentation quota and a safe --first-seed for a new campaign."""
    counts = document['expected_counts']
    missing = {name: base_per_pair - counts[name] for name in PROFILES}
    seeds = {name: sorted(row['seed'] for row in document['imports'] if row['profile'] == name and type(row.get('seed')) is int)
             for name in PROFILES}
    first = max([max((seed for values in seeds.values() for seed in values), default=-1) + 1]
                + [batch['reserved_seed_end'] for batch in document['inputs']['batches']])
    result = {'imported': counts, 'base_per_pair': base_per_pair, 'new_base_by_profile': missing,
              'suggested_first_seed': first,
              'imported_seed_range_by_profile': {name: [values[0], values[-1]] if values else None for name, values in seeds.items()}}
    if all(value > 0 for value in missing.values()):
        result['max_augmentation_episodes'] = 2 * min(missing.values())
    else:
        result['problem'] = 'campaign.py needs a positive deficit for both profiles; lower the imports or raise --base-per-pair'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ledger', type=Path, action='append', default=[], help='Existing import ledger; rows are copied verbatim')
    parser.add_argument('--batch', type=Path, action='append', default=[], help='Seeded batch directory (contains batch.json)')
    parser.add_argument('--batch-profiles', nargs='+', choices=PROFILES, default=list(PROFILES),
                        help='Profiles imported from --batch inputs (default: both)')
    parser.add_argument('--exclude-attempt', type=Path, action='append', default=[],
                        help='Attempt directory to leave out (e.g. a source that will not survive a reboot)')
    parser.add_argument('--output', type=Path, required=True, help='New ledger file; never overwritten')
    parser.add_argument('--base-per-pair', type=int, default=100, help='Campaign target per profile, for the printed advice')
    args = parser.parse_args(argv)
    if not args.ledger and not args.batch:
        parser.error('Give at least one --ledger or --batch')
    output = args.output.expanduser().absolute()
    require(not output.exists(), f'{output} exists; a campaign pins its ledger, so write a new file')
    document = build(args.ledger, args.batch, set(args.batch_profiles), args.exclude_attempt)
    protected = {Path(row['batch']).resolve().parent for row in document['imports'] if 'batch' in row}
    protected |= {Path(row['attempt']).resolve() for row in document['imports']}
    require(not any(output.resolve().is_relative_to(path) for path in protected), 'Write the ledger outside the imported batches')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=output.parent, prefix='.' + output.name, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, output)  # fails instead of replacing a ledger created meanwhile
    finally:
        temporary.unlink()
    print(json.dumps({'ledger': str(output), 'sha256': sha(output), **advice(document, args.base_per_pair)}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f'Import ledger not written: {exc}') from exc
