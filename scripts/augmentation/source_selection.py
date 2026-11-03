"""Select complete, verified parent episodes without altering source data."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

from render_scene import read, require, sha, write

RECEIPT = Path('meta/nepes_storage_verification.json')
PROFILES = ('left2', 'right4')


def seeded_module(name, bundle=None):
    scripts = Path(bundle).resolve(strict=True) / 'scripts' if bundle is not None else Path(__file__).resolve().parents[1]
    path = scripts / 'seeded' / (name + '.py')
    spec = importlib.util.spec_from_file_location('augmentation_seeded_' + name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def source_quotas(plan):
    require(('quota_each' in plan) != ('quota_by_profile' in plan), 'Ambiguous/missing source quotas')
    quotas = ({name: plan['quota_each'] for name in PROFILES} if 'quota_each' in plan else dict(plan['quota_by_profile']))
    require(set(quotas) == set(PROFILES) and all(type(v) is int and v > 0 for v in quotas.values()), 'Invalid source quotas')
    return quotas


def source_quota_arguments(plan):
    quotas = source_quotas(plan)
    if 'quota_each' in plan:
        return ['--expected-per-pair', str(plan['quota_each'])]
    return ['--expected-left', str(quotas['left2']), '--expected-right', str(quotas['right4'])]


def complete_source(batch, bundle, episodes=None):
    batch, bundle = Path(batch).resolve(strict=True), Path(bundle).resolve(strict=True)
    state = read(batch / 'batch.json'); plan = state['plan']
    require(state.get('schema') == 'nepes_seeded_batch_v1' and state.get('status') == 'complete',
            'Finish and verify source collection before augmentation')
    require(Path(plan['bundle']).resolve() == bundle and Path(plan['output']).resolve() == batch,
            'Source batch location/bundle differs; audit paths are not silently relocated')
    quotas = source_quotas(plan)
    total = sum(quotas.values())
    groups = {name: [r for r in state['attempts'] if r['profile'] == name and r['status'] == 'accepted'] for name in PROFILES}
    require(all(len(groups[name]) == quotas[name] for name in PROFILES), 'Completed source does not have its exact quota')
    require(state.get('counts') == quotas, 'Source count ledger differs')
    count = total if episodes is None else episodes
    require(type(count) is int and 1 <= count <= total, '--episodes must be between1 and source total')
    dataset = Path(state['dataset']).resolve(strict=True)
    require(dataset.is_relative_to(batch), 'Source dataset escaped batch')
    receipt = read(dataset / RECEIPT)
    require(receipt.get('verified') is True and receipt.get('kind') == 'merged_dataset'
            and receipt.get('total_episodes') == total, 'Source merge is not verified/complete')
    manifest = read(dataset / 'meta/nepes_stationary_collection.json')
    require(manifest.get('schema') == 'nepes_seeded_collection_aggregate_v1'
            and manifest.get('total_episodes') == total
            and (manifest.get('expected_by_profile') == quotas if 'quota_by_profile' in plan
                 else manifest.get('expected_per_pair') == plan['quota_each']), 'Wrong completed source manifest')
    require(sha(bundle / 'manifest.json') == plan['bundle_manifest_sha256'], 'Source bundle changed')
    mapping = {r['verification_sha256']: r for r in manifest['episodes']}
    require(len(mapping) == total, 'Source episode mappings are not unique')
    selected = []
    for local in range(max(quotas.values())):
        for profile in PROFILES:
            if local >= quotas[profile]:
                continue
            record = groups[profile][local]; attempt = Path(record['directory']).resolve(strict=True)
            require(attempt.is_relative_to(batch), 'Source attempt escaped batch')
            evidence_path = attempt / 'verification.json'
            require(sha(evidence_path) == record['verification_sha256'], 'Source verification changed')
            evidence = read(evidence_path); entry = mapping[record['verification_sha256']]
            require(evidence['profile'] == profile and evidence['seed'] == record['seed'] == entry['seed'],
                    'Source record/evidence identity differs')
            archive = Path(evidence['archive']).resolve(strict=True)
            require(archive.is_relative_to(attempt) and sha(archive / RECEIPT) == evidence['archive_receipt_sha256'],
                    'Source archive binding differs')
            selected.append({'index': len(selected), 'profile': profile, 'seed': record['seed'],
                             'parent_attempt': str(attempt), 'parent_verification': str(evidence_path),
                             'parent_verification_sha256': sha(evidence_path),
                             'parent_config_sha256': sha(attempt / 'profile/config.json'),
                             'parent_archive': str(archive), 'parent_archive_receipt_sha256': sha(archive / RECEIPT),
                             'source_episode': deepcopy(entry)})
    selected = selected[:count]
    return {'ready': True, 'parent_batch': str(batch), 'parent_batch_sha256': sha(batch / 'batch.json'),
            'parent_merged_dataset': str(dataset), 'parent_merge_receipt_sha256': sha(dataset / RECEIPT),
            'source_total_episodes': total,
            **({'source_quota_by_profile': quotas} if 'quota_by_profile' in plan else {}), 'selected_episodes': count,
            'expected_profile_counts': {name: sum(item['profile'] == name for item in selected) for name in PROFILES}, 'parents': selected}


def selected_video_source(bundle, source, output):
    """A small requested subset is independently merged and verified first."""
    if source['selected_episodes'] == source['source_total_episodes'] and 'source_quota_by_profile' not in source:
        return Path(source['parent_merged_dataset'])
    sys.path.insert(0, str(Path(bundle) / 'runtime/left2/source/robros_motion_planning'))
    from robros_motion_planning import compact_dataset as compact
    output = Path(output).resolve()
    roots = [Path(row['parent_archive']) for row in source['parents']]
    mapping = []
    for row in source['parents']:
        item = deepcopy(row['source_episode'])
        item.update(parent_episode_index=item['episode_index'], episode_index=len(mapping),
                    split_group='seeded_motion_' + item['motion_exact_sha256'])
        mapping.append(item)
    document = {'schema': 'nepes_augmentation_parent_selection_v1',
                'selection_from_verified_complete_batch': True,
                'parent_batch': source['parent_batch'], 'parent_batch_sha256': source['parent_batch_sha256'],
                'parent_merged_dataset': source['parent_merged_dataset'],
                'parent_merge_receipt_sha256': source['parent_merge_receipt_sha256'],
                'total_episodes': len(mapping), 'profile_counts': source['expected_profile_counts'],
                'episodes': mapping}
    manifest = output.with_name(output.name + '.sources.json')
    if manifest.exists():
        require(read(manifest) == document, 'Existing selected-source lineage differs')
    else:
        write(manifest, document)
    if output.exists():
        sources = read(output / compact.SOURCES_PATH)
        require(sources['source_shards'] == [{'path': str(root), 'receipt_sha256': sha(root / compact.RECEIPT_PATH)} for root in roots],
                'Existing selected source parents changed')
        require(sources['collection_manifest'] == {'path': str(manifest), 'sha256': sha(manifest)}, 'Selected source manifest changed')
        receipt = compact.verify_dataset(output, expected_episodes=len(roots), video_workers=1)
    else:
        receipt = compact.merge_verified_shards(roots, output, repo_id='local/igris-selected-parents', collection_manifest=manifest)
    require(receipt['verified'] is True and receipt['total_episodes'] == len(roots), 'Selected source merge failed verification')
    for index, root in enumerate(roots):
        require(receipt['episode_payload_sha256'][str(index)] == read(root / compact.RECEIPT_PATH)['episode_payload_sha256']['0'],
                'Selection changed parent numerical payload')
    return output
