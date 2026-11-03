#!/usr/bin/env python3
"""Verify and aggregate an exact requested count of native render derivatives."""
import argparse
from collections import Counter
from pathlib import Path

from render_scene import read, require, sha
from render_verify import compact_module

def merge(bundle, manifest, output):
    compact = compact_module(bundle)
    manifest, output = Path(manifest).resolve(strict=True), Path(output).resolve()
    document = read(manifest)
    require(document['schema'] == 'nepes_native_render_sources_v1', 'Wrong render source manifest')
    rows = document['episodes']
    count = document['expected_episodes']
    require(type(count) is int and count > 0 and len(rows) == count
            and {name: sum(row['profile'] == name for row in rows) for name in ('left2', 'right4')} == document['expected_profile_counts'], 'Requested render quota differs')
    require([row['index'] for row in rows] == list(range(count)), 'Source indices differ')
    require(all(row['status'] == 'accepted' for row in rows), 'Unaccepted source included')
    require(len({row['parent_verification_sha256'] for row in rows}) == count, 'Duplicate parent condition')
    roots, receipts, groups = [], [], []
    for row in rows:
        root = Path(row['derivative']).resolve(strict=True)
        require(not output.is_relative_to(root) and not root.is_relative_to(output), 'Merge output overlaps source')
        require(sha(root / compact.RECEIPT_PATH) == row['receipt_sha256'], 'Derivative receipt changed')
        receipt = read(root / compact.RECEIPT_PATH)
        require(receipt['verified'] and receipt['kind'] == 'single_episode_shard', 'Unverified render shard')
        require(compact._inventory(root) == receipt['files'], 'Render shard files changed')
        audit = read(root / 'meta/render_augmentation.json')
        require(audit['verified'] and audit['physical_success_revalidated'] and audit['new_physics_execution'], 'Missing native physics gate')
        require(audit['parent_verification_sha256'] == row['parent_verification_sha256'] and audit['case'] == row['case'], 'Wrong parent/visual lineage')
        require(audit['parent_actions_states_timestamps_reused'] is False, 'Child falsely reuses parent numerical labels')
        require(audit['split_group'] == 'seeded_motion_' + audit['parent_motion_exact_sha256'], 'Missing sibling split group')
        roots.append(root); receipts.append(receipt); groups.append(audit['split_group'])
    require(len(set(roots)) == count and len(set(groups)) == count, 'Duplicate render sources/parent trajectories')
    if output.exists():
        sources = read(output / compact.SOURCES_PATH)
        require(sources['source_shards'] == [{'path': str(root), 'receipt_sha256': sha(root / compact.RECEIPT_PATH)} for root in roots], 'Existing merge sources differ')
        require(sources['collection_manifest'] == {'path': str(manifest), 'sha256': sha(manifest)}, 'Existing merge manifest differs')
        result = compact.verify_dataset(output, expected_episodes=count, video_workers=1)
    else:
        result = compact.merge_verified_shards(roots, output, repo_id='local/igris-c-native-render', collection_manifest=manifest)
    require(result['verified'] and result['kind'] == 'merged_dataset' and result['total_episodes'] == count, 'Wrong merged output')
    for index, receipt in enumerate(receipts):
        require(result['episode_payload_sha256'][str(index)] == receipt['episode_payload_sha256']['0'], 'Merge changed child numeric payload')
    return {'status': 'complete', 'dataset': str(output), 'episodes': count, 'frames': result['total_frames'], 'receipt_sha256': sha(output / compact.RECEIPT_PATH)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(merge(args.bundle, args.manifest, args.output))
