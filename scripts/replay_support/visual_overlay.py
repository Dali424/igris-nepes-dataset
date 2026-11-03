"""Restore the four audited visual overrides without resolving old USD paths."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

CASES = {
    'dim_matte': {'lighting_gain': .70, 'crs_roughness': .60},
    'bright_satin': {'lighting_gain': 1.30, 'crs_roughness': .35},
}
PROPERTIES = {
    '/World/Light.inputs:intensity': 350.,
    '/World/RectLight.inputs:intensity': 15000000.,
    '/World/Ring_02/Visual/_materials/Plastic_Dark_Charcoal/Principled_BSDF.inputs:roughness': .4300000071525574,
    '/World/Ring_04/Visual/_materials/Plastic_Dark_Charcoal/Principled_BSDF.inputs:roughness': .4300000071525574,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def authored_properties(path):
    """Parse only our tiny over/float USD dialect; never load a USD sublayer.

    Full USD composition is intentionally not performed on the retained file:
    its sublayer may point at the original server. Unsupported authored prims,
    connections, relations, references, time samples and metadata are rejected.
    The unchanged stage metadata comes from the digest-verified packaged scene.
    """
    text = Path(path).read_text()
    match = re.fullmatch(r'#usda 1\.0\s*\([\s\S]*?\)\s*(over\s+"World"[\s\S]*)', text)
    require(match is not None, 'Unsupported visual overlay USDA structure')
    body = match.group(1)
    token = re.compile(r'\s*("[A-Za-z_][A-Za-z0-9_]*"|[A-Za-z_][A-Za-z0-9_:]*|'
                       r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[{}=])')
    tokens, position = [], 0
    while position < len(body):
        if not body[position:].strip():
            break
        found = token.match(body, position)
        require(found is not None, 'Unsupported authored visual overlay syntax')
        tokens.append(found.group(1)); position = found.end()
    allowed_prims = set()
    for attribute in PROPERTIES:
        parts = attribute.split('.', 1)[0].strip('/').split('/')
        allowed_prims.update('/' + '/'.join(parts[:i]) for i in range(1, len(parts) + 1))
    values, index, seen_prims = {}, 0, set()

    def take(expected=None):
        nonlocal index
        require(index < len(tokens), 'Incomplete visual overlay')
        value = tokens[index]; index += 1
        require(expected is None or value == expected, 'Unexpected visual overlay token: ' + value)
        return value

    def prim(parent):
        take('over')
        name = take()
        require(name.startswith('"') and name.endswith('"'), 'Overlay prim name must be quoted')
        path = parent + '/' + name[1:-1]
        require(path in allowed_prims and path not in seen_prims, 'Unauthorized/duplicate overlay prim: ' + path)
        seen_prims.add(path)
        take('{')
        while index < len(tokens) and tokens[index] != '}':
            if tokens[index] == 'over':
                prim(path)
            else:
                take('float'); attribute = path + '.' + take(); take('=')
                value = float(take())
                require(math.isfinite(value) and attribute in PROPERTIES and attribute not in values,
                        'Unauthorized/duplicate visual override: ' + attribute)
                values[attribute] = float32(value)
        take('}')

    prim('')
    require(index == len(tokens) and set(values) == set(PROPERTIES), 'Wrong authored visual property set')
    return values


def load_visual_overlay(attempt, run, run_config, baseline_scene):
    """Bind retained profile/visual.json, config and overlay to the native run."""
    attempt, run = Path(attempt).resolve(), Path(run).resolve()
    candidates = {attempt / 'profile/visual.json', attempt / 'source_overlay/visual.json',
                  run / 'source_overlay/visual.json'}
    candidates = [path for path in candidates if path.is_file()]
    require(len(candidates) == 1,
            'Native visual-overlay replay needs its retained profile/{visual.json,config.json,scene_visual_overlay.usda}; '
            'supply the whole render trial, or copy that profile into the run as source_overlay/')
    visual_path = candidates[0].resolve(strict=True)
    profile_config = visual_path.with_name('config.json').resolve(strict=True)
    overlay_path = visual_path.with_name('scene_visual_overlay.usda').resolve(strict=True)
    for path in (visual_path, profile_config, overlay_path):
        require(path.is_relative_to(attempt), 'Visual replay input escaped the supplied attempt')
    visual = read(visual_path)
    require(visual.get('schema') == 'nepes_native_render_overlay_v1', 'Unsupported visual overlay receipt')
    case = visual.get('case')
    require(case in CASES and visual.get('recipe') == CASES[case], 'Visual overlay recipe changed')
    require(digest(baseline_scene) == visual.get('source_scene_sha256'), 'Packaged base scene differs from the visual parent')
    require(digest(profile_config) == visual.get('config_sha256'), 'Retained visual config checksum differs')
    require(read(profile_config) == run_config, 'Retained visual config differs from the native run config')
    scene = run_config['single_arm_layout']
    require(scene['scene_usd'] == visual.get('overlay'), 'Native scene path differs from its visual receipt')
    require(digest(overlay_path) == scene['scene_sha256'] == visual.get('overlay_sha256'),
            'Retained visual overlay checksum differs')
    for key in ('all_physics_geometry_poses_and_cameras_unchanged',
                'only_visual_allowlisted_properties_authored', 'stage_metadata_unchanged'):
        require(visual.get(key) is True, 'Incomplete visual isolation evidence: ' + key)
    changes = visual.get('changes', [])
    require(len(changes) == len(PROPERTIES) and {row['attribute'] for row in changes} == set(PROPERTIES),
            'Visual receipt has unexpected properties')
    require(len(visual.get('composed_differences', [])) == len(PROPERTIES)
            and set(visual['composed_differences']) == set(PROPERTIES), 'Unexpected composed visual differences')
    authored = authored_properties(overlay_path)
    for row in changes:
        attribute = row['attribute']
        expected = (PROPERTIES[attribute] * CASES[case]['lighting_gain']
                    if attribute.endswith('intensity') else CASES[case]['crs_roughness'])
        require(row['before'] == PROPERTIES[attribute] and row['after'] == float32(expected)
                and authored[attribute] == row['after'], 'Visual values differ from the recorded condition: ' + attribute)
    files = {str(path): digest(path) for path in (visual_path, profile_config, overlay_path)}
    return {'schema': 'nepes_replay_visual_overlay_v1', 'case': case, 'recipe': CASES[case],
            'source_files_sha256': files, 'baseline_scene_sha256': visual['source_scene_sha256'],
            'changes': changes, 'old_sublayer_paths_resolved': False,
            'stage_metadata_source': 'digest-verified packaged base scene',
            'video_postprocessing_applied': False}


def apply_visual_overlay(stage, visual):
    """Author only validated visual floats on the newly loaded portable scene."""
    if visual is None:
        return None
    readback = []
    for row in visual['changes']:
        attribute = row['attribute']
        require(attribute in PROPERTIES, 'Unexpected visual replay property')
        target = '/World/Scene' + attribute[len('/World'):]
        attr = stage.GetAttributeAtPath(target)
        require(bool(attr) and str(attr.GetTypeName()) == 'float'
                and not attr.GetConnections() and not attr.GetTimeSamples(), 'Unexpected replay visual property: ' + target)
        require(abs(float(attr.Get()) - row['before']) < 1e-6 * max(1., abs(row['before'])),
                'Packaged visual value changed: ' + target)
        require(attr.Set(row['after']), 'Could not restore visual replay value: ' + target)
        require(float(attr.Get()) == row['after'], 'Visual replay readback differs: ' + target)
        readback.append({'attribute': target, 'value': float(attr.Get())})
    return {'case': visual['case'], 'attributes': readback, 'readback_verified': True,
            'old_sublayer_paths_resolved': False}
