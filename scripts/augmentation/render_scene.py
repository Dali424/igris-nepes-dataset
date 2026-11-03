#!/usr/bin/env python3
"""Create a private light/visual-material overlay. OpenUSD only; never starts Kit."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

CASES = {
    'dim_matte': {'lighting_gain': 0.70, 'crs_roughness': 0.60},
    'bright_satin': {'lighting_gain': 1.30, 'crs_roughness': 0.35},
}
PROPERTIES = {
    '/World/Light.inputs:intensity': 350.0,
    '/World/RectLight.inputs:intensity': 15000000.0,
    '/World/Ring_02/Visual/_materials/Plastic_Dark_Charcoal/Principled_BSDF.inputs:roughness': 0.4300000071525574,
    '/World/Ring_04/Visual/_materials/Plastic_Dark_Charcoal/Principled_BSDF.inputs:roughness': 0.4300000071525574,
}

def require(value, message):
    if not value:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)

def inventory(stage):
    attrs, relations, prims = {}, {}, {}
    for prim in stage.TraverseAll():
        prims[str(prim.GetPath())] = (prim.GetTypeName(), prim.IsActive(), prim.IsInstanceable(), tuple(prim.GetAppliedSchemas()))
        for attr in prim.GetAttributes():
            attrs[str(attr.GetPath())] = (str(attr.GetTypeName()), str(attr.Get()), tuple(map(str, attr.GetConnections())), tuple(attr.GetTimeSamples()))
        for relation in prim.GetRelationships():
            relations[str(relation.GetPath())] = tuple(map(str, relation.GetTargets()))
    return attrs, relations, prims

def prepare(parent_config, bundle, output, case):
    from pxr import Usd, Sdf, UsdShade
    parent_config, bundle = Path(parent_config).resolve(strict=True), Path(bundle).resolve(strict=True)
    output = Path(output).resolve()
    require(case in CASES, 'Unknown visual condition')
    require(not output.is_relative_to(bundle) and not bundle.is_relative_to(output), 'Overlay output overlaps immutable bundle')
    require(not output.exists(), 'Refuse to overwrite an existing overlay')
    original = read(parent_config)
    scene = Path(original['single_arm_layout']['scene_usd']).resolve(strict=True)
    require(scene == bundle / 'assets/workcell/layout.usda', 'Parent must use the audited unmodified portable scene')
    require(sha(scene) == original['single_arm_layout']['scene_sha256'], 'Parent scene checksum mismatch')
    baseline = Usd.Stage.Open(str(scene)); require(bool(baseline), 'Cannot open baseline scene')
    material_bindings = {}
    for ring in ('Ring_02', 'Ring_04'):
        material_path = '/World/' + ring + '/Visual/_materials/Plastic_Dark_Charcoal'
        shader_path = material_path + '/Principled_BSDF'
        shader = UsdShade.Shader(baseline.GetPrimAtPath(shader_path))
        require(shader.GetIdAttr().Get() == 'UsdPreviewSurface', 'CRS visual shader type changed')
        bound = []
        for prim in Usd.PrimRange(baseline.GetPrimAtPath('/World/' + ring + '/Visual')):
            if prim.GetTypeName() == 'Mesh':
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material and str(material.GetPath()) == material_path:
                    surface = material.ComputeSurfaceSource()[0]
                    require(str(surface.GetPath()) == shader_path, 'Roughness input is not on the bound surface shader')
                    bound.append(str(prim.GetPath()))
        require(bool(bound), 'CRS material is not bound to visible CAD meshes')
        material_bindings[ring] = {'material': material_path, 'shader_id': 'UsdPreviewSurface', 'bound_mesh_count': len(bound)}
    before = inventory(baseline)
    output.mkdir(parents=True)
    overlay = output / 'scene_visual_overlay.usda'
    layer = Sdf.Layer.CreateNew(str(overlay))
    layer.subLayerPaths = [os.path.relpath(scene, output)]
    for key, value in baseline.GetPseudoRoot().GetAllMetadata().items():
        layer.pseudoRoot.SetInfo(key, value)
    stage = Usd.Stage.Open(layer); require(bool(stage), 'Cannot compose overlay')
    changes = []
    for path, expected in PROPERTIES.items():
        attr = stage.GetAttributeAtPath(path)
        require(bool(attr) and str(attr.GetTypeName()) == 'float' and not attr.GetConnections() and not attr.GetTimeSamples(), 'Unexpected visual property: ' + path)
        old = float(attr.Get())
        require(abs(old - expected) < 1e-6 * max(1.0, abs(old)), 'Visual source value changed: ' + path)
        value = old * CASES[case]['lighting_gain'] if path.endswith('intensity') else CASES[case]['crs_roughness']
        require(attr.Set(value), 'Could not author visual input')
        changes.append({'attribute': path, 'before': old, 'after': float(attr.Get())})
    layer.Save()
    after = inventory(stage)
    require(before[1:] == after[1:], 'Prim types, activation, APIs or relationships changed')
    require(set(before[0]) == set(after[0]), 'Scene attributes were added or removed')
    differences = [key for key in before[0] if before[0][key] != after[0][key]]
    require(set(differences) == set(PROPERTIES), 'Unexpected composed changes')
    require(stage.GetPseudoRoot().GetAllMetadata() == baseline.GetPseudoRoot().GetAllMetadata(), 'Stage units/axis/camera/render metadata changed')
    authored = []
    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec):
            authored.append(str(path)); require(str(path) in PROPERTIES, 'Unauthorized override')
        require(not isinstance(spec, Sdf.RelationshipSpec), 'Relationships cannot be overridden')
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    require(set(authored) == set(PROPERTIES), 'Wrong authored property set')
    config = deepcopy(original)
    config['single_arm_layout'].update(scene_usd=str(overlay), scene_sha256=sha(overlay))
    restored = deepcopy(config); restored['single_arm_layout'] = deepcopy(original['single_arm_layout'])
    require(restored == original, 'A non-scene configuration setting changed')
    write(output / 'config.json', config)
    evidence = {'schema': 'nepes_native_render_overlay_v1', 'case': case, 'recipe': CASES[case],
                'parent_config': str(parent_config), 'parent_config_sha256': sha(parent_config),
                'source_scene': str(scene), 'source_scene_sha256': sha(scene),
                'overlay': str(overlay), 'overlay_sha256': sha(overlay), 'config_sha256': sha(output / 'config.json'),
                'changes': changes, 'composed_differences': differences, 'attributes_compared': len(before[0]),
                'crs_material_bindings': material_bindings,
                'only_visual_allowlisted_properties_authored': True, 'all_physics_geometry_poses_and_cameras_unchanged': True,
                'stage_metadata_unchanged': True, 'simulation_executed': False,
                'native_semantics': 'A separate native physics episode must supply its own actions, states, camera timestamps and successful-task admission.',
                'limitations': ['Captured NuRec room appearance need not relight with ordinary USD lights.',
                               'Roughness changes only bound CRS visual shader inputs; physical friction and pads are unchanged.',
                               'CPU composition checks do not prove visible effect; encoded-camera pilot QA is mandatory.']}
    require(sha(scene) == original['single_arm_layout']['scene_sha256'], 'Original scene changed during authoring')
    write(output / 'visual.json', evidence)
    return evidence

def main():
    # Expose Isaac's CPU USD libraries in this helper process only. Importing
    # SimulationApp is neither needed nor allowed for overlay preparation.
    if importlib.util.find_spec('pxr') is None:
        require(os.environ.get('_NEPES_CPU_USD_BOOTSTRAPPED') != '1', 'CPU USD bootstrap failed')
        spec = importlib.util.find_spec('isaacsim')
        require(spec is not None and spec.submodule_search_locations, 'Use the Isaac Python for CPU overlay preparation')
        installation = Path(next(iter(spec.submodule_search_locations)))
        candidates = sorted((installation / 'extscache').glob('omni.usd.libs-*/pxr/Usd/__init__.py'))
        require(len(candidates) == 1, 'Expected one installed Isaac CPU USD library version')
        library = candidates[0].parents[2]
        env = dict(os.environ, _NEPES_CPU_USD_BOOTSTRAPPED='1', PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH'] = str(library)
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(library / 'bin'), str(Path(sys.prefix) / 'lib'),
                                                str(installation / 'kit/kernel/plugins'), env.get('LD_LIBRARY_PATH', '')])
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], env)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-config', 'bundle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_config, args.bundle, args.output, args.case), indent=2))

if __name__ == '__main__':
    main()
