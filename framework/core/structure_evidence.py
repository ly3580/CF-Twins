from __future__ import annotations
from copy import deepcopy
import json
import math
from shapely import wkb
from .comparison_background import AREA_NAMES
from .observation_contract import _fields, _int, _text
from .observation_views import _ProgramView
from .spatial_background import RegionQuery, TABLE_FIELDS, _sha, _validate_query
FORMAT_VERSION = 'common-structure-evidence/1'
GEOMETRY_FRAME = 'STATIC_REGION_LOCAL_METRES'
SCOPE_FIELDS = set('region_id source_kind comparison_wkb_hex layer_wkb_hex areas_m2 evidence'.split())
SCOPE_EVIDENCE_FIELDS = set('source_coordinate_system geometry_frame physical_free_space_proven common_neutral eligible_for_matching height_status obstacle_coverage source_scope_accuracy_proven'.split())
TOP_FIELDS = set('format_version status common_neutral eligible_for_matching geometry_frame region_id snapshot_id region layers snapshot items unlocated audit'.split())

def _json(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError('evidence requires finite JSON-compatible values') from exc

def _geometry(value, *, allow_empty=True):
    _text(value, 'geometry')
    try:
        geometry = wkb.loads(bytes.fromhex(value))
    except Exception as exc:
        raise ValueError('invalid evidence geometry') from exc
    if not geometry.is_valid or geometry.has_z or (geometry.is_empty and (not allow_empty)) or (not geometry.is_empty and (not all((math.isfinite(v) for v in geometry.bounds)))):
        raise ValueError('invalid planar evidence geometry')
    return geometry

def _status(geometry):
    return 'EMPTY' if geometry.is_empty else 'POSITIVE_AREA' if geometry.area > 0 else 'BOUNDARY_TOUCH'

def _validate_scope(region, scope, query):
    _fields(scope, SCOPE_FIELDS, 'comparison scope')
    if scope['region_id'] != region['region_id']:
        raise ValueError('scope region binding differs')
    native = region['map_role'] == 'SOURCE_NATIVE'
    if scope['source_kind'] != ('NUPLAN_NATIVE_RANGE' if native else 'TG_COMPARISON_MASK'):
        raise ValueError('scope source kind differs from region role')
    evidence = scope['evidence']
    _fields(evidence, SCOPE_EVIDENCE_FIELDS, 'scope evidence')
    if evidence['geometry_frame'] != GEOMETRY_FRAME or any((evidence[name] is not False for name in ('physical_free_space_proven', 'common_neutral', 'eligible_for_matching', 'source_scope_accuracy_proven'))) or evidence['height_status'] != 'UNKNOWN' or (evidence['obstacle_coverage'] != 'UNKNOWN') or (evidence['source_coordinate_system'] != ('EXTERNAL_BINDING_REQUIRED' if native else 'TARTANAIR_NWU_METRES')):
        raise ValueError('unsupported source scope evidence')
    _fields(scope['layer_wkb_hex'], set(AREA_NAMES), 'scope layers')
    _fields(scope['areas_m2'], set(AREA_NAMES), 'scope areas')
    geometries = {}
    for name in AREA_NAMES:
        value, area = (scope['layer_wkb_hex'][name], scope['areas_m2'][name])
        if value is None:
            if area is not None or not native or name == 'comparison_mask':
                raise ValueError('missing layer geometry/area binding differs')
            geometries[name] = None
            continue
        if native and name != 'comparison_mask':
            raise ValueError('native source has no recorded mask layer')
        geometry = _geometry(value)
        if type(area) not in (int, float) or not math.isfinite(area) or area < 0 or (not math.isclose(area, geometry.area, rel_tol=1e-10, abs_tol=1e-10)) or (not geometry.is_empty and (not query.square.buffer(1e-06).covers(geometry))):
            raise ValueError('layer area or region geometry differs')
        geometries[name] = geometry
    comparison = _geometry(scope['comparison_wkb_hex'])
    if not comparison.equals(geometries['comparison_mask']):
        raise ValueError('comparison mask geometry differs')
    if native:
        if not comparison.equals(query.road):
            raise ValueError('native comparison geometry differs from region')
    elif not geometries['unobserved'].equals(query.square.difference(geometries['observed'])):
        raise ValueError('unobserved geometry differs from observed evidence')
    return (comparison, geometries)

def _validate_snapshot(snapshot, region, query):
    _fields(snapshot, TABLE_FIELDS['snapshots'], 'snapshot')
    _text(snapshot['snapshot_id'], 'snapshot identity')
    if snapshot['region_id'] != region['region_id']:
        raise ValueError('snapshot region binding differs')
    native = region['map_role'] == 'SOURCE_NATIVE'
    if snapshot['kind'] != ('NUPLAN_LIDAR_FRAME' if native else 'TG_STATIC_GRID'):
        raise ValueError('snapshot kind differs from region source')
    binding = snapshot['source_binding']
    if native:
        _int(snapshot['source_time_ns'], 'native time')
        if snapshot['source_time_ns'] % 1000:
            raise ValueError('native time is not integer microseconds times 1000')
        _fields(binding, set('native_source_root native_manifest_sha256 source_db_sha256 lidar_pc_token'.split()), 'native source binding')
        _text(binding['native_source_root'], 'native source root')
        if not _sha(binding['native_manifest_sha256']) or not _sha(binding['source_db_sha256']):
            raise ValueError('invalid native source binding')
        token = binding['lidar_pc_token']
        if not isinstance(token, str) or not token or len(token) % 2 or any((c not in '0123456789abcdef' for c in token)):
            raise ValueError('invalid native frame token')
    else:
        if snapshot['source_time_ns'] is not None:
            raise ValueError('static grid cannot claim synchronized time')
        _fields(binding, set('summary_path summary_sha256 summary_manifest_path summary_manifest_sha256'.split()), 'static source binding')
        _text(binding['summary_path'], 'summary path')
        _text(binding['summary_manifest_path'], 'summary manifest path')
        if not _sha(binding['summary_sha256']) or not _sha(binding['summary_manifest_sha256']):
            raise ValueError('invalid static source binding')
    _validate_query(snapshot, query)

def _height(attributes, native):
    if not native:
        return dict(model='TG_ORIGINAL_CELL_HEIGHT_STATISTICS', status='RECORDED', common_vertical_reference_status='NOT_ESTABLISHED', source_vertical_values_access='AUDIT_ONLY', common_surface_height_proven=False)
    z, height = (attributes['z'], attributes['height'])
    invalid = any((isinstance(value, str) for value in (z, height)))
    return dict(model='NUPLAN_NATIVE_BOX_VERTICAL_ATTRIBUTES', status='INVALID_SOURCE_ATTRIBUTE' if invalid else 'NOT_PROVIDED' if z is None and height is None else 'PARTIALLY_RECORDED' if z is None or height is None else 'RECORDED', native_box_height_m=None if isinstance(height, str) else height, common_vertical_reference_status='NOT_ESTABLISHED', source_vertical_values_access='AUDIT_ONLY', common_surface_height_proven=False)

def _number_items(rows, prefix):
    return [dict(item_id=f'{prefix}_{index:06d}', **row) for index, row in enumerate(sorted(rows, key=_json))]

def build_structure_evidence(region: dict, scope: dict, snapshot: dict | None=None) -> dict:
    """Construct one region or selected source snapshot without instance input.

    Source item identities/times/world coordinates are restricted to ``audit``.
    ``items`` retains every observation record, including comparison OUTSIDE;
    ``structure_view`` selects intersecting records for its requested scope.
    """
    query = RegionQuery(region)
    _text(region['region_id'], 'region identity')
    comparison, geometries = _validate_scope(region, scope, query)
    if snapshot is not None:
        _validate_snapshot(snapshot, region, query)
    observed, unobserved = (scope['areas_m2']['observed'], scope['areas_m2']['unobserved'])
    coverage = 'NOT_ESTABLISHED' if observed is None else 'UNOBSERVED' if observed == 0 else 'FULLY_OBSERVED' if unobserved == 0 else 'PARTIALLY_OBSERVED'
    public_region = dict(anchor_type=region['anchor_type'], size_m=region['size_m'], observation_wkb_hex=region['observation_wkb_hex'], comparison_wkb_hex=scope['comparison_wkb_hex'], source_road_scope_wkb_hex=region['structure_wkb_hex'], source_kind=scope['source_kind'], map_role='SOURCE_DEVELOPMENT_NOT_COMMON_NEUTRAL', comparison_status=_status(comparison), observation_coverage_status=coverage, observation_observed_area_m2=observed, observation_unobserved_area_m2=unobserved, physical_free_space_proven=False, source_semantics_equal_across_datasets=False)
    layers = [dict(layer_name=name, geometry_wkb_hex=scope['layer_wkb_hex'][name], area_m2=scope['areas_m2'][name], status='NOT_PROVIDED' if geometries[name] is None else _status(geometries[name]), geometry_frame=GEOMETRY_FRAME, layer_scope='OBSERVATION', physical_role='NOT_ASSIGNED') for name in AREA_NAMES]
    items, unlocated, public_snapshot = ([], [], None)
    if snapshot is not None:
        native = snapshot['kind'] == 'NUPLAN_LIDAR_FRAME'
        public_snapshot = dict(kind=snapshot['kind'], query_status=snapshot['query_status'], temporal_model='NATIVE_FRAME_INSTANT' if native else 'STATIC_SOURCE_NO_TEMPORAL_JOIN', item_identity_scope='WITHIN_SELECTED_SNAPSHOT_ONLY', unlocated_in_source_count=len(snapshot['unlocated']), outside_observation_source_count=snapshot['outside_count'], source_record_count=snapshot['source_record_count'], physical_free_space_proven=False, physical_obstacle_completeness='NOT_PROVEN', surface_layer_status='UNKNOWN', permanent_static_status='NOT_PROVEN', source_semantics_equal_across_datasets=False)
        for source in snapshot['items']:
            clipped = _geometry(source['observation_wkb_hex'], allow_empty=False).intersection(comparison)
            items.append(dict(category_name=source['category_name'], source_class=source['source_class'], geometry_model=source['geometry_model'], source_geometry_wkb_hex=source['source_geometry_wkb_hex'], observation_wkb_hex=source['observation_wkb_hex'], observation_relation='POSITIVE_AREA' if source['observation_relation'] == 'AREA' else source['observation_relation'], comparison_wkb_hex=None if clipped.is_empty else clipped.wkb_hex, comparison_relation='OUTSIDE' if clipped.is_empty else _status(clipped), original_cell_count=None if native else source['source_attributes']['count'], count_scope='NOT_APPLICABLE' if native else 'ORIGINAL_UNCLIPPED_CELL', height_evidence=_height(source['source_attributes'], native), surface_layer_status='UNKNOWN', permanent_static_status='NOT_PROVEN'))
        for source in snapshot['unlocated']:
            unlocated.append(dict(category_name=source['category_name'], source_class=source['source_class'], reason=source['reason'], spatial_relation='UNLOCATED', height_evidence=_height(source['source_attributes'], native)))
    result = dict(format_version=FORMAT_VERSION, status='DEVELOPMENT', common_neutral=False, eligible_for_matching=False, geometry_frame=GEOMETRY_FRAME, region_id=region['region_id'], snapshot_id=None if snapshot is None else snapshot['snapshot_id'], region=public_region, layers=layers, snapshot=public_snapshot, items=_number_items(items, 'snapshot_item'), unlocated=_number_items(unlocated, 'unlocated_item'), audit=dict(region=region, scope=scope, snapshot=snapshot))
    _json(result)
    return deepcopy(result)

def validate_structure_evidence(evidence: dict) -> dict:
    """Check exact local derivation; external source authentication is separate."""
    _fields(evidence, TOP_FIELDS, 'structure evidence')
    _fields(evidence['audit'], {'region', 'scope', 'snapshot'}, 'source audit')
    expected = build_structure_evidence(evidence['audit']['region'], evidence['audit']['scope'], evidence['audit']['snapshot'])
    if _json(evidence) != _json(expected):
        raise ValueError('structure evidence differs from the supported source derivation')
    items = evidence['items']
    return dict(item_count=len(items), unlocated_count=len(evidence['unlocated']), layer_count=len(evidence['layers']), comparison_item_count=sum((row['comparison_relation'] != 'OUTSIDE' for row in items)), comparison_outside_count=sum((row['comparison_relation'] == 'OUTSIDE' for row in items)), comparison_boundary_count=sum((row['comparison_relation'] == 'BOUNDARY_TOUCH' for row in items)))

def structure_view(evidence: dict, *, scope: str='comparison') -> _ProgramView:
    """Return an allowlisted immutable snapshot with no source clock/identity."""
    if scope not in ('comparison', 'observation'):
        raise ValueError('scope must be comparison or observation')
    validate_structure_evidence(evidence)
    region = evidence['region']
    selected = _geometry(region[scope + '_wkb_hex'])
    layers = []
    for row in evidence['layers']:
        geometry = None if row['geometry_wkb_hex'] is None else _geometry(row['geometry_wkb_hex']).intersection(selected)
        layers.append(dict(layer_name=row['layer_name'], geometry_wkb_hex=None if geometry is None else geometry.wkb_hex, area_m2=None if geometry is None else float(geometry.area), status='NOT_PROVIDED' if geometry is None else _status(geometry), geometry_frame=GEOMETRY_FRAME, layer_scope=scope.upper(), physical_role='NOT_ASSIGNED'))
    item_fields = ('item_id', 'category_name', 'source_class', 'geometry_model', 'original_cell_count', 'count_scope', 'height_evidence', 'surface_layer_status', 'permanent_static_status')
    items = [dict({name: row[name] for name in item_fields}, geometry_wkb_hex=row[scope + '_wkb_hex'], spatial_relation=row[scope + '_relation']) for row in evidence['items'] if row[scope + '_wkb_hex'] is not None]
    snapshot = evidence['snapshot'] or dict(kind='NOT_REQUESTED', query_status='NOT_REQUESTED', temporal_model='NOT_REQUESTED', item_identity_scope='WITHIN_SELECTED_SNAPSHOT_ONLY', physical_free_space_proven=False, physical_obstacle_completeness='NOT_PROVEN', surface_layer_status='UNKNOWN', source_semantics_equal_across_datasets=False)
    return _ProgramView(dict(region_spec=[dict(region, geometry_frame=GEOMETRY_FRAME, scope_name=scope.upper(), geometry_wkb_hex=region[scope + '_wkb_hex'])], structure_layers=layers, background_evidence=[snapshot], background_items=items, unlocated_evidence=evidence['unlocated']))
