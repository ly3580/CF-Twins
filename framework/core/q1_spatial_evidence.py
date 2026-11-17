from copy import deepcopy
import json
import math
from shapely import affinity, from_wkb
from shapely.geometry import mapping, shape
from . import q1_atomics
from .observation_views import TaskContextView
VERSION = 'q1-spatial-evidence/0.1.0'
_MAP_KEYS = {'scope', 'geometry_frame', 'coverage_status', 'free_space_status', 'geometry', 'layers', 'road_components'}
_GEO_KEYS = {'Point': {'type', 'coordinates'}, 'MultiPoint': {'type', 'coordinates'}, 'LineString': {'type', 'coordinates'}, 'MultiLineString': {'type', 'coordinates'}, 'Polygon': {'type', 'coordinates'}, 'MultiPolygon': {'type', 'coordinates'}, 'GeometryCollection': {'type', 'geometries'}}

def _geo(value):
    result = dict(mapping(from_wkb(bytes.fromhex(value))))
    result = _lists(result)
    return result

def _lists(value):
    if isinstance(value, dict):
        return {key: _lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lists(item) for item in value]
    return value

def _point(point, transform):
    c, s = (math.cos(transform['theta_rad']), math.sin(transform['theta_rad']))
    return [c * point[0] - s * point[1] + transform['x_m'], s * point[0] + c * point[1] + transform['y_m']]

def _transform_map(value, transform):
    result = deepcopy(value)
    result['geometry'] = _transform_geo_hex(value['_geometry_wkb_hex'], transform)
    for layer, source in zip(result['layers'], value['_layer_wkb_hex']):
        layer['geometry'] = None if source is None else _transform_geo_hex(source, transform)
    for component, source in zip(result['road_components'], value['_component_wkb_hex']):
        component['geometry'] = _transform_geo_hex(source['geometry'], transform)
        component['boundary_contact'] = _transform_geo_hex(source['boundary_contact'], transform)
    for key in ('_geometry_wkb_hex', '_layer_wkb_hex', '_component_wkb_hex'):
        del result[key]
    return result

def _transform_geo_hex(value, transform):
    c, s = (math.cos(transform['theta_rad']), math.sin(transform['theta_rad']))
    return _lists(dict(mapping(affinity.affine_transform(from_wkb(bytes.fromhex(value)), [c, -s, s, c, transform['x_m'], transform['y_m']]))))

def _local_map(value):
    region = value['region']
    layers = []
    layer_wkb = []
    for row in value['layers']:
        layer_wkb.append(row['geometry_wkb_hex'])
        layers.append(dict(layer_name=row['layer_name'], status=row['status'], physical_role=row['physical_role'], layer_scope=row['layer_scope'], coverage_status=row['coverage_status'], geometry=None if row['geometry_wkb_hex'] is None else _geo(row['geometry_wkb_hex'])))
    components, component_wkb, component_ids = ([], [], {})
    for ordinal, row in enumerate(value['road_components']):
        local_id = 'component_%06d' % ordinal
        component_ids[row['component_id']] = local_id
        component_wkb.append(dict(geometry=row['component_wkb_hex'], boundary_contact=row['boundary_contact_wkb_hex']))
        components.append(dict(component_ordinal=ordinal, geometry=_geo(row['component_wkb_hex']), boundary_contact=_geo(row['boundary_contact_wkb_hex']), boundary_contact_status=row['boundary_contact_status']))
    result = dict(scope=region['scope'], geometry_frame=region['geometry_frame'], coverage_status=region['coverage_status'], free_space_status=value['free_space_status'], geometry=_geo(region['geometry_wkb_hex']), layers=layers, road_components=components, _geometry_wkb_hex=region['geometry_wkb_hex'], _layer_wkb_hex=layer_wkb, _component_wkb_hex=component_wkb)
    return (result, component_ids)

def _entry(view, instance_id, component_ids):
    value = q1_atomics._entry(view, instance_id)
    result = dict(status=value['status'], reason_codes=deepcopy(value['reason_codes']), interval_ns=deepcopy(value['interval_ns']), entry_road_status=value['entry_road_status'], first_point=_timed_point(value['first_point']), witness=None if value['witness'] is None else _timed_point(value['witness']), direction=_direction(q1_atomics._entry_direction(value), ('ENTRY_MOTION_NOT_WITNESSED',)), candidate_components=[component_ids[item] for item in value['candidate_component_ids']], first_components=[component_ids[item] for item in value['first_component_ids']], reference_point=value['reference_point'])
    return result

def _timed_point(value):
    return dict(relative_time_ns=value['relative_time_ns'], x_m=value['x_m'], y_m=value['y_m'], unit='METER')

def _direction(value, reasons=()):
    return dict(value_rad=value, status='VALID' if value is not None else 'INSUFFICIENT_EVIDENCE', unit='RADIAN', reason_codes=[] if value is not None else list(reasons))

def _exit(value):
    direction = q1_atomics._exit_direction(value)
    if value['status'] != 'VALID':
        reasons = ('EXIT_NOT_OBSERVED',)
    elif value['stop_witness'] is None:
        reasons = ('EXIT_MOTION_WITNESS_NOT_PROVIDED',)
    else:
        reasons = ('EXIT_MOTION_NOT_WITNESSED',)
    return dict(status=value['status'], reason_codes=deepcopy(value['reason_codes']), interval_ns=deepcopy(value['interval_ns']), end_reason=value['end_reason'], right_censored=value['right_censored'], last_point=_timed_point(value['last_point']), stop_witness=None if value['stop_witness'] is None else _timed_point(value['stop_witness']), direction=_direction(direction, reasons))

def _mapped_endpoint(value, transform):
    result = deepcopy(value)
    point_keys = ('first_point', 'last_point', 'witness', 'stop_witness')
    for key in point_keys:
        point = result.get(key)
        if point is not None:
            point['x_m'], point['y_m'] = _point([point['x_m'], point['y_m']], transform)
    direction = result.get('direction')
    if direction is not None and direction['value_rad'] is not None:
        direction['value_rad'] = float(q1_atomics.wrap_to_pi(direction['value_rad'] + transform['theta_rad']))
    return result

def collect_spatial_evidence(context_a, context_b, *, instance_a, instance_b, alignment, profile='ENTRY', posthoc=None):
    """Collect numeric U01 geometry and witnessed points under each retained B-to-A transform."""
    for context in (context_a, context_b):
        if type(context) is not TaskContextView:
            raise TypeError('requires TaskContextView inputs')
    if profile not in ('ENTRY', 'GT_POSTHOC'):
        raise ValueError('unsupported task comparison profile')
    if type(alignment) is not q1_atomics._AlignmentResult:
        raise ValueError('requires immutable U02 alignment')
    if profile == 'ENTRY' and posthoc is not None:
        raise ValueError('posthoc evidence is forbidden in ENTRY profile')
    if profile == 'GT_POSTHOC' and type(posthoc) is not q1_atomics.PosthocEvaluationContext:
        raise TypeError('GT_POSTHOC requires PosthocEvaluationContext')
    data = alignment.as_dict()
    if data['transform_direction'] != 'B_TO_A':
        raise ValueError('alignment must map B to A')
    maps = []
    for context, side in ((context_a, 'a'), (context_b, 'b')):
        static = q1_atomics._context_structure(context)
        if static != data['input_' + side]:
            raise ValueError('alignment does not bind to these static task contexts')
        maps.append(_local_map(data['input_' + side]))
    entries = [_entry(context_a, instance_a, maps[0][1]), _entry(context_b, instance_b, maps[1][1])]
    ends = None
    if profile == 'GT_POSTHOC':
        ends = json.loads(object.__getattribute__(posthoc, '_snapshot'))
        for index, end in enumerate(ends):
            if end['instance_id'] != (instance_a if index == 0 else instance_b):
                raise ValueError('posthoc evidence belongs to different tasks')
            if end['region_id'] != data['region_a' if index == 0 else 'region_b']:
                raise ValueError('posthoc evidence belongs to different regions')
    sides = {'a': dict(local_map=_public_map(maps[0][0]), entry=entries[0]), 'b': dict(local_map=_public_map(maps[1][0]), entry=entries[1])}
    if ends is not None:
        sides['a']['observed_exit'] = _exit(ends[0])
        sides['b']['observed_exit'] = _exit(ends[1])
    transforms = []
    for row in data['transforms']:
        transform = {key: row[key] for key in ('theta_rad', 'x_m', 'y_m')}
        aligned = dict(grid_index=deepcopy(row['grid_index']), transform=deepcopy(transform), transform_direction='B_TO_A', b_local_map_in_a=_transform_public_map(_public_map(maps[1][0]), transform), b_entry_in_a=_mapped_endpoint(entries[1], transform))
        if ends is not None:
            aligned['b_observed_exit_in_a'] = _mapped_endpoint(sides['b']['observed_exit'], transform)
        transforms.append(aligned)
    result = dict(tool_id='Q1_SPATIAL_EVIDENCE', definition_version=VERSION, evidence_scope='U01_GEOMETRY_AND_WITNESSED_ENDPOINTS', profile=profile, coordinate_frame='SIDE_LOCAL_METRES', transform_direction='B_TO_A', geometry_coordinate_unit='METER', direction_unit='RADIAN', point_semantics=dict(entry='first internal sample; witness is preceding observed sample', exit='last internal sample; witness is observed stop sample when available'), sides=sides, per_alignment=transforms)
    return validate_spatial_evidence(result)

def _public_map(value):
    return {key: deepcopy(item) for key, item in value.items() if not key.startswith('_')}

def validate_spatial_evidence(value):
    """Strictly validate and detach the spatial evidence profile for persistence/readback."""
    if type(value) is not dict or set(value) != {'tool_id', 'definition_version', 'evidence_scope', 'profile', 'coordinate_frame', 'transform_direction', 'geometry_coordinate_unit', 'direction_unit', 'point_semantics', 'sides', 'per_alignment'}:
        raise ValueError('spatial evidence has missing or unexpected fields')
    if (value['tool_id'], value['definition_version'], value['evidence_scope']) != ('Q1_SPATIAL_EVIDENCE', VERSION, 'U01_GEOMETRY_AND_WITNESSED_ENDPOINTS'):
        raise ValueError('spatial evidence identity differs')
    if value['profile'] not in ('ENTRY', 'GT_POSTHOC') or value['coordinate_frame'] != 'SIDE_LOCAL_METRES' or value['transform_direction'] != 'B_TO_A' or (value['geometry_coordinate_unit'] != 'METER') or (value['direction_unit'] != 'RADIAN'):
        raise ValueError('spatial evidence profile or coordinate contract differs')
    if set(value['sides']) != {'a', 'b'} or type(value['per_alignment']) is not list:
        raise ValueError('spatial evidence sides or alignments are invalid')
    expected_semantics = dict(entry='first internal sample; witness is preceding observed sample', exit='last internal sample; witness is observed stop sample when available')
    if value['point_semantics'] != expected_semantics:
        raise ValueError('spatial evidence point semantics differ')
    for side in ('a', 'b'):
        row = value['sides'][side]
        required = {'local_map', 'entry'} | ({'observed_exit'} if value['profile'] == 'GT_POSTHOC' else set())
        if type(row) is not dict or set(row) != required:
            raise ValueError('spatial evidence side fields differ')
        _validate_map(row['local_map'])
        _validate_endpoint(row['entry'], is_exit=False)
        components = {'component_%06d' % item['component_ordinal'] for item in row['local_map']['road_components']}
        if any((item not in components for field in ('candidate_components', 'first_components') for item in row['entry'][field])):
            raise ValueError('spatial entry references an absent local component')
        if 'observed_exit' in row:
            _validate_endpoint(row['observed_exit'], is_exit=True)
    for row in value['per_alignment']:
        required = {'grid_index', 'transform', 'transform_direction', 'b_local_map_in_a', 'b_entry_in_a'}
        if value['profile'] == 'GT_POSTHOC':
            required.add('b_observed_exit_in_a')
        if type(row) is not dict or set(row) != required or row['transform_direction'] != 'B_TO_A':
            raise ValueError('spatial evidence alignment fields differ')
        if type(row['grid_index']) is not list or len(row['grid_index']) != 3 or any((type(i) is not int or i < 0 for i in row['grid_index'])):
            raise ValueError('spatial evidence grid index is invalid')
        transform = row['transform']
        if type(transform) is not dict or set(transform) != {'theta_rad', 'x_m', 'y_m'}:
            raise ValueError('spatial evidence transform fields differ')
        _finite_map(transform)
        _validate_map(row['b_local_map_in_a'])
        _validate_endpoint(row['b_entry_in_a'], is_exit=False)
        if value['profile'] == 'GT_POSTHOC':
            _validate_endpoint(row['b_observed_exit_in_a'], is_exit=True)
        if row['b_local_map_in_a'] != _transform_public_map(value['sides']['b']['local_map'], transform):
            raise ValueError('mapped geometry differs from B_TO_A transform')
        if row['b_entry_in_a'] != _mapped_endpoint(value['sides']['b']['entry'], transform):
            raise ValueError('mapped entry differs from B_TO_A transform')
        if value['profile'] == 'GT_POSTHOC' and row['b_observed_exit_in_a'] != _mapped_endpoint(value['sides']['b']['observed_exit'], transform):
            raise ValueError('mapped exit differs from B_TO_A transform')
    detached = deepcopy(value)
    if _keys(detached) & {'region_id', 'instance_id', 'case_id', 'source_path', 'component_id'}:
        raise ValueError('spatial evidence contains identity fields')
    return detached

def _transform_geometry(geometry, transform):
    result = deepcopy(geometry)
    _transform_coordinates(result, transform)
    return result

def _transform_public_map(value, transform):
    result = deepcopy(value)
    result['geometry'] = _transform_geometry(result['geometry'], transform)
    for row in result['layers']:
        if row['geometry'] is not None:
            row['geometry'] = _transform_geometry(row['geometry'], transform)
    for row in result['road_components']:
        row['geometry'] = _transform_geometry(row['geometry'], transform)
        row['boundary_contact'] = _transform_geometry(row['boundary_contact'], transform)
    return result

def _keys(value):
    if type(value) is dict:
        return set(value).union(*(_keys(item) for item in value.values())) if value else set()
    if type(value) is list:
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()

def _transform_coordinates(value, transform):
    if type(value) is dict:
        for key, item in value.items():
            if key in ('coordinates',):
                value[key] = _transform_coordinate_tree(item, transform)
            elif key == 'geometries':
                for geometry in item:
                    _transform_coordinates(geometry, transform)

def _transform_coordinate_tree(value, transform):
    if not value:
        return []
    if type(value[0]) in (int, float):
        xy = _point(value[:2], transform)
        return xy + value[2:]
    return [_transform_coordinate_tree(item, transform) for item in value]

def _validate_map(value):
    if type(value) is not dict or set(value) != _MAP_KEYS:
        raise ValueError('spatial map fields differ')
    _validate_geometry(value['geometry'])
    if type(value['layers']) is not list or type(value['road_components']) is not list:
        raise ValueError('spatial map rows are invalid')
    for row in value['layers']:
        if type(row) is not dict or set(row) != {'layer_name', 'status', 'physical_role', 'layer_scope', 'coverage_status', 'geometry'}:
            raise ValueError('spatial layer fields differ')
        if row['geometry'] is not None:
            _validate_geometry(row['geometry'])
    for ordinal, row in enumerate(value['road_components']):
        if type(row) is not dict or set(row) != {'component_ordinal', 'geometry', 'boundary_contact', 'boundary_contact_status'} or row['component_ordinal'] != ordinal:
            raise ValueError('spatial road component fields differ')
        _validate_geometry(row['geometry'])
        _validate_geometry(row['boundary_contact'])

def _validate_geometry(value):
    if type(value) is not dict or value.get('type') not in _GEO_KEYS or set(value) != _GEO_KEYS[value.get('type')]:
        raise ValueError('numeric geometry has invalid type or fields')
    _finite_map(value)
    if value['type'] == 'GeometryCollection':
        for geometry in value['geometries']:
            _validate_geometry(geometry)
    else:
        coordinates = value['coordinates']
        if value['type'] == 'Point':
            if coordinates:
                _validate_xy(coordinates)
        elif value['type'] == 'MultiPoint':
            for point in coordinates:
                _validate_xy(point)
        elif value['type'] == 'LineString':
            _validate_line(coordinates)
        elif value['type'] == 'MultiLineString':
            for line in coordinates:
                _validate_line(line)
        elif value['type'] == 'Polygon':
            _validate_polygon(coordinates)
        else:
            for polygon in coordinates:
                _validate_polygon(polygon)
    geometry = shape(value)
    if geometry.has_z or not geometry.is_valid:
        raise ValueError('numeric geometry must be valid 2D geometry')

def _validate_xy(value):
    if type(value) is not list or len(value) != 2 or any((type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        raise ValueError('numeric geometry coordinate must be a finite XY pair')

def _validate_line(value):
    if type(value) is not list:
        raise ValueError('numeric geometry line must be an array')
    if not value:
        return
    if len(value) < 2:
        raise ValueError('numeric geometry line must contain at least two coordinates')
    for point in value:
        _validate_xy(point)

def _validate_polygon(value):
    if type(value) is not list:
        raise ValueError('numeric geometry polygon rings must be arrays')
    for ring in value:
        _validate_line(ring)
        if ring[0] != ring[-1]:
            raise ValueError('numeric polygon ring must be closed')

def _validate_endpoint(value, *, is_exit):
    keys = {'status', 'reason_codes', 'interval_ns', 'end_reason', 'right_censored', 'last_point', 'stop_witness', 'direction'} if is_exit else {'status', 'reason_codes', 'interval_ns', 'entry_road_status', 'first_point', 'witness', 'direction', 'candidate_components', 'first_components', 'reference_point'}
    if type(value) is not dict or set(value) != keys:
        raise ValueError('spatial endpoint fields differ')
    if value['status'] not in ('VALID', 'INSUFFICIENT_EVIDENCE') or type(value['reason_codes']) is not list or any((type(reason) is not str for reason in value['reason_codes'])):
        raise ValueError('spatial endpoint status or reasons are invalid')
    interval = value['interval_ns']
    if type(interval) is not list or len(interval) != 2 or any((item is not None and (type(item) is not int or not -2 ** 63 <= item < 2 ** 63) for item in interval)):
        raise ValueError('spatial endpoint interval is invalid')
    point_names = ('last_point', 'stop_witness') if is_exit else ('first_point', 'witness')
    for name in point_names:
        point = value[name]
        if point is not None and (type(point) is not dict or set(point) != {'relative_time_ns', 'x_m', 'y_m', 'unit'} or point['unit'] != 'METER'):
            raise ValueError('spatial endpoint point is invalid')
    if not is_exit:
        if type(value['entry_road_status']) is not str or type(value['reference_point']) is not str or type(value['candidate_components']) is not list or (type(value['first_components']) is not list) or any((type(item) is not str for item in value['candidate_components'] + value['first_components'])):
            raise ValueError('spatial entry metadata is invalid')
    else:
        if type(value['right_censored']) is not bool or type(value['end_reason']) is not str:
            raise ValueError('spatial exit censoring is invalid')
        if value['right_censored'] == (value['end_reason'] == 'OBSERVED_EXIT'):
            raise ValueError('spatial exit reason and censoring disagree')
        if value['last_point'] is None or interval[0] != value['last_point']['relative_time_ns']:
            raise ValueError('spatial exit lower bound differs from last internal point')
        if value['stop_witness'] is not None and (value['right_censored'] or interval[1] != value['stop_witness']['relative_time_ns']):
            raise ValueError('spatial exit witness differs from observed interval')
    direction = value['direction']
    if type(direction) is not dict or set(direction) != {'value_rad', 'status', 'unit', 'reason_codes'} or direction['unit'] != 'RADIAN' or ((direction['value_rad'] is None) != (direction['status'] != 'VALID')):
        raise ValueError('spatial endpoint direction is invalid')
    _finite_map(value)
    point, witness = (value['last_point'], value['stop_witness']) if is_exit else (value['first_point'], value['witness'])
    expected = None
    if witness is not None:
        dx = witness['x_m'] - point['x_m'] if is_exit else point['x_m'] - witness['x_m']
        dy = witness['y_m'] - point['y_m'] if is_exit else point['y_m'] - witness['y_m']
        if dx != 0 or dy != 0:
            expected = math.atan2(dy, dx)
    actual = direction['value_rad']
    if (expected is None) != (actual is None) or (expected is not None and abs(float(q1_atomics.wrap_to_pi(expected - actual))) > 1e-12):
        raise ValueError('spatial endpoint direction differs from its witnessed points')

def _finite_map(value):
    if type(value) in (int, float):
        if type(value) is bool or not math.isfinite(value):
            raise ValueError('spatial evidence contains non-finite numbers')
    elif type(value) is dict:
        for item in value.values():
            _finite_map(item)
    elif type(value) is list:
        for item in value:
            _finite_map(item)
