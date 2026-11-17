from __future__ import annotations
from copy import deepcopy
from collections.abc import Mapping
from itertools import product
import json
import math
from shapely import affinity, from_wkb, get_coordinates
from shapely.geometry import Point
from .geometry import wrap_to_pi
from .observation_views import StructureView, TaskView, TaskContextView, DecisionView
from .observation_field_sources import LOCAL_FIELDS
VERSION = 'q1-planar-atomics/0.2.0'
_CONFIG = frozenset('theta_origin theta_step theta_count x_origin x_step x_count y_origin y_step y_count search_budget epsilon_C epsilon_D boundary_step_m area_tolerance_m2 distance_tolerance_m'.split())
_REGION_RESULT = frozenset('tool_id definition_version status reason_codes region layers road_components evidence_scope free_space_status'.split())
_ALIGNMENT_RESULT = frozenset('tool_id definition_version status reason_codes input_a input_b region_a region_b transform_direction evidence_scope parameters_used layer_pairs candidate_count evaluated_candidates transforms'.split())

class _AlignmentResult(Mapping):
    """Keep task code from changing or pruning structure-selected solutions."""
    __slots__ = ('_snapshot', '_keys')

    def __init__(self, result):
        object.__setattr__(self, '_snapshot', json.dumps(result, allow_nan=False).encode('utf-8'))
        object.__setattr__(self, '_keys', tuple(result))

    def __setattr__(self, name, value):
        raise AttributeError('structure alignment is immutable')

    def __delattr__(self, name):
        raise AttributeError('structure alignment is immutable')

    def __getitem__(self, key):
        return self.as_dict()[key]

    def __iter__(self):
        return iter(object.__getattribute__(self, '_keys'))

    def __len__(self):
        return len(object.__getattribute__(self, '_keys'))

    def as_dict(self):
        """Detached serialization; U04 accepts only the original result object."""
        return json.loads(object.__getattribute__(self, '_snapshot'))

def _fields(value, names, label):
    if type(value) not in (dict, _AlignmentResult) or set(value) != set(names):
        raise ValueError(label + ' has missing or unexpected fields')

def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise ValueError(label + ' must be a finite number')
    return float(value)

def _integer(value, label, *, positive=False):
    if type(value) is not int or (positive and value <= 0) or (not -2 ** 63 <= value < 2 ** 63):
        raise ValueError(label + ' must be an int64 integer' + (' > 0' if positive else ''))
    return value

def _view(view, expected):
    if type(view) is not expected:
        raise TypeError('requires ' + expected.__name__)

def _one(view, table, *, instance_id=None):
    rows = view.table(table)
    if instance_id is not None:
        rows = [row for row in rows if row['instance_id'] == instance_id]
    if len(rows) != 1:
        raise ValueError(table + ' requires exactly one selected record')
    return rows[0]

def _geometry(hex_wkb):
    if not isinstance(hex_wkb, str):
        raise ValueError('geometry must be hex WKB')
    geometry = from_wkb(hex_wkb)
    if geometry.has_z or not geometry.is_valid or (not math.isfinite(geometry.area)) or (not math.isfinite(geometry.length)) or any((not math.isfinite(float(x)) for p in get_coordinates(geometry) for x in p)):
        raise ValueError('requires valid finite geometry; no implicit repair')
    return geometry

def _polygon(hex_wkb):
    geometry = _geometry(hex_wkb)
    if geometry.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('requires a two-dimensional Polygon or MultiPolygon')
    return geometry

def _tool(tool_id, status, reasons=(), **fields):
    result = dict(tool_id=tool_id, definition_version=VERSION, status=status, reason_codes=list(reasons), **fields)
    return _AlignmentResult(result) if tool_id == 'U02' else result

def _measurement(quantity, value, unit, *, status='VALID', reasons=(), scope='STATIC_REGION', parameters=None):
    if status == 'VALID':
        value = _finite(value, quantity)
    elif value is not None:
        raise ValueError('non-VALID measurements must be null')
    return dict(quantity_id=quantity, definition_version=VERSION, value=value, status=status, unit=unit, observation_scope=scope, evidence_basis='RULE_DERIVED', reason_codes=list(reasons), support_summary=dict(evidence_scope='SCOPE_GEOMETRY_ONLY'), parameters_used={} if parameters is None else deepcopy(parameters))

def get_local_region(view, *, scope):
    """U01: static integrated geometry, with scope and physical unknowns intact."""
    _view(view, StructureView)
    if scope not in ('comparison', 'observation'):
        raise ValueError('scope must be comparison or observation')
    source = _one(view, 'region_spec')
    required = {'region_id', 'geometry_frame', 'comparison_wkb_hex', 'observation_wkb_hex', 'source_road_scope_wkb_hex', 'observation_coverage_status'}
    if not required <= set(source):
        raise ValueError('requires an integrated structure view with explicit comparison scope')
    if source['geometry_frame'] != 'STATIC_REGION_LOCAL_METRES':
        raise ValueError('unsupported geometry frame')
    selected = _polygon(source[scope + '_wkb_hex'])
    layers = []
    for name, field in [('comparison_scope', 'comparison_wkb_hex'), ('source_road_scope', 'source_road_scope_wkb_hex')]:
        geometry = _polygon(source[field]).intersection(selected)
        layers.append(dict(layer_name=name, geometry_wkb_hex=geometry.wkb_hex, status='EMPTY' if geometry.is_empty else 'POSITIVE_AREA', physical_role='NOT_ASSIGNED', layer_scope=scope.upper(), coverage_status='UNKNOWN'))
    for row in view.table('structure_layers'):
        if row['region_id'] != source['region_id']:
            raise ValueError('structure layer belongs to another region')
        same_scope = row['layer_scope'] == scope.upper()
        geometry = row['geometry_wkb_hex'] if same_scope else None
        if geometry is not None:
            _geometry(geometry)
        layers.append(dict(layer_name=row['layer_name'], geometry_wkb_hex=geometry, status=row['status'] if same_scope else 'NOT_PROVIDED', physical_role=row['physical_role'], layer_scope=scope.upper(), coverage_status='UNKNOWN'))
    if len({row['layer_name'] for row in layers}) != len(layers):
        raise ValueError('duplicate structure layer names')
    components = []
    for row in view.table('road_components'):
        if row['region_id'] != source['region_id']:
            raise ValueError('road component belongs to another region')
        _polygon(row['component_wkb_hex'])
        components.append({key: row[key] for key in ('component_id', 'component_wkb_hex', 'boundary_contact_wkb_hex', 'boundary_contact_status')})
    region = dict(region_id=source['region_id'], scope=scope, geometry_frame=source['geometry_frame'], geometry_wkb_hex=selected.wkb_hex, coverage_status=source['observation_coverage_status'])
    result = _tool('U01', 'VALID', region=region, layers=sorted(layers, key=lambda row: row['layer_name']), road_components=sorted(components, key=lambda row: row['component_id']), evidence_scope='SCOPE_GEOMETRY_ONLY', free_space_status='NOT_PROVIDED')
    if 'local_static_grid' in view.table_names:
        from copy import deepcopy
        import numpy as np
        grids = view.table('local_static_grid')
        base_grids = view.table('local_structure_grid')
        if len(grids) != 1 or len(base_grids) != 1:
            raise ValueError('local_static_grid requires exactly one grid row')
        grid, base_grid = (grids[0], base_grids[0])
        if set(grid) != {'region_id', 'geometry_frame', 'cell_m', 'origin_xy_m', 'base_mask_semantics', 'base_mask', 'observed_mask_semantics', 'observed_mask'} or grid['region_id'] != source['region_id'] or base_grid.get('region_id') != source['region_id'] or (grid['geometry_frame'] != source['geometry_frame']) or (grid['base_mask_semantics'] not in ('TG_FROZEN_COMPARISON_GROUND_CANDIDATE', 'NUPLAN_SOURCE_ROAD_MASK')):
            raise ValueError('local_static_grid belongs to another region or has invalid fields')
        if grid['cell_m'] != base_grid.get('cell_m') or grid['origin_xy_m'] != base_grid.get('origin_xy_m'):
            raise ValueError('local_static_grid frame differs from local_structure_grid')
        base_mask = np.asarray(base_grid.get('mask'))
        supplied_base = np.asarray(grid['base_mask'])
        observed = grid['observed_mask']
        if base_mask.dtype != np.bool_ or supplied_base.dtype != np.bool_ or base_mask.ndim != 2 or (not np.array_equal(base_mask, supplied_base)):
            raise ValueError('local_static_grid base mask differs from local_structure_grid')
        if observed is not None:
            observed = np.asarray(observed)
            if observed.dtype != np.bool_ or observed.shape != base_mask.shape or grid['observed_mask_semantics'] != 'TG_ALL_RAW_SEMANTIC_CELL_SUPPORT':
                raise ValueError('local static observation support must match the base grid')
            observed = observed.tolist()
        elif grid['observed_mask_semantics'] != 'NOT_PROVIDED_NUPLAN':
            raise ValueError('missing local static observation support has invalid semantics')
        result['static_grid'] = dict(geometry_frame=source['geometry_frame'], cell_m=grid['cell_m'], origin_xy_m=deepcopy(grid['origin_xy_m']), base_mask_semantics=grid['base_mask_semantics'], base_mask=base_mask.tolist(), observed_mask_semantics=grid['observed_mask_semantics'], observed_mask=observed)
    return result

def _entry(view, instance_id):
    source = _one(view, 'task_spec', instance_id=instance_id)
    _fields(source, ('instance_id', 'region_id', 'policy_version', 'cap_ns', 'start_kind', 'entry_road_status', 'reference_point', 'time_basis', 'entry_lower_ns', 'entry_upper_ns', 'first_x_m', 'first_y_m'), 'integrated task prior')
    if source['time_basis'] != 'RELATIVE_NS':
        raise ValueError('integrated entry requires relative nanoseconds')
    evidence = _one(view, 'entry_road_evidence', instance_id=instance_id)
    pose = _one(view, 'initial_pose', instance_id=instance_id)
    _fields(evidence, ('instance_id', 'region_id', 'entry_road_status', 'reason', 'entry_point_status', 'candidate_component_ids', 'first_component_ids', 'first', 'witness'), 'entry evidence')
    _fields(pose, {'instance_id', 'region_id', 'ordinal', 'relative_time_ns'} | LOCAL_FIELDS, 'initial pose')
    if pose['ordinal'] != 0 or pose['relative_time_ns'] != 0:
        raise ValueError('initial pose must be the first internal observation')
    if any((row['region_id'] != source['region_id'] for row in (evidence, pose))):
        raise ValueError('entry records belong to different regions')
    if source['entry_road_status'] != evidence['entry_road_status']:
        raise ValueError('entry road status differs from task prior')
    first = dict(relative_time_ns=0, x_m=_finite(source['first_x_m'], 'first_x_m'), y_m=_finite(source['first_y_m'], 'first_y_m'))
    if evidence['first'] is not None and evidence['first'] != first:
        raise ValueError('entry first point differs from task prior')
    witness = deepcopy(evidence['witness'])
    if witness is not None:
        _fields(witness, ('relative_time_ns', 'x_m', 'y_m'), 'entry witness')
        if _integer(witness['relative_time_ns'], 'witness time') >= 0:
            raise ValueError('entry witness must precede first internal sample')
        _finite(witness['x_m'], 'witness x')
        _finite(witness['y_m'], 'witness y')
    interval = [source['entry_lower_ns'], source['entry_upper_ns']]
    for value in interval:
        if value is not None:
            _integer(value, 'entry interval')
    observed = source['start_kind'] == 'OBSERVED_ENTRY'
    if source['start_kind'] not in ('OBSERVED_ENTRY', 'LEFT_CENSORED'):
        raise ValueError('unknown entry kind')
    if observed and (None in interval or interval[0] >= interval[1] or interval[1] != 0):
        raise ValueError('invalid observed entry interval')
    if not observed and interval != [None, 0]:
        raise ValueError('left-censored entry must retain its unknown lower bound')
    yaw = _finite(pose['yaw_rad'], 'initial yaw') if pose['yaw_status'] == 'KNOWN' else None
    return _tool('U03_ENTRY', 'VALID' if observed else 'INSUFFICIENT_EVIDENCE', () if observed else ('ENTRY_CENSORED',), instance_id=instance_id, region_id=source['region_id'], first_point=first, witness=witness, interval_ns=interval, entry_road_status=evidence['entry_road_status'], candidate_component_ids=deepcopy(evidence['candidate_component_ids']), first_component_ids=deepcopy(evidence['first_component_ids']), reference_point=source['reference_point'], initial_yaw_rad=yaw, initial_yaw_status='KNOWN' if yaw is not None else 'UNKNOWN')

def get_entry(view, *, instance_id):
    """U03 task-only entry prior; never consumes a DecisionView."""
    _view(view, TaskView)
    return _entry(view, instance_id)

def get_observed_exit(view, *, instance_id):
    """U03 decision-only exit interval and last internal sample, without interpolation."""
    _view(view, DecisionView)
    end = _one(view, 'observation_end', instance_id=instance_id)
    fields = ('instance_id', 'region_id', 'end_reason', 'right_censored', 'last_time_ns', 'deadline_ns', 'exit_lower_ns', 'exit_upper_ns', 'stop_lower_ns', 'stop_upper_ns', 'administrative_end_ns', 'outcome_status', 'observed_span_ns')
    if set(end) not in (set(fields), set(fields) | {'stop_witness'}):
        raise ValueError('observation end has missing or unexpected fields')
    if end['end_reason'] not in ('OBSERVED_EXIT', 'TIME_LIMIT', 'EXIT_OR_TIME_LIMIT', 'SOURCE_END', 'OBSERVATION_GAP'):
        raise ValueError('unknown observation end reason')
    rows = sorted((row for row in view.table('observed_trajectory') if row['instance_id'] == instance_id), key=lambda row: row['ordinal'])
    if not rows or [row['ordinal'] for row in rows] != list(range(len(rows))):
        raise ValueError('exit requires an ordered complete observation sample list')
    if any((row['region_id'] != end['region_id'] for row in rows)):
        raise ValueError('exit samples belong to another region')
    last = rows[-1]
    last_point = {key: last[key] for key in ('relative_time_ns', 'x_m', 'y_m')}
    _integer(last_point['relative_time_ns'], 'last relative time')
    _finite(last_point['x_m'], 'last x')
    _finite(last_point['y_m'], 'last y')
    if last_point['relative_time_ns'] != end['last_time_ns']:
        raise ValueError('last point differs from observation end')
    observed = end['end_reason'] == 'OBSERVED_EXIT'
    if type(end['right_censored']) is not bool or observed == end['right_censored']:
        raise ValueError('exit reason and censoring disagree')
    interval = [end['exit_lower_ns'], end['exit_upper_ns']]
    if observed:
        for value in interval:
            _integer(value, 'exit interval')
        if interval[0] != last_point['relative_time_ns'] or interval[0] >= interval[1]:
            raise ValueError('invalid observed exit interval')
    else:
        if interval[0] != last_point['relative_time_ns']:
            raise ValueError('censored exit must retain its observed lower bound')
        if end['end_reason'] == 'EXIT_OR_TIME_LIMIT':
            if _integer(interval[1], 'possible exit upper bound') <= end['deadline_ns']:
                raise ValueError('possible exit must be beyond the observation deadline')
        elif interval[1] is not None:
            raise ValueError('interrupted observation has no known exit upper bound')
    stop_witness = end.get('stop_witness')
    if stop_witness is not None:
        if not observed:
            raise ValueError('stop witness requires an observed exit')
        _fields(stop_witness, ('relative_time_ns', 'x_m', 'y_m'), 'stop witness')
        stop_time = _integer(stop_witness['relative_time_ns'], 'stop witness time')
        _finite(stop_witness['x_m'], 'stop witness x')
        _finite(stop_witness['y_m'], 'stop witness y')
        if stop_time <= last_point['relative_time_ns'] or stop_time != interval[1]:
            raise ValueError('stop witness must be the observed exit upper-bound sample')
    return _tool('U03_EXIT', 'VALID' if observed else 'INSUFFICIENT_EVIDENCE', () if observed else ('EXIT_NOT_OBSERVED',), instance_id=instance_id, region_id=end['region_id'], interval_ns=interval, last_point=last_point, stop_witness=deepcopy(stop_witness), end_reason=end['end_reason'], right_censored=end['right_censored'])

def get_entry_exit(view, *, instance_id):
    """U03 combined evidence is decision-only; it is not a Task prior."""
    _view(view, DecisionView)
    entry, end = (_entry(view, instance_id), get_observed_exit(view, instance_id=instance_id))
    status = 'VALID' if entry['status'] == end['status'] == 'VALID' else 'INSUFFICIENT_EVIDENCE'
    return _tool('U03', status, entry['reason_codes'] + end['reason_codes'], entry=entry, exit=end)

def _config(config):
    _fields(config, _CONFIG, 'alignment config')
    for key in _CONFIG:
        if key.endswith('_count'):
            _integer(config[key], key, positive=True)
        elif key == 'search_budget':
            if _integer(config[key], key) < 0:
                raise ValueError('search_budget must be nonnegative')
        else:
            value = _finite(config[key], key)
            if (key.endswith('_step') or key == 'boundary_step_m') and value <= 0:
                raise ValueError(key + ' must be positive')
            if (key.startswith('epsilon_') or 'tolerance' in key) and value < 0:
                raise ValueError(key + ' must be nonnegative')

def _boundary_integral(boundary, target, step):
    total = error = 0.0
    lines = [boundary] if boundary.geom_type in ('LineString', 'LinearRing') else boundary.geoms
    for line in lines:
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length == 0:
                continue
            count = math.ceil(length / step)
            segment = length / count
            for i in range(count):
                fraction = (i + 0.5) / count
                point = Point(a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
                total += segment * point.distance(target)
            error += length * segment / 4.0
    return (total, error)

def _boundary_distance(a, b, config):
    ab, ea = _boundary_integral(a.boundary, b.boundary, config['boundary_step_m'])
    ba, eb = _boundary_integral(b.boundary, a.boundary, config['boundary_step_m'])
    length = a.length + b.length
    return ((ab + ba) / length, (ea + eb) / length + config['distance_tolerance_m'])

def _region_result(result):
    if type(result) is not dict or set(result) not in (_REGION_RESULT, _REGION_RESULT | {'static_grid'}):
        raise ValueError('U01 result has missing or unexpected fields')
    if result['tool_id'] != 'U01' or result['definition_version'] != VERSION or result['status'] != 'VALID':
        raise ValueError('requires a valid U01 result of this version')
    _fields(result['region'], ('region_id', 'scope', 'geometry_frame', 'geometry_wkb_hex', 'coverage_status'), 'U01 region')
    if result['region']['scope'] != 'comparison':
        raise ValueError('Q1 alignment requires comparison scope')
    if result['region']['geometry_frame'] != 'STATIC_REGION_LOCAL_METRES':
        raise ValueError('unsupported U01 geometry frame')
    _polygon(result['region']['geometry_wkb_hex'])
    if not isinstance(result['layers'], list) or not isinstance(result['road_components'], list):
        raise ValueError('U01 layers and components must be record lists')
    for row in result['layers']:
        _fields(row, ('layer_name', 'geometry_wkb_hex', 'status', 'physical_role', 'layer_scope', 'coverage_status'), 'U01 layer')
        if row['layer_scope'] != 'COMPARISON':
            raise ValueError('U01 layer scope differs from comparison region')
        if row['geometry_wkb_hex'] is not None:
            geometry = _geometry(row['geometry_wkb_hex'])
            expected = 'EMPTY' if geometry.is_empty else 'POSITIVE_AREA' if geometry.area > 0 else 'BOUNDARY_TOUCH'
            if row['status'] != expected:
                raise ValueError('U01 layer status differs from its geometry')
        elif row['status'] != 'NOT_PROVIDED':
            raise ValueError('unknown geometry cannot claim known or empty support')
    if len({row['layer_name'] for row in result['layers']}) != len(result['layers']):
        raise ValueError('duplicate U01 layer names')
    for row in result['road_components']:
        _fields(row, ('component_id', 'component_wkb_hex', 'boundary_contact_wkb_hex', 'boundary_contact_status'), 'U01 road component')
        _polygon(row['component_wkb_hex'])
    if len({row['component_id'] for row in result['road_components']}) != len(result['road_components']):
        raise ValueError('duplicate U01 component IDs')
    if 'static_grid' in result:
        import numpy as np
        grid = result['static_grid']
        _fields(grid, ('geometry_frame', 'cell_m', 'origin_xy_m', 'base_mask_semantics', 'base_mask', 'observed_mask_semantics', 'observed_mask'), 'U01 static grid')
        if grid['geometry_frame'] != 'STATIC_REGION_LOCAL_METRES':
            raise ValueError('unsupported U01 static-grid frame')
        if type(grid['cell_m']) not in (float, int) or grid['cell_m'] <= 0 or type(grid['origin_xy_m']) is not list or (len(grid['origin_xy_m']) != 2) or any((type(value) not in (float, int) or not math.isfinite(value) for value in grid['origin_xy_m'])):
            raise ValueError('invalid U01 static-grid coordinates')
        base = np.asarray(grid['base_mask'])
        observed = grid['observed_mask']
        if grid['base_mask_semantics'] not in ('TG_FROZEN_COMPARISON_GROUND_CANDIDATE', 'NUPLAN_SOURCE_ROAD_MASK') or base.dtype != np.bool_ or base.ndim != 2:
            raise ValueError('invalid U01 static-grid base mask')
        if observed is None:
            if grid['observed_mask_semantics'] != 'NOT_PROVIDED_NUPLAN':
                raise ValueError('U01 static-grid observation support is missing')
        else:
            observed = np.asarray(observed)
            if observed.dtype != np.bool_ or observed.shape != base.shape or grid['observed_mask_semantics'] != 'TG_ALL_RAW_SEMANTIC_CELL_SUPPORT':
                raise ValueError('invalid U01 static-grid observation support')
    return {row['layer_name']: row for row in result['layers']}

def align_region_structures(region_a, region_b, *, config, layer_pairs):
    """U02: finite 2D SE(2), mapping B into A, with bounded midpoint integration.

    Semantic roles are declarations, not inferred from source labels. Scope
    geometry remains descriptive and cannot certify physical road identity.
    """
    layers_a, layers_b = (_region_result(region_a), _region_result(region_b))
    input_a, input_b = (deepcopy(region_a), deepcopy(region_b))
    input_a.pop('static_grid', None)
    input_b.pop('static_grid', None)
    base = dict(input_a=input_a, input_b=input_b, region_a=region_a['region']['region_id'], region_b=region_b['region']['region_id'], transform_direction='B_TO_A', evidence_scope='SCOPE_GEOMETRY_ONLY', parameters_used=deepcopy(config), layer_pairs=deepcopy(layer_pairs), candidate_count=0, evaluated_candidates=[], transforms=[])
    if config is None:
        return _tool('U02', 'NOT_CALIBRATED', ('CONFIG_NOT_PROVIDED',), **base)
    _config(config)
    base['candidate_count'] = config['theta_count'] * config['x_count'] * config['y_count']
    if not isinstance(layer_pairs, list) or not layer_pairs:
        raise ValueError('layer_pairs must be a nonempty predeclared list')
    pairs = []
    for pair in layer_pairs:
        _fields(pair, ('layer_a', 'layer_b', 'semantic_role'), 'layer pair')
        a_name, b_name, role = (pair['layer_a'], pair['layer_b'], pair['semantic_role'])
        if a_name not in layers_a or b_name not in layers_b:
            raise ValueError('declared layer is not present in U01')
        if role in ('COMPARISON_SCOPE', 'SOURCE_ROAD_SCOPE'):
            expected = 'comparison_scope' if role == 'COMPARISON_SCOPE' else 'source_road_scope'
            if a_name != expected or b_name != expected:
                raise ValueError('scope role does not permit other source layers')
        elif role != 'BASIC_ROAD_STRUCTURE':
            raise ValueError('unsupported Q1 semantic role')
        a, b = (layers_a[a_name], layers_b[b_name])
        if a['geometry_wkb_hex'] is None or b['geometry_wkb_hex'] is None or (role == 'BASIC_ROAD_STRUCTURE' and (a['physical_role'] != role or b['physical_role'] != role)):
            return _tool('U02', 'INSUFFICIENT_EVIDENCE', ('LAYER_OR_SEMANTICS_NOT_PROVIDED',), **base)
        pairs.append((_polygon(a['geometry_wkb_hex']), _polygon(b['geometry_wkb_hex'])))
    for index in product(range(config['theta_count']), range(config['x_count']), range(config['y_count'])):
        if len(base['evaluated_candidates']) >= config['search_budget']:
            return _tool('U02', 'INSUFFICIENT_EVIDENCE', ('SEARCH_LIMITED',), **base)
        theta = _finite(config['theta_origin'] + index[0] * config['theta_step'], 'grid angle')
        x = _finite(config['x_origin'] + index[1] * config['x_step'], 'grid x')
        y = _finite(config['y_origin'] + index[2] * config['y_step'], 'grid y')
        c, s = (math.cos(theta), math.sin(theta))
        coverages, coverage_errors, distances, distance_bounds = ([], [], [], [])
        for a, b in pairs:
            moved = affinity.affine_transform(b, [c, -s, s, c, x, y])
            if a.is_empty and moved.is_empty:
                continue
            if a.is_empty or moved.is_empty:
                coverages.append(0.0)
                coverage_errors.append(0.0)
                continue
            area = a.intersection(moved).area
            coverage = min(area / a.area, area / moved.area)
            coverages.append(coverage)
            coverage_errors.append(config['area_tolerance_m2'] / min(a.area, moved.area))
            distance, bound = _boundary_distance(a, moved, config)
            distances.append(distance)
            distance_bounds.append(bound)
        coverage = min(coverages) if coverages else None
        ce = max(coverage_errors) if coverage_errors else 0.0
        distance = max(distances) if distances else None
        de = max(distance_bounds) if distance_bounds else 0.0
        row = dict(grid_index=list(index), theta_rad=theta, x_m=x, y_m=y, coverage=_measurement('U02.coverage', coverage, 'RATIO', status='VALID' if coverage is not None else 'NOT_DEFINED', parameters=config), coverage_error_bound=ce, boundary_distance_m=_measurement('U02.boundary_distance', distance, 'METER', status='VALID' if distance is not None else 'NOT_DEFINED', parameters=config), boundary_error_bound_m=de)
        base['evaluated_candidates'].append(row)
    candidates = base['evaluated_candidates']
    if candidates[0]['coverage']['value'] is not None:
        best_lower = max((row['coverage']['value'] - row['coverage_error_bound'] for row in candidates))
        candidates = [row for row in candidates if row['coverage']['value'] + row['coverage_error_bound'] >= best_lower - config['epsilon_C']]
    if candidates[0]['boundary_distance_m']['value'] is not None:
        best_upper = min((row['boundary_distance_m']['value'] + row['boundary_error_bound_m'] for row in candidates))
        candidates = [row for row in candidates if row['boundary_distance_m']['value'] - row['boundary_error_bound_m'] <= best_upper + config['epsilon_D']]
    base['transforms'] = deepcopy(candidates)
    unconstrained = all((a.is_empty or b.is_empty for a, b in pairs))
    ambiguous = unconstrained or len(candidates) != 1
    return _tool('U02', 'AMBIGUOUS' if ambiguous else 'VALID', ('NO_OR_NONUNIQUE_GEOMETRIC_ALIGNMENT',) if ambiguous else (), **base)

class PosthocEvaluationContext:
    """Immutable, decision-derived endpoints accepted only by GT_POSTHOC."""
    __slots__ = ('_snapshot',)

    def __init__(self, decision_a, decision_b, *, instance_a, instance_b):
        ends = [get_observed_exit(decision_a, instance_id=instance_a), get_observed_exit(decision_b, instance_id=instance_b)]
        object.__setattr__(self, '_snapshot', json.dumps(ends, allow_nan=False).encode('utf-8'))

    def __setattr__(self, name, value):
        raise AttributeError('posthoc context is immutable')

    def __delattr__(self, name):
        raise AttributeError('posthoc context is immutable')

    @classmethod
    def from_decisions(cls, decision_a, decision_b, *, instance_a, instance_b):
        return cls(decision_a, decision_b, instance_a=instance_a, instance_b=instance_b)

def _context_structure(view):
    return get_local_region(StructureView({name: view.table(name) for name in ('region_spec', 'structure_layers', 'road_components')}), scope='comparison')

def _point(point, transform):
    c, s = (math.cos(transform['theta_rad']), math.sin(transform['theta_rad']))
    return (c * point['x_m'] - s * point['y_m'] + transform['x_m'], s * point['x_m'] + c * point['y_m'] + transform['y_m'])

def _distance(a, b, transform):
    x, y = _point(b, transform)
    return math.hypot(a['x_m'] - x, a['y_m'] - y)

def _entry_direction(entry):
    witness = entry['witness']
    if witness is None:
        return None
    dx, dy = (entry['first_point']['x_m'] - witness['x_m'], entry['first_point']['y_m'] - witness['y_m'])
    return math.atan2(dy, dx) if dx != 0 or dy != 0 else None

def _exit_direction(exit):
    witness = exit['stop_witness']
    if witness is None:
        return None
    dx, dy = (witness['x_m'] - exit['last_point']['x_m'], witness['y_m'] - exit['last_point']['y_m'])
    return math.atan2(dy, dx) if dx != 0 or dy != 0 else None

def compare_local_tasks(task_a, task_b, *, instance_a, instance_b, alignment, profile='ENTRY', posthoc=None):
    """U04: all retained alignments, separate components, no Q1 truth label.

    Entry direction is the witnessed entering displacement, not body yaw.
    GT exit position is the last internal sample, not an interpolated crossing.
    """
    _view(task_a, TaskContextView)
    _view(task_b, TaskContextView)
    if profile not in ('ENTRY', 'GT_POSTHOC'):
        raise ValueError('unsupported task comparison profile')
    if profile == 'ENTRY' and posthoc is not None:
        raise ValueError('posthoc evidence is forbidden in ENTRY profile')
    if profile == 'GT_POSTHOC' and type(posthoc) is not PosthocEvaluationContext:
        raise TypeError('GT_POSTHOC requires PosthocEvaluationContext')
    if type(alignment) is not _AlignmentResult:
        raise ValueError('requires the immutable U02 result, not an edited or deserialized mapping')
    alignment = alignment.as_dict()
    _fields(alignment, _ALIGNMENT_RESULT, 'U02 result')
    if alignment['tool_id'] != 'U02' or alignment['definition_version'] != VERSION:
        raise ValueError('requires a U02 alignment of this version')
    if alignment['transform_direction'] != 'B_TO_A':
        raise ValueError('alignment must map B to A')
    for view, key in [(task_a, 'a'), (task_b, 'b')]:
        static = _context_structure(view)
        if static != alignment['input_' + key] or static['region']['region_id'] != alignment['region_' + key]:
            raise ValueError('alignment does not bind to these static task contexts')
    a, b = (_entry(task_a, instance_a), _entry(task_b, instance_b))
    ends = None
    if posthoc is not None:
        ends = json.loads(object.__getattribute__(posthoc, '_snapshot'))
        for end, entry in zip(ends, (a, b)):
            if end['instance_id'] != entry['instance_id'] or end['region_id'] != entry['region_id']:
                raise ValueError('posthoc evidence belongs to different tasks')
    base = dict(profile=profile, evidence_scope='SCOPE_GEOMETRY_ONLY', terrain_status='NOT_PROVIDED', basic_road_topology_status='NOT_PROVIDED', comparisons=[])
    if alignment['status'] not in ('VALID', 'AMBIGUOUS') or not alignment['transforms']:
        return _tool('U04', 'INSUFFICIENT_EVIDENCE', ('STRUCTURE_ALIGNMENT_UNAVAILABLE',), **base)
    parameters = alignment['parameters_used']
    _config(parameters)
    components_a = {r['component_id']: r for r in alignment['input_a']['road_components']}
    components_b = {r['component_id']: r for r in alignment['input_b']['road_components']}
    for transform in alignment['transforms']:
        i, j, k = transform['grid_index']
        for index, axis in zip((i, j, k), ('theta', 'x', 'y')):
            if type(index) is not int or not 0 <= index < parameters[axis + '_count']:
                raise ValueError('invalid alignment grid index')
            field = 'theta_rad' if axis == 'theta' else axis + '_m'
            if transform[field] != parameters[axis + '_origin'] + index * parameters[axis + '_step']:
                raise ValueError('transform differs from declared static grid')
        direction_a, direction_b = (_entry_direction(a), _entry_direction(b))
        direction = None if direction_a is None or direction_b is None else abs(float(wrap_to_pi(direction_a - direction_b - transform['theta_rad'])))
        distance = _distance(a['first_point'], b['first_point'], transform)
        exit_distance, exit_status, exit_reasons = (None, 'NOT_REQUESTED', ())
        exit_direction = None
        exit_direction_status = 'NOT_REQUESTED'
        exit_direction_reasons = ()
        if ends is not None:
            if all((end['status'] == 'VALID' for end in ends)):
                exit_distance = _distance(ends[0]['last_point'], ends[1]['last_point'], transform)
                exit_status = 'VALID'
                directions = [_exit_direction(end) for end in ends]
                if all((direction is not None for direction in directions)):
                    exit_direction = abs(float(wrap_to_pi(directions[0] - directions[1] - transform['theta_rad'])))
                    exit_direction_status = 'VALID'
                else:
                    exit_direction_status = 'INSUFFICIENT_EVIDENCE'
                    exit_direction_reasons = ('EXIT_MOTION_WITNESS_NOT_PROVIDED' if any((end['stop_witness'] is None for end in ends)) else 'EXIT_MOTION_NOT_WITNESSED',)
            else:
                exit_status, exit_reasons = ('INSUFFICIENT_EVIDENCE', ('EXIT_NOT_OBSERVED',))
                exit_direction_status = 'INSUFFICIENT_EVIDENCE'
                exit_direction_reasons = ('EXIT_NOT_OBSERVED',)
        candidate_pairs = []
        for id_a, id_b in product(a['candidate_component_ids'], b['candidate_component_ids']):
            if id_a not in components_a or id_b not in components_b:
                raise ValueError('entry candidate refers to absent static component')
            ca, cb = (_polygon(components_a[id_a]['component_wkb_hex']), _polygon(components_b[id_b]['component_wkb_hex']))
            c, s = (math.cos(transform['theta_rad']), math.sin(transform['theta_rad']))
            moved = affinity.affine_transform(cb, [c, -s, s, c, transform['x_m'], transform['y_m']])
            candidate_pairs.append(dict(component_a=id_a, component_b=id_b, geometry_distance_m=_measurement('U04.candidate_geometry_distance', ca.distance(moved), 'METER', scope='ENTRY_OBSERVATION'), relation='GEOMETRY_ONLY_NOT_ROAD_IDENTITY'))
        base['comparisons'].append(dict(grid_index=deepcopy(transform['grid_index']), entry_road_status_a=a['entry_road_status'], entry_road_status_b=b['entry_road_status'], candidate_support_status='VALID' if candidate_pairs else 'INSUFFICIENT_EVIDENCE', candidate_reason_codes=[] if candidate_pairs else ['ENTRY_CANDIDATE_GEOMETRY_NOT_PROVIDED'], reference_point_a=a['reference_point'], reference_point_b=b['reference_point'], entry_distance_m=_measurement('U04.entry_distance', distance, 'METER', scope='ENTRY_OBSERVATION'), entry_direction_difference_rad=_measurement('U04.entry_direction_difference', direction, 'RADIAN', status='VALID' if direction is not None else 'INSUFFICIENT_EVIDENCE', reasons=() if direction is not None else ('ENTRY_MOTION_NOT_WITNESSED',), scope='ENTRY_OBSERVATION'), observed_exit_distance_m=_measurement('U04.observed_exit_distance', exit_distance, 'METER', status=exit_status, reasons=exit_reasons, scope='POSTHOC_OBSERVED_EXIT'), observed_exit_direction_difference_rad=_measurement('U04.observed_exit_direction_difference', exit_direction, 'RADIAN', status=exit_direction_status, reasons=exit_direction_reasons, scope='POSTHOC_OBSERVED_EXIT'), candidate_pairs=candidate_pairs))
    status = alignment['status']
    reasons = list(alignment['reason_codes'])
    if a['status'] != 'VALID' or b['status'] != 'VALID':
        status = 'INSUFFICIENT_EVIDENCE'
        reasons.append('ENTRY_NOT_FULLY_OBSERVED')
    return _tool('U04', status, reasons, **base)
