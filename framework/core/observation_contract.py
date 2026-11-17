from collections import defaultdict
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import numpy as np
from shapely import wkb
from shapely.geometry import box
from shapely import points, covers
from .region_experiences import CAP_NS, POLICY_VERSION
from .observation_projection import _planar_road_part
from .serialization import deterministic_json_bytes
FORMAT_VERSION = 'region-observation-tasks/0.1.0'
TABLE_FIELDS = {'regions': set('region_id center_x center_y heading_rad size_m observation_wkb_hex structure_wkb_hex map_role structure_status anchor_type'.split()), 'instances': set('instance_id region_id subject_id time_basis reference_point first_time_ns last_time_ns deadline_ns entry_lower_ns entry_upper_ns exit_lower_ns exit_upper_ns stop_lower_ns stop_upper_ns administrative_end_ns start_kind end_reason right_censored entry_road_status outcome_status observed_span_ns'.split()), 'samples': set('instance_id ordinal source_sample_index source_time_ns relative_time_ns x_m y_m source_x_m source_y_m road_relation square_relation source_row source_sample_id'.split()), 'events': set('instance_id event_type lower_relative_ns upper_relative_ns censored'.split()), 'bindings': set('instance_id source_root input_path input_sha256 episode region_source'.split()), 'backgrounds': set('instance_id kind status binding_json'.split())}
TOP_FIELDS = set(TABLE_FIELDS) | {'format_version', 'policy_version', 'cap_ns', 'status'}

def _fields(row, expected, name):
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(name + ' has missing or unexpected fields')

def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be nonempty text')

def _int(value, name, nullable=False):
    if nullable and value is None:
        return
    if type(value) is not int or not -2 ** 63 <= value < 2 ** 63:
        raise ValueError(name + ' requires an int64 integer, not float/bool')

def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise ValueError(name + ' requires a finite number')

def _geometry(value, name, allow_collection=False):
    _text(value, name)
    try:
        g = wkb.loads(bytes.fromhex(value))
    except Exception as exc:
        raise ValueError(name + ' is invalid WKB') from exc
    allowed = ('Polygon', 'MultiPolygon', 'GeometryCollection') if allow_collection else ('Polygon', 'MultiPolygon')
    if g.geom_type not in allowed or g.is_empty or (not g.is_valid) or g.has_z or (g.area <= 0):
        raise ValueError(name + ' requires valid nonempty planar polygons')
    if allow_collection and (not _planar_road_part(g)):
        raise ValueError(name + ' contains unsupported road geometry members')
    return g

def _unique(rows, key, name):
    result = {}
    for row in rows:
        _text(row[key], name)
        if row[key] in result:
            raise ValueError('duplicate ' + name)
        result[row[key]] = row
    return result

def _strict_json(text):
    try:
        return json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid background binding JSON') from exc

def validate_observation_bundle(bundle):
    """Validate closed fields, evidence consistency and local coordinates.

    Authenticity of external point/background sources is separately verified by
    the materializer/store. Hashes are not a substitute for these semantic checks.
    """
    _fields(bundle, TOP_FIELDS, 'observation bundle')
    if bundle['format_version'] != FORMAT_VERSION or bundle['policy_version'] != POLICY_VERSION or bundle['status'] != 'DEVELOPMENT':
        raise ValueError('unsupported observation version/policy/publication state')
    _int(bundle['cap_ns'], 'cap_ns')
    if bundle['cap_ns'] != CAP_NS:
        raise ValueError('observation cap differs')
    for table, fields in TABLE_FIELDS.items():
        if not isinstance(bundle[table], list):
            raise ValueError(table + ' must be a list')
        for row in bundle[table]:
            _fields(row, fields, table)
    regions = _unique(bundle['regions'], 'region_id', 'region identity')
    instances = _unique(bundle['instances'], 'instance_id', 'instance identity')
    geometries = {}
    for rid, r in regions.items():
        for field in ['center_x', 'center_y', 'heading_rad', 'size_m']:
            _number(r[field], field)
        if r['size_m'] not in (25.0, 50.0, 75.0, 100.0):
            raise ValueError('unsupported square size')
        if r['anchor_type'] != 'STATIC_REGION_FRAME' or r['structure_status'] != 'ROAD_SCOPE_ONLY' or r['map_role'] not in ('SOURCE_NATIVE', 'DERIVED_ROAD_SCOPE'):
            raise ValueError('unsupported region frame/structure evidence')
        square = _geometry(r['observation_wkb_hex'], 'observation geometry')
        road = _geometry(r['structure_wkb_hex'], 'structure geometry', allow_collection=True)
        h = r['size_m'] / 2
        if square.symmetric_difference(box(-h, -h, h, h)).area > 0.0001:
            raise ValueError('local square differs from static size/frame')
        if not square.buffer(1e-06).covers(road):
            raise ValueError('road structure escapes observation square')
        geometries[rid] = square
    grouped = defaultdict(list)
    for row in bundle['samples']:
        if row['instance_id'] not in instances:
            raise ValueError('sample has unknown instance')
        for field in ['ordinal', 'source_sample_index', 'source_row', 'source_time_ns', 'relative_time_ns', 'road_relation', 'square_relation']:
            _int(row[field], field)
        if any((row[f] < 0 for f in ['ordinal', 'source_sample_index', 'source_row', 'relative_time_ns'])):
            raise ValueError('negative index or local time')
        if row['road_relation'] not in (-1, 0, 1, 2) or row['square_relation'] not in (1, 2):
            raise ValueError('invalid sample relation')
        for field in ['x_m', 'y_m', 'source_x_m', 'source_y_m']:
            _number(row[field], field)
        _text(row['source_sample_id'], 'source sample identity')
        grouped[row['instance_id']].append(row)
    for iid, i in instances.items():
        if i['region_id'] not in regions:
            raise ValueError('instance has unknown region')
        for field in ['subject_id', 'reference_point']:
            _text(i[field], field)
        if i['time_basis'] not in ('ABSOLUTE_NS', 'DERIVED_RELATIVE_NS_FROM_SECONDS'):
            raise ValueError('unsupported clock evidence')
        for field in [f for f in TABLE_FIELDS['instances'] if f.endswith('_ns')]:
            _int(i[field], field, field in {'entry_lower_ns', 'exit_upper_ns', 'administrative_end_ns'})
        first, last, deadline = (i['first_time_ns'], i['last_time_ns'], i['deadline_ns'])
        if not first <= last <= deadline or deadline != first + CAP_NS or i['observed_span_ns'] != last - first:
            raise ValueError('inconsistent observation clock/span/deadline')
        if i['entry_upper_ns'] != first or i['exit_lower_ns'] != last or (i['entry_lower_ns'] is not None and i['entry_lower_ns'] >= first):
            raise ValueError('invalid entry/exit evidence bounds')
        if i['start_kind'] not in ('OBSERVED_ENTRY', 'LEFT_CENSORED') or (i['entry_lower_ns'] is None) != (i['start_kind'] == 'LEFT_CENSORED'):
            raise ValueError('entry censoring differs')
        if type(i['right_censored']) is not bool or i['right_censored'] != (i['end_reason'] != 'OBSERVED_EXIT'):
            raise ValueError('exit censoring differs')
        if i['entry_road_status'] != 'UNKNOWN' or i['outcome_status'] != 'NOT_DEFINED_BY_OBSERVATION':
            raise ValueError('unsupported inferred entry road or outcome')
        reason = i['end_reason']
        upper = i['exit_upper_ns']
        admin = i['administrative_end_ns']
        if reason == 'OBSERVED_EXIT':
            if upper is None or not last < upper <= deadline or admin is not None:
                raise ValueError('invalid observed exit')
            stop = (last, upper)
        elif reason == 'TIME_LIMIT':
            if upper is not None or admin != deadline:
                raise ValueError('invalid capped observation')
            stop = (deadline, deadline)
        elif reason == 'EXIT_OR_TIME_LIMIT':
            if upper is None or upper <= deadline or admin != deadline:
                raise ValueError('invalid ambiguous exit/cap')
            stop = (last, deadline)
        elif reason in ('SOURCE_END', 'OBSERVATION_GAP'):
            if upper is not None or admin is not None:
                raise ValueError('invalid interrupted observation')
            stop = (last, last)
        else:
            raise ValueError('unknown end reason')
        if (i['stop_lower_ns'], i['stop_upper_ns']) != stop:
            raise ValueError('observation stop differs')
        samples = sorted(grouped[iid], key=lambda s: s['ordinal'])
        if not samples or [s['ordinal'] for s in samples] != list(range(len(samples))):
            raise ValueError('missing or duplicate sample ordinals')
        times = [s['source_time_ns'] for s in samples]
        indices = [s['source_sample_index'] for s in samples]
        if times[0] != first or times[-1] != last or any((b <= a for a, b in zip(times, times[1:]))) or any((b != a + 1 for a, b in zip(indices, indices[1:]))) or (len({s['source_row'] for s in samples}) != len(samples)) or (len({s['source_sample_id'] for s in samples}) != len(samples)) or (samples[0]['square_relation'] != 1):
            raise ValueError('invalid sample sequence')
        if any((s['relative_time_ns'] != s['source_time_ns'] - first for s in samples)):
            raise ValueError('relative time differs from integer clock')
        r = regions[i['region_id']]
        c, s = (math.cos(r['heading_rad']), math.sin(r['heading_rad']))
        source = np.array([[v['source_x_m'] - r['center_x'], v['source_y_m'] - r['center_y']] for v in samples])
        local = np.array([[v['x_m'], v['y_m']] for v in samples])
        expected = source @ np.array([[c, -s], [s, c]])
        if not np.allclose(expected, local, rtol=0.0, atol=1e-07):
            raise ValueError('task local coordinate differs from static frame')
        if not np.all(covers(geometries[i['region_id']].buffer(1e-06), points(local))):
            raise ValueError('retained sample outside observation square')
    events = defaultdict(dict)
    for e in bundle['events']:
        iid = e['instance_id']
        if iid not in instances or e['event_type'] not in ('OBSERVATION_START', 'OBSERVATION_END') or e['event_type'] in events[iid]:
            raise ValueError('invalid event identity')
        _int(e['lower_relative_ns'], 'event lower', True)
        _int(e['upper_relative_ns'], 'event upper')
        if type(e['censored']) is not bool:
            raise ValueError('event censor must be bool')
        events[iid][e['event_type']] = e
    for iid, i in instances.items():
        first = i['first_time_ns']
        expected = [('OBSERVATION_START', None if i['entry_lower_ns'] is None else i['entry_lower_ns'] - first, 0, i['start_kind'] == 'LEFT_CENSORED'), ('OBSERVATION_END', i['stop_lower_ns'] - first, i['stop_upper_ns'] - first, i['right_censored'])]
        if set(events[iid]) != {v[0] for v in expected}:
            raise ValueError('missing observation events')
        for kind, lo, hi, censored in expected:
            e = events[iid][kind]
            if (e['lower_relative_ns'], e['upper_relative_ns'], e['censored']) != (lo, hi, censored):
                raise ValueError('event differs from instance evidence')
    for table in ['bindings', 'backgrounds']:
        bound = _unique(bundle[table], 'instance_id', table + ' instance')
        if set(bound) != set(instances):
            raise ValueError(table + ' coverage differs')
    for binding in bundle['bindings']:
        _text(binding['source_root'], 'source root')
        _text(binding['input_path'], 'point source')
        p = binding['input_path']
        if PureWindowsPath(p).drive or PurePosixPath(p).is_absolute() or '..' in p.replace('\\', '/').split('/'):
            raise ValueError('unsafe point source path')
        sha = binding['input_sha256']
        if not isinstance(sha, str) or len(sha) != 64 or any((c not in '0123456789abcdef' for c in sha)):
            raise ValueError('invalid point source digest')
        if not isinstance(binding['episode'], dict) or not isinstance(binding['region_source'], dict):
            raise ValueError('missing source evidence')
        deterministic_json_bytes(binding['episode'])
        deterministic_json_bytes(binding['region_source'])
    statuses = {'NONE': 'NOT_AVAILABLE', 'NUPLAN_ALL_NATIVE_CATEGORIES_FRAME_REFERENCE': 'SOURCE_REFERENCE_NOT_SPATIALLY_QUERIED', 'TG_ALL_CATEGORY_STATIC_SQUARE_SUMMARY': 'STATIC_CELLS_BOUNDARY_APPROXIMATE'}
    for b in bundle['backgrounds']:
        if b['kind'] not in statuses or b['status'] != statuses[b['kind']]:
            raise ValueError('unsupported background evidence state')
        if not isinstance(_strict_json(b['binding_json']), dict):
            raise ValueError('background binding must be object')
    return dict(status='PASS_DEVELOPMENT_INTERFACE_NOT_CANONICAL', region_count=len(regions), instance_count=len(instances), sample_count=len(bundle['samples']), event_count=len(bundle['events']))
