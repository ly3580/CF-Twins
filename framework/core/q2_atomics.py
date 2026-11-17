from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from .observation_views import DecisionView, TaskView
from .q1_atomics import _AlignmentResult, _finite, _integer, _one, _fields, get_entry, get_observed_exit
from .route_pair_direction import measure_route_pair_v2
from .route_shape_atomics import _position_grid, measure_route_shape
from .route_window_regression import _ESTIMATOR, measure_route_pair_windowed, measure_route_shape_windowed
IMPLEMENTATION_VERSION = 'q2-atomics/0.3.0'
_SHAPE_UNITS = {'length_m': 'METER', 'chord_m': 'METER', 'excess_length_m': 'METER', 'chord_ratio': 'RATIO', 'lateral_rms_m': 'METER', 'lateral_max_m': 'METER', 'mean_abs_turn_rate_rad_per_m': 'RADIAN_PER_METER', 'max_abs_turn_rad': 'RADIAN', 'turn_sign_changes': 'COUNT'}
_TRAJ_SCHEMA = 'q2-trajectory-atomics/0.1.0'
_GT_TRAJ_SCHEMA = 'q2-gt-display-trajectory/0.1.0'
_ALIGNED_SCHEMA = 'q2-aligned-trajectories/0.1.0'
_GT_ALIGNED_SCHEMA = 'q2-gt-display-aligned/0.1.0'
_DIFF_SCHEMA = 'q2-trajectory-discrepancy/0.1.0'
_GT_DIFF_SCHEMA = 'q2-gt-display-discrepancy/0.2.0'
_COMPARE_SCHEMA = 'q2-trajectory-comparison/0.1.0'
_TRAJ_FIELDS = frozenset(('tool_id', 'definition_version', 'instance_id', 'region_id', 'reference_point', 'coordinate_frame', 'continuous', 'points', 'observation_start', 'observation_end', 'pose_status'))

def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, Real) or (not math.isfinite(v)):
        raise ValueError(name + ' must be finite and not bool')
    return float(v)

def get_local_trajectory(view, *, instance_id, continuous, profile='OBSERVED'):
    """U05: copy the complete allowlisted local observation sequence."""
    if type(view) is not DecisionView:
        raise TypeError('requires DecisionView')
    if type(continuous) is not bool:
        raise ValueError('continuous must be an explicit bool')
    if profile not in ('OBSERVED', 'GT_DISPLAY'):
        raise ValueError('profile must be OBSERVED or GT_DISPLAY')
    if profile == 'GT_DISPLAY' and continuous:
        raise ValueError('GT_DISPLAY cannot claim source continuity')
    spec = _one(view, 'task_spec', instance_id=instance_id)
    _fields(spec, ('instance_id', 'region_id', 'policy_version', 'cap_ns', 'start_kind', 'entry_road_status', 'reference_point', 'time_basis', 'entry_lower_ns', 'entry_upper_ns', 'first_x_m', 'first_y_m'), 'integrated task prior')
    if not isinstance(spec['region_id'], str) or not spec['region_id'] or (not isinstance(spec['reference_point'], str)) or (not spec['reference_point']):
        raise ValueError('task prior lacks region or reference point')
    region = _one(view, 'region_spec')
    if region.get('region_id') != spec['region_id'] or region.get('geometry_frame') != 'STATIC_REGION_LOCAL_METRES':
        raise ValueError('DecisionView region or coordinate frame differs')
    rows = [r for r in view.table('observed_trajectory') if r.get('instance_id') == instance_id]
    rows.sort(key=lambda r: r.get('ordinal', -1))
    if not rows or any((type(r.get('ordinal')) is not int for r in rows)):
        raise ValueError('trajectory requires complete integer ordinals')
    if [r['ordinal'] for r in rows] != list(range(len(rows))):
        raise ValueError('trajectory ordinals must be unique and complete from zero')
    evidence = None
    if profile == 'GT_DISPLAY':
        evidence = _one(view, 'gt_display_evidence', instance_id=instance_id)
        _fields(evidence, ('instance_id', 'region_id', 'evidence_scope', 'point_count', 'route_xy_sha256'), 'GT display evidence')
        if evidence['region_id'] != spec['region_id'] or evidence['evidence_scope'] != 'SOURCE_BOUND_GT_DISPLAY_POLYLINE':
            raise ValueError('GT display evidence region or scope differs')
        if type(evidence['point_count']) is not int or evidence['point_count'] != len(rows):
            raise ValueError('GT display evidence point count differs')
    points, previous_time = ([], None)
    known_times = []
    known_indices = []
    for row in rows:
        if row.get('region_id') != spec['region_id']:
            raise ValueError('trajectory samples belong to different regions')
        raw_time = row.get('relative_time_ns')
        if profile == 'GT_DISPLAY' and raw_time is None:
            if not points:
                raise ValueError('GT display first sample clock must be known at zero')
            t = None
        else:
            t = _integer(raw_time, 'relative_time_ns')
            if t < 0 or (not points and t != 0):
                raise ValueError('relative trajectory time must start at zero and remain nonnegative')
            if previous_time is not None and t <= previous_time:
                raise ValueError('trajectory time must strictly increase')
            previous_time = t
            known_times.append(t)
            known_indices.append(row['ordinal'])
        points.append(dict(ordinal=row['ordinal'], relative_time_ns=t, x_m=_num(row.get('x_m'), 'x_m'), y_m=_num(row.get('y_m'), 'y_m')))
    task_view = TaskView({name: view.table(name) for name in ('task_spec', 'entry_road_evidence', 'initial_pose', 'task_events') if name in view.table_names})
    start = get_entry(task_view, instance_id=instance_id)
    end = get_observed_exit(view, instance_id=instance_id)
    if profile == 'GT_DISPLAY':
        if not known_times or known_times[0] != 0 or known_times[-1] != end['last_point']['relative_time_ns']:
            raise ValueError('GT display clock endpoints differ from witnessed samples')
        digest = hashlib.sha256(json.dumps([[float(p['x_m']), float(p['y_m'])] for p in points], separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        if not isinstance(evidence['route_xy_sha256'], str) or evidence['route_xy_sha256'] != digest:
            raise ValueError('GT display route digest differs')
        if known_indices == [0, len(points) - 1] and len(points) > 2:
            time_status = 'KNOWN_ENDPOINT_CLOCKS_ONLY'
        elif len(known_times) < len(points):
            time_status = 'PARTIAL_SOURCE_CLOCKS'
        else:
            time_status = 'SOURCE_CLOCKS_COMPLETE'
        return dict(tool_id='U05', definition_version=_GT_TRAJ_SCHEMA, profile='GT_DISPLAY', evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', source_continuity_status='UNKNOWN', time_status=time_status, instance_id=instance_id, region_id=spec['region_id'], reference_point=spec['reference_point'], coordinate_frame=region['geometry_frame'], continuous=False, points=points, observation_start=dict(start_kind=spec['start_kind'], status=start['status'], reason_codes=deepcopy(start['reason_codes']), interval_ns=deepcopy(start['interval_ns'])), observation_end={k: deepcopy(end[k]) for k in ('end_reason', 'right_censored', 'interval_ns', 'last_point', 'stop_witness')}, pose_status='NOT_PROVIDED')
    return dict(tool_id='U05', definition_version=_TRAJ_SCHEMA, instance_id=instance_id, region_id=spec['region_id'], reference_point=spec['reference_point'], continuous=continuous, coordinate_frame=region['geometry_frame'], points=points, observation_start=dict(start_kind=spec['start_kind'], status=start['status'], reason_codes=deepcopy(start['reason_codes']), interval_ns=deepcopy(start['interval_ns'])), observation_end={k: deepcopy(end[k]) for k in ('end_reason', 'right_censored', 'interval_ns', 'last_point', 'stop_witness')}, pose_status='NOT_PROVIDED')

def _trajectory(t, label):
    if type(t) is not dict or t.get('tool_id') != 'U05':
        raise ValueError(label + ' must be a U05 trajectory')
    gt = t.get('profile') == 'GT_DISPLAY'
    expected_fields = _TRAJ_FIELDS | {'profile', 'evidence_scope', 'source_continuity_status', 'time_status'} if gt else _TRAJ_FIELDS
    expected_schema = _GT_TRAJ_SCHEMA if gt else _TRAJ_SCHEMA
    if set(t) != expected_fields or t.get('definition_version') != expected_schema:
        raise ValueError(label + ' profile or trajectory fields differ')
    if type(t['continuous']) is not bool or not isinstance(t['points'], list) or (not t['points']):
        raise ValueError(label + ' trajectory contract differs')
    if not isinstance(t['region_id'], str) or not isinstance(t['reference_point'], str) or t['coordinate_frame'] != 'STATIC_REGION_LOCAL_METRES':
        raise ValueError(label + ' coordinate frame is missing')
    start = t['observation_start']
    if type(start) is not dict or set(start) != {'start_kind', 'status', 'reason_codes', 'interval_ns'}:
        raise ValueError(label + ' observation start contract differs')
    if gt and (t['continuous'] is not False or t['evidence_scope'] != 'GT_DISPLAY_POLYLINE_GEOMETRY' or t['source_continuity_status'] != 'UNKNOWN' or (t['time_status'] not in ('KNOWN_ENDPOINT_CLOCKS_ONLY', 'PARTIAL_SOURCE_CLOCKS', 'SOURCE_CLOCKS_COMPLETE'))):
        raise ValueError(label + ' GT display semantics differ')
    prior = None
    first_time = last_time = None
    for i, p in enumerate(t['points']):
        if type(p) is not dict or set(p) != {'ordinal', 'relative_time_ns', 'x_m', 'y_m'} or p['ordinal'] != i:
            raise ValueError(label + ' point sequence differs')
        raw = p['relative_time_ns']
        if gt and raw is None:
            ti = None
        else:
            ti = _integer(raw, 'relative_time_ns')
            if ti < 0 or (i == 0 and ti != 0) or (prior is not None and ti <= prior):
                raise ValueError(label + ' point time is invalid or not strictly increasing')
            prior = ti
        if i == 0:
            first_time = ti
        if i == len(t['points']) - 1:
            last_time = ti
        _num(p['x_m'], 'x_m')
        _num(p['y_m'], 'y_m')
    if gt and (first_time != 0 or last_time is None or last_time != t['observation_end']['last_point']['relative_time_ns']):
        raise ValueError(label + ' GT display endpoint clocks differ')
    return t

def align_trajectories(traj_a, traj_b, alignment):
    """U06: apply every immutable structure-selected B-to-A transform."""
    a, b = (_trajectory(traj_a, 'traj_a'), _trajectory(traj_b, 'traj_b'))
    if type(alignment) is not _AlignmentResult:
        raise TypeError('requires the original immutable U02 result')
    result = alignment.as_dict()
    if result.get('tool_id') != 'U02' or result.get('definition_version') != 'q1-planar-atomics/0.2.0':
        raise ValueError('alignment is not a supported U02 result')
    if result.get('transform_direction') != 'B_TO_A' or (result.get('region_a'), result.get('region_b')) != (a['region_id'], b['region_id']):
        raise ValueError('alignment regions or direction differ')
    is_gt = a.get('profile') == 'GT_DISPLAY' and b.get('profile') == 'GT_DISPLAY'
    if 'GT_DISPLAY' in (a.get('profile'), b.get('profile')) and (not is_gt):
        raise ValueError('cannot mix GT_DISPLAY and OBSERVED trajectories')
    if a['coordinate_frame'] != b['coordinate_frame'] or (not is_gt and a['reference_point'] != b['reference_point']):
        raise ValueError('trajectories do not share the same reference point')
    transforms = result.get('transforms')
    if not isinstance(transforms, list):
        raise ValueError('alignment transforms are invalid')
    pairs = []
    for tr in transforms:
        theta, tx, ty = (_num(tr[k], k) for k in ('theta_rad', 'x_m', 'y_m'))
        c, s = (math.cos(theta), math.sin(theta))
        moved = []
        for p in b['points']:
            moved.append(dict(p, x_m=_num(c * p['x_m'] - s * p['y_m'] + tx, 'transformed x'), y_m=_num(s * p['x_m'] + c * p['y_m'] + ty, 'transformed y')))
        pairs.append(dict(grid_index=deepcopy(tr['grid_index']), transform=deepcopy(tr), trajectory_a=deepcopy(a['points']), trajectory_b=moved))
    complete = result.get('status') in ('VALID', 'AMBIGUOUS') and bool(transforms) and ('SEARCH_LIMITED' not in result.get('reason_codes', [])) and (result.get('evaluated_candidates') is not None) and (result.get('candidate_count') == len(result['evaluated_candidates']))
    if is_gt:
        return dict(tool_id='U06', definition_version=_GT_ALIGNED_SCHEMA, profile='GT_DISPLAY', evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', status=result['status'], search_complete=complete, region_a=a['region_id'], region_b=b['region_id'], reference='GT_DISPLAY_POINT_REFERENCE', reference_point_a=a['reference_point'], reference_point_b=b['reference_point'], reference_comparability='NOT_ESTABLISHED', coordinate_frame=a['coordinate_frame'], source_continuity_a='UNKNOWN', source_continuity_b='UNKNOWN', time_status_a=a['time_status'], time_status_b=b['time_status'], continuous_a=False, continuous_b=False, observation_start_a=deepcopy(a['observation_start']), observation_start_b=deepcopy(b['observation_start']), observation_end_a=deepcopy(a['observation_end']), observation_end_b=deepcopy(b['observation_end']), pairs=pairs)
    return dict(tool_id='U06', definition_version=_ALIGNED_SCHEMA, status=result['status'], search_complete=complete, region_a=a['region_id'], region_b=b['region_id'], reference_point=a['reference_point'], coordinate_frame=a['coordinate_frame'], continuous_a=a['continuous'], continuous_b=b['continuous'], observation_start_a=deepcopy(a['observation_start']), observation_start_b=deepcopy(b['observation_start']), observation_end_a=deepcopy(a['observation_end']), observation_end_b=deepcopy(b['observation_end']), pairs=pairs)

def _grid(length, step, budget):
    return _position_grid(length, step, 1e-12 * max(1.0, length), budget)

def _length(points):
    try:
        return _finite(math.fsum((_finite(math.hypot(b['x_m'] - a['x_m'], b['y_m'] - a['y_m']), 'segment length') for a, b in zip(points, points[1:]))), 'route length')
    except OverflowError as error:
        raise ValueError('route length is not representable as finite') from error

def _at(points, distance):
    for a, b in zip(points, points[1:]):
        d = math.hypot(b['x_m'] - a['x_m'], b['y_m'] - a['y_m'])
        if distance <= d or a is points[-2]:
            if d == 0:
                return (a['x_m'], a['y_m'])
            f = min(1.0, distance / d)
            return (a['x_m'] + f * (b['x_m'] - a['x_m']), a['y_m'] + f * (b['y_m'] - a['y_m']))
        distance -= d
    return (points[-1]['x_m'], points[-1]['y_m'])

def _prepare_segments(points):
    """Prepare once per aligned polyline; retain zero segments and source order.

    Inspired by OASIS's reusable/batched segment projection. Keep this project's
    scaled arithmetic instead of squaring large lengths or allocating N*M arrays.
    """
    segments = []
    for a, b in zip(points, points[1:]):
        dx, dy = (_finite(b['x_m'] - a['x_m'], 'segment dx'), _finite(b['y_m'] - a['y_m'], 'segment dy'))
        length = _finite(math.hypot(dx, dy), 'segment length')
        segments.append((a['x_m'], a['y_m'], dx, dy, length))
    return tuple(segments)

def _distance_to_segments(p, segments):
    best = math.inf
    for ax, ay, dx, dy, length in segments:
        if length == 0:
            closest = (ax, ay)
        else:
            rx, ry = (_finite(p[0] - ax, 'relative x'), _finite(p[1] - ay, 'relative y'))
            scale = max(abs(rx), abs(ry))
            if scale == 0:
                fraction = 0.0
            else:
                projection_scaled = rx / scale * (dx / length) + ry / scale * (dy / length)
                if projection_scaled <= 0:
                    fraction = 0.0
                elif length >= scale:
                    fraction = min(1.0, projection_scaled * (scale / length))
                else:
                    limit = length / scale
                    fraction = 1.0 if projection_scaled >= limit else projection_scaled / limit
            closest = (_finite(ax + fraction * dx, 'closest x'), _finite(ay + fraction * dy, 'closest y'))
        best = min(best, _finite(math.hypot(p[0] - closest[0], p[1] - closest[1]), 'point distance'))
    return _finite(best, 'point-to-polyline distance')

def _distances_to_segments(probes, segments):
    return [_distance_to_segments(point, segments) for point in probes]

def _distance_to_line(p, points):
    return _distance_to_segments(p, _prepare_segments(points))

def _turn_measurement(value, reason=None, status='VALID', unit='RADIAN'):
    return dict(value=value, status=status, reason=reason, unit=unit)

def _local_turn_measurements(profile):
    """Summarize only the kernel's retained local signed-turn samples."""
    names = ('mean_signed_turn_rad', 'mean_abs_turn_rad', 'rms_turn_rad', 'coherence', 'mean_abs_turn_change_rad')

    def side_values(side):
        values = profile.get('signed_turn_' + side + '_rad')
        reasons = profile.get('signed_turn_' + side + '_reason')
        if profile.get('reason'):
            return (None, profile['reason'])
        if not isinstance(values, list) or not values or (not isinstance(reasons, list)) or (len(values) != len(reasons)):
            return (None, 'LOCAL_TURN_PROFILE_UNAVAILABLE')
        if any((reason is not None for reason in reasons)):
            return (None, next((reason for reason in reasons if reason is not None)))
        if any((isinstance(value, bool) or not isinstance(value, Real) or (not math.isfinite(value)) for value in values)):
            return (None, 'LOCAL_TURN_SAMPLE_INVALID')
        return ([float(value) for value in values], None)
    result = {}
    sides = {}
    for side in ('a', 'b'):
        values, reason = side_values(side)
        if reason:
            result[side] = {name: _turn_measurement(None, reason, 'INSUFFICIENT_EVIDENCE', unit='DIMENSIONLESS' if name == 'coherence' else 'RADIAN') for name in names}
            sides[side] = (None, reason)
            continue
        sides[side] = (values, None)
        scale = max((abs(value) for value in values))
        scaled = [value / scale for value in values] if scale else values
        mean_signed = math.fsum((value / len(values) for value in values))
        mean_abs = math.fsum((abs(value) / len(values) for value in values))
        rms = 0.0 if scale == 0 else scale * math.sqrt(math.fsum((value * value for value in scaled)) / len(values))
        result[side] = {'mean_signed_turn_rad': _turn_measurement(_finite(mean_signed, 'mean signed turn')), 'mean_abs_turn_rad': _turn_measurement(_finite(mean_abs, 'mean absolute turn')), 'rms_turn_rad': _turn_measurement(_finite(rms, 'turn RMS')), 'coherence': _turn_measurement(_finite(abs(math.fsum(scaled)) / math.fsum((abs(value) for value in scaled)), 'turn coherence'), unit='DIMENSIONLESS') if scale else _turn_measurement(None, 'ZERO_TOTAL_ABS_TURN', 'NOT_DEFINED', unit='DIMENSIONLESS'), 'mean_abs_turn_change_rad': _turn_measurement(_finite(math.fsum((abs(b - a) / (len(values) - 1) for a, b in zip(values, values[1:]))), 'mean absolute turn change')) if len(values) > 1 else _turn_measurement(None, 'FEWER_THAN_TWO_TURN_SAMPLES', 'INSUFFICIENT_EVIDENCE')}
    va, ra = sides['a']
    vb, rb = sides['b']
    if ra or rb or (va is not None and vb is not None and (len(va) != len(vb))):
        reason = ra or rb or 'LOCAL_TURN_SAMPLE_COUNTS_DIFFER'
        difference = {name: _turn_measurement(None, reason, 'INSUFFICIENT_EVIDENCE', unit='DIMENSIONLESS' if name == 'cosine_similarity' else 'RADIAN') for name in ('mean_abs_turn_difference_rad', 'max_abs_turn_difference_rad', 'cosine_similarity')}
    else:
        delta = [_finite(a - b, 'signed turn difference') for a, b in zip(va, vb)]
        differences = [abs(value) for value in delta]
        scale_a = max((abs(value) for value in va))
        scale_b = max((abs(value) for value in vb))
        if scale_a == 0 or scale_b == 0:
            cosine = _turn_measurement(None, 'ZERO_TURN_NORM', 'NOT_DEFINED', unit='DIMENSIONLESS')
        else:
            na = [value / scale_a for value in va]
            nb = [value / scale_b for value in vb]
            dot = math.fsum((a * b for a, b in zip(na, nb)))
            norm = math.sqrt(math.fsum((a * a for a in na)) * math.fsum((b * b for b in nb)))
            cosine = _turn_measurement(_finite(max(-1.0, min(1.0, dot / norm)), 'turn cosine similarity'), unit='DIMENSIONLESS')
        difference = {'mean_abs_turn_difference_rad': _turn_measurement(_finite(math.fsum((v / len(differences) for v in differences)), 'mean turn difference')), 'max_abs_turn_difference_rad': _turn_measurement(max(differences)), 'cosine_similarity': cosine}
    return dict(turn_profile_a=result['a'], turn_profile_b=result['b'], turn_profile_difference=difference)

def _alignment_summary(rows):
    count = len(rows)
    measurements = {}

    def numeric(name, values):
        valid = []
        invalid = 0
        units = []
        for value, status, unit in values:
            if status != 'VALID' or isinstance(value, bool) or (not isinstance(value, Real)) or (not math.isfinite(value)):
                invalid += 1
            else:
                valid.append(float(value))
                if unit is not None:
                    units.append(unit)
        same_unit = not units or len(set(units)) == 1
        if not same_unit:
            invalid = count
        result = dict(status='VALID' if count and invalid == 0 else 'INSUFFICIENT_EVIDENCE', min=min(valid) if count and invalid == 0 else None, max=max(valid) if count and invalid == 0 else None, count=count, invalid_count=invalid)
        if units and same_unit:
            result['unit'] = units[0]
        measurements[name] = result

    def boolean(name, values):
        valid = []
        invalid = 0
        units = []
        for value, status, unit in values:
            if status != 'VALID' or type(value) is not bool:
                invalid += 1
            else:
                valid.append(value)
                if unit is not None:
                    units.append(unit)
        same_unit = not units or len(set(units)) == 1
        if not same_unit:
            invalid = count
        result = dict(status='VALID' if count and invalid == 0 else 'INSUFFICIENT_EVIDENCE', all_true=all(valid) if count and invalid == 0 else None, all_false=not any(valid) if count and invalid == 0 else None, count=count, invalid_count=invalid)
        if units and same_unit:
            result['unit'] = units[0]
        measurements[name] = result
    position_names = sorted({'position_rms_m', 'centered_rms_m', 'chord_residual_rms_m', 'signed_turn_profile_rms_rad', 'excess_length_difference_m', 'reverse_direction', 'local_direction_difference_mean_rad'} | {name for row in rows for name in row.get('position', {})})
    for name in position_names:
        values = [(row.get('position', {}).get(name, {}).get('value'), row.get('position', {}).get(name, {}).get('status'), row.get('position', {}).get(name, {}).get('unit')) for row in rows]
        (boolean if name == 'reverse_direction' else numeric)('position.' + name, values)
    for side in ('shape_a', 'shape_b'):
        names = sorted({name for row in rows for name in row.get(side, {})})
        for name in names:
            values = [(row.get(side, {}).get(name, {}).get('value'), row.get(side, {}).get(name, {}).get('status'), row.get(side, {}).get(name, {}).get('unit')) for row in rows]
            numeric(side + '.' + name, values)
    shape_names = sorted({'length_m', 'chord_m', 'excess_length_m', 'chord_ratio', 'lateral_rms_m', 'lateral_max_m', 'mean_abs_turn_rate_rad_per_m', 'max_abs_turn_rad', 'turn_sign_changes'} | {name for row in rows for side in ('shape_a', 'shape_b') for name in row.get(side, {})})
    for name in shape_names:
        values = []
        for row in rows:
            a = row.get('shape_a', {}).get(name, {})
            b = row.get('shape_b', {}).get(name, {})
            av, bv = (a.get('value'), b.get('value'))
            valid = a.get('status') == 'VALID' and b.get('status') == 'VALID' and (not isinstance(av, bool)) and (not isinstance(bv, bool)) and isinstance(av, Real) and isinstance(bv, Real) and math.isfinite(av) and math.isfinite(bv) and (a.get('unit') == b.get('unit'))
            values.append((abs(float(av) - float(bv)) if valid else None, 'VALID' if valid else 'INSUFFICIENT_EVIDENCE', a.get('unit') if a.get('unit') == b.get('unit') else None))
        numeric('shape_difference.' + name, values)
    for name in ('start_m', 'end_m'):
        values = [(row.get('endpoint_deviation', {}).get(name), row.get('endpoint_deviation', {}).get('status'), row.get('endpoint_deviation', {}).get('unit')) for row in rows]
        numeric('endpoint.' + name, values)
    for name in ('mean_a_to_b_m', 'mean_b_to_a_m', 'rms_a_to_b_m', 'rms_b_to_a_m', 'max_a_to_b_m', 'max_b_to_a_m'):
        values = [(row.get('point_to_polyline', {}).get(name), row.get('point_to_polyline', {}).get('status'), row.get('point_to_polyline', {}).get('unit')) for row in rows]
        numeric('coverage.' + name, values)
    values = []
    for row in rows:
        profile = row.get('ordered_profile', {})
        samples = profile.get('samples')
        sample_values = [s.get('distance_m') for s in samples] if isinstance(samples, list) else []
        valid = profile.get('status') == 'VALID' and bool(sample_values) and all((not isinstance(v, bool) and isinstance(v, Real) and math.isfinite(v) for v in sample_values))
        values.append((max(sample_values) if valid else None, 'VALID' if valid else 'INSUFFICIENT_EVIDENCE', profile.get('unit')))
    numeric('ordered.max_distance_m', values)
    for block, names in (('turn_profile_a', ('mean_signed_turn_rad', 'mean_abs_turn_rad', 'rms_turn_rad', 'coherence', 'mean_abs_turn_change_rad')), ('turn_profile_b', ('mean_signed_turn_rad', 'mean_abs_turn_rad', 'rms_turn_rad', 'coherence', 'mean_abs_turn_change_rad')), ('turn_profile_difference', ('mean_abs_turn_difference_rad', 'max_abs_turn_difference_rad', 'cosine_similarity'))):
        for name in names:
            values = [(row.get(block, {}).get(name, {}).get('value'), row.get(block, {}).get(name, {}).get('status'), row.get(block, {}).get(name, {}).get('unit')) for row in rows]
            numeric(block + '.' + name, values)
    return dict(retained_count=count, measurements=measurements)

def trajectory_discrepancy(aligned, *, sample_step_m, turn_window_m, angle_deadband_rad, progress_samples, max_samples):
    """Measure sampled point-reference polylines; max_samples bounds arc-length grids.

    progress_samples independently bounds normalized-progress comparison samples.
    Neither budget bounds every internal operation in the reused measurement kernels.
    GT_DISPLAY uses versioned continuous-window regression for direction/turn
    only. OBSERVED retains the original chord estimator; neither alters source XY.
    """
    if type(aligned) is not dict or aligned.get('tool_id') != 'U06' or aligned.get('definition_version') not in (_ALIGNED_SCHEMA, _GT_ALIGNED_SCHEMA):
        raise ValueError('requires U06 aligned trajectories')
    is_gt = aligned['definition_version'] == _GT_ALIGNED_SCHEMA
    if is_gt and (aligned.get('profile') != 'GT_DISPLAY' or aligned.get('evidence_scope') != 'GT_DISPLAY_POLYLINE_GEOMETRY' or aligned.get('reference_comparability') != 'NOT_ESTABLISHED'):
        raise ValueError('GT display alignment semantics differ')
    step = _num(sample_step_m, 'sample_step_m')
    window = _num(turn_window_m, 'turn_window_m')
    dead = _num(angle_deadband_rad, 'angle_deadband_rad')
    if step <= 0 or window <= 0 or (not 0 <= dead <= math.pi):
        raise ValueError('invalid measurement parameter')
    if type(progress_samples) is not int or progress_samples < 2 or progress_samples > 10000:
        raise ValueError('progress_samples must be an integer in [2,10000]')
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError('max_samples must be a positive integer budget >= 2')
    all_results = []
    for pair in aligned['pairs']:
        a, b = (pair['trajectory_a'], pair['trajectory_b'])
        ca, cb = (aligned['continuous_a'], aligned['continuous_b'])
        geometry_continuous_a = True if is_gt else ca
        geometry_continuous_b = True if is_gt else cb
        shape_kernel = measure_route_shape_windowed if is_gt else measure_route_shape
        pair_kernel = measure_route_pair_windowed if is_gt else measure_route_pair_v2
        shape_a = shape_kernel([(p['x_m'], p['y_m']) for p in a], continuous=geometry_continuous_a, sample_step_m=step, turn_window_m=window, angle_deadband_rad=dead, max_samples=max_samples)
        shape_b = shape_kernel([(p['x_m'], p['y_m']) for p in b], continuous=geometry_continuous_b, sample_step_m=step, turn_window_m=window, angle_deadband_rad=dead, max_samples=max_samples)
        pair_measure = pair_kernel([(p['x_m'], p['y_m']) for p in a], [(p['x_m'], p['y_m']) for p in b], continuous_a=geometry_continuous_a, continuous_b=geometry_continuous_b, turn_window_m=window, progress_samples=progress_samples)
        la, lb = (_length(a), _length(b))
        if geometry_continuous_a and geometry_continuous_b and (len(a) >= 2) and (len(b) >= 2) and (la > 0) and (lb > 0):
            ga, gb = (_grid(la, step, max_samples), _grid(lb, step, max_samples))
            segments_a, segments_b = (_prepare_segments(a), _prepare_segments(b))
            da = _distances_to_segments((_at(a, s) for s in ga), segments_b)
            db = _distances_to_segments((_at(b, s) for s in gb), segments_a)

            def rms(xs):
                scale = max(xs)
                return 0.0 if scale == 0 else _finite(scale * math.sqrt(math.fsum(((x / scale) ** 2 for x in xs)) / len(xs)), 'distance RMS')
            dist = dict(status='VALID', unit='METER', count_a=len(da), count_b=len(db), distances_a_to_b_m=da, distances_b_to_a_m=db, mean_a_to_b_m=math.fsum((x / len(da) for x in da)), mean_b_to_a_m=math.fsum((x / len(db) for x in db)), rms_a_to_b_m=rms(da), rms_b_to_a_m=rms(db), max_a_to_b_m=max(da), max_b_to_a_m=max(db))
            n = progress_samples
            ordered = []
            for i in range(n):
                progress = i / (n - 1)
                point_a, point_b = (_at(a, progress * la), _at(b, progress * lb))
                ordered.append(dict(progress=progress, distance_m=_finite(math.hypot(*(point_a[j] - point_b[j] for j in (0, 1))), 'ordered profile distance')))
            endpoints = dict(status='VALID', unit='METER', start_m=_finite(math.hypot(a[0]['x_m'] - b[0]['x_m'], a[0]['y_m'] - b[0]['y_m']), 'start deviation'), end_m=_finite(math.hypot(a[-1]['x_m'] - b[-1]['x_m'], a[-1]['y_m'] - b[-1]['y_m']), 'end deviation'))
        else:
            reason = 'CONTINUITY_NOT_PROVEN' if not (geometry_continuous_a and geometry_continuous_b) else 'ZERO_LENGTH_OR_TOO_FEW_POINTS'
            dist = dict(status='INSUFFICIENT_EVIDENCE', reason=reason, unit='METER', count_a=0, count_b=0, distances_a_to_b_m=[], distances_b_to_a_m=[])
            ordered = []
            endpoints = dict(status='INSUFFICIENT_EVIDENCE', reason=reason, unit='METER')
        position = deepcopy(pair_measure['measurements'])
        for name, measurement in position.items():
            measurement['unit'] = 'RADIAN' if name in ('signed_turn_profile_rms_rad', 'local_direction_difference_mean_rad') else 'METER' if name != 'reverse_direction' else 'BOOLEAN'
        row = dict(grid_index=deepcopy(pair['grid_index']), position=position, ordered_profile=dict(status='VALID' if ordered else 'INSUFFICIENT_EVIDENCE', unit='METER', samples=ordered), endpoint_deviation=endpoints, point_to_polyline=dist, shape_a={name: dict(value, unit=_SHAPE_UNITS[name]) for name, value in shape_a['measurements'].items()}, shape_b={name: dict(value, unit=_SHAPE_UNITS[name]) for name, value in shape_b['measurements'].items()}, route_choice=dict(status='INSUFFICIENT_EVIDENCE', reason='COMMON_GRAPH_CORRESPONDENCE_NOT_PROVIDED'))
        if is_gt:
            local_profile = deepcopy(pair_measure['diagnostics']['local_profile'])
            row['local_turn_profile'] = local_profile
            row.update(_local_turn_measurements(local_profile))
        all_results.append(row)
    common = dict(tool_id='U07', reference='POINT_REFERENCE', coordinate_frame=aligned['coordinate_frame'], measurement_scope='SAMPLED_POLYLINE_GEOMETRY', search_complete=aligned['search_complete'], config=dict(sample_step_m=step, turn_window_m=window, angle_deadband_rad=dead, progress_samples=progress_samples, max_samples=max_samples), per_alignment=all_results)
    if is_gt:
        return dict(common, definition_version=_GT_DIFF_SCHEMA, profile='GT_DISPLAY', alignment_summary=_alignment_summary(all_results), direction_estimator=deepcopy(_ESTIMATOR), evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', geometry_reference='GT_DISPLAY_POINT_REFERENCE', reference_point_a=aligned['reference_point_a'], reference_point_b=aligned['reference_point_b'], reference_comparability='NOT_ESTABLISHED', source_continuity_a='UNKNOWN', source_continuity_b='UNKNOWN', time_status_a=aligned['time_status_a'], time_status_b=aligned['time_status_b'])
    return dict(common, definition_version=_DIFF_SCHEMA, reference_point=aligned['reference_point'])

def compare_trajectories(discrepancy, *, config):
    if type(discrepancy) is not dict or discrepancy.get('tool_id') != 'U07' or discrepancy.get('definition_version') not in (_DIFF_SCHEMA, _GT_DIFF_SCHEMA):
        raise ValueError('requires U07 discrepancy')
    is_gt = discrepancy['definition_version'] == _GT_DIFF_SCHEMA
    if is_gt and (discrepancy.get('profile') != 'GT_DISPLAY' or discrepancy.get('evidence_scope') != 'GT_DISPLAY_POLYLINE_GEOMETRY' or discrepancy.get('reference_comparability') != 'NOT_ESTABLISHED'):
        raise ValueError('GT display discrepancy semantics differ')
    if config is None:
        result = dict(tool_id='U08', definition_version=_COMPARE_SCHEMA, label='Uncertain', status='NOT_CALIBRATED', reason_codes=['CONFIG_NOT_PROVIDED'], per_alignment=[])
        return dict(result, profile='GT_DISPLAY', evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', algorithm_scope='DECLARED_COMPONENTS_ONLY', reference='POINT_REFERENCE', geometry_reference='GT_DISPLAY_POINT_REFERENCE', reference_point_a=discrepancy['reference_point_a'], reference_point_b=discrepancy['reference_point_b'], reference_comparability='NOT_ESTABLISHED', source_continuity_a='UNKNOWN', source_continuity_b='UNKNOWN', time_status_a=discrepancy['time_status_a'], time_status_b=discrepancy['time_status_b']) if is_gt else result
    allowed = {'version', 'calibration_source', 'required_blocks', 'thresholds'}
    if type(config) is not dict or set(config) != allowed or (not isinstance(config['version'], str)) or (not config['version']) or (not isinstance(config['calibration_source'], str)) or (not config['calibration_source']):
        raise ValueError('comparison config fields differ')
    blocks = config['required_blocks']
    thresholds = config['thresholds']
    if type(blocks) is not list or not blocks or len(set(blocks)) != len(blocks) or (not set(blocks) <= {'route_choice', 'path_shape'}):
        raise ValueError('required_blocks must be a nonempty subset')
    allowed_shape = {'chord_residual_rms_m', 'signed_turn_profile_rms_rad', 'excess_length_difference_m'}
    if type(thresholds) is not dict or set(thresholds) - allowed_shape:
        raise ValueError('unknown threshold measurement')
    for k, v in thresholds.items():
        if type(v) is not dict or set(v) != {'same_max', 'different_min', 'unit'}:
            raise ValueError('threshold fields differ')
        lo = _num(v['same_max'], k)
        hi = _num(v['different_min'], k)
        expected_unit = 'RADIAN' if k == 'signed_turn_profile_rms_rad' else 'METER'
        if lo < 0 or hi <= lo or v['unit'] != expected_unit:
            raise ValueError('invalid threshold boundary or unit')
    if 'path_shape' in blocks and (not thresholds):
        raise ValueError('path_shape requires calibrated measurement thresholds')
    labels = []
    details = []
    for row in discrepancy['per_alignment']:
        decision = 'Uncertain'
        reasons = []
        if 'route_choice' in blocks:
            reasons.append('ROUTE_CHOICE_NOT_PROVIDED')
        if 'path_shape' in blocks:
            values = []
            sufficient = True
            for name, t in thresholds.items():
                m = row['position'].get(name)
                if m is None or m.get('status') != 'VALID' or m.get('unit') != t['unit']:
                    sufficient = False
                    reasons.append('REQUIRED_MEASUREMENT_UNAVAILABLE:' + name)
                    continue
                value = m.get('value')
                if isinstance(value, bool) or not isinstance(value, Real) or (not math.isfinite(value)) or (value < 0):
                    sufficient = False
                    reasons.append('REQUIRED_MEASUREMENT_INVALID:' + name)
                    continue
                values.append((float(value), t))
            if sufficient and 'route_choice' not in blocks:
                if any((v >= t['different_min'] for v, t in values)):
                    decision = 'Different'
                elif all((v <= t['same_max'] for v, t in values)):
                    decision = 'Same'
                else:
                    reasons.append('THRESHOLD_GRAY_ZONE')
                reasons.append('DECLARED_COMPONENTS_ONLY')
        labels.append(decision)
        details.append(dict(grid_index=deepcopy(row.get('grid_index')), label=decision, reason_codes=reasons))
    overall = 'Uncertain'
    if discrepancy['search_complete'] and labels and (len(set(labels)) == 1) and (labels[0] in ('Same', 'Different')):
        overall = labels[0]
    status = 'DECLARED_COMPONENTS_ONLY' if 'route_choice' not in blocks else 'INSUFFICIENT_EVIDENCE'
    result = dict(tool_id='U08', definition_version=_COMPARE_SCHEMA, label=overall, status=status, required_blocks=deepcopy(blocks), config=deepcopy(config), per_alignment=details, reason_codes=[] if overall != 'Uncertain' else ['EVIDENCE_OR_ALIGNMENT_INCOMPLETE'])
    if is_gt:
        result.update(profile='GT_DISPLAY', evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', algorithm_scope='DECLARED_COMPONENTS_ONLY', reference='POINT_REFERENCE', geometry_reference='GT_DISPLAY_POINT_REFERENCE', reference_point_a=discrepancy['reference_point_a'], reference_point_b=discrepancy['reference_point_b'], reference_comparability='NOT_ESTABLISHED', source_continuity_a='UNKNOWN', source_continuity_b='UNKNOWN', time_status_a=discrepancy['time_status_a'], time_status_b=discrepancy['time_status_b'])
    return result
