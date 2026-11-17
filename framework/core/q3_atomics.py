from copy import deepcopy
import math
from numbers import Real
from .q2_atomics import _trajectory
from .route_window_regression import _ESTIMATOR, measure_route_shape_windowed
IMPLEMENTATION_VERSION = 'q3-atomics/0.3.0'
_DEMAND_SCHEMA = 'q3-display-demands/0.2.0'
_COMPARISON_SCHEMA = 'q3-display-demand-comparison/0.2.0'
_EVIDENCE_SCHEMA = 'q3-explanation-evidence/0.3.0'
_UNITS = {'length_m': 'METER', 'chord_m': 'METER', 'excess_length_m': 'METER', 'chord_ratio': 'RATIO', 'lateral_rms_m': 'METER', 'lateral_max_m': 'METER', 'mean_abs_turn_rate_rad_per_m': 'RADIAN_PER_METER', 'max_abs_turn_rad': 'RADIAN', 'turn_sign_changes': 'COUNT'}
_PATTERN_POSITION_UNITS = {'position_rms_m': 'METER', 'centered_rms_m': 'METER', 'chord_residual_rms_m': 'METER', 'signed_turn_profile_rms_rad': 'RADIAN'}
_PATTERN_UNITS = {**_PATTERN_POSITION_UNITS, 'centered_to_position_rms_ratio': 'RATIO', 'ordered_distance_peak_m': 'METER', 'ordered_distance_peak_progress_min': 'NORMALIZED_ARC_PROGRESS', 'ordered_distance_peak_progress_max': 'NORMALIZED_ARC_PROGRESS', 'ordered_distance_peak_progress_count': 'COUNT', 'turn_difference_peak_abs_rad': 'RADIAN', 'turn_difference_peak_progress_min': 'NORMALIZED_ARC_PROGRESS', 'turn_difference_peak_progress_max': 'NORMALIZED_ARC_PROGRESS', 'turn_difference_peak_progress_count': 'COUNT', 'peak_progress_min_abs_difference': 'NORMALIZED_ARC_PROGRESS'}

def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or (not math.isfinite(value)):
        raise ValueError(name + ' must be finite and not bool')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + ' is not representable as finite')
    return result

def describe_display_demands(trajectory, *, sample_step_m, turn_window_m, angle_deadband_rad, max_samples=10000):
    """Q3D01: measure source-bound GT_DISPLAY polyline geometry only."""
    trajectory = _trajectory(trajectory, 'trajectory')
    if trajectory.get('profile') != 'GT_DISPLAY':
        raise ValueError('Q3D01 requires a source-bound U05 GT_DISPLAY trajectory')
    points = [(point['x_m'], point['y_m']) for point in trajectory['points']]
    measured = measure_route_shape_windowed(points, continuous=True, sample_step_m=sample_step_m, turn_window_m=turn_window_m, angle_deadband_rad=angle_deadband_rad, max_samples=max_samples)
    measurements = {}
    for name, value in measured['measurements'].items():
        if name not in _UNITS or set(value) != {'value', 'status', 'reason'}:
            raise ValueError('route-shape measurement contract differs')
        measurements[name] = dict(value, unit=_UNITS[name])
    return dict(tool_id='Q3D01', definition_version=_DEMAND_SCHEMA, direction_estimator=deepcopy(measured['estimator']), profile='GT_DISPLAY', evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', geometry_assumption='GT_DISPLAY_POLYLINE_AS_CONTINUOUS_CURVE', source_continuity_status='UNKNOWN', time_status=trajectory['time_status'], reference_point=trajectory['reference_point'], reference_comparability='NOT_ESTABLISHED', coordinate_frame=trajectory['coordinate_frame'], config=deepcopy(measured['config']), measurements=measurements)

def _demand(value, label):
    if type(value) is not dict or value.get('tool_id') != 'Q3D01' or value.get('definition_version') != _DEMAND_SCHEMA:
        raise ValueError(label + ' must be a Q3D01 result')
    required = {'tool_id', 'definition_version', 'profile', 'evidence_scope', 'geometry_assumption', 'source_continuity_status', 'time_status', 'reference_point', 'reference_comparability', 'coordinate_frame', 'config', 'measurements', 'direction_estimator'}
    if set(value) != required or value['profile'] != 'GT_DISPLAY' or value['evidence_scope'] != 'GT_DISPLAY_POLYLINE_GEOMETRY' or (value['geometry_assumption'] != 'GT_DISPLAY_POLYLINE_AS_CONTINUOUS_CURVE') or (value['source_continuity_status'] != 'UNKNOWN') or (value['reference_comparability'] != 'NOT_ESTABLISHED'):
        raise ValueError(label + ' Q3D01 semantics differ')
    if set(value['measurements']) != set(_UNITS):
        raise ValueError(label + ' measurements differ')
    if value['direction_estimator'] != _ESTIMATOR:
        raise ValueError(label + ' direction estimator differs')
    for name, unit in _UNITS.items():
        _measurement(value['measurements'][name], name, unit, label)
    return value

def _measurement(row, name, unit, label):
    if type(row) is not dict or set(row) != {'value', 'status', 'reason', 'unit'} or row['unit'] != unit:
        raise ValueError(label + ' measurement contract differs')
    if row['status'] == 'VALID':
        _finite(row['value'], label + '.' + name)
        if row['reason'] is not None:
            raise ValueError(label + ' valid measurement has a reason')
    elif row['status'] in ('INSUFFICIENT_EVIDENCE', 'NOT_DEFINED'):
        if row['value'] is not None or type(row['reason']) is not str or (not row['reason']):
            raise ValueError(label + ' unavailable measurement contract differs')
    else:
        raise ValueError(label + ' measurement status is invalid')

def compare_display_demands(a, b):
    """Q3D02: signed A-minus-B and ratio comparisons for fixed geometric measures."""
    a, b = (_demand(a, 'a'), _demand(b, 'b'))
    if a['config'] != b['config']:
        raise ValueError('Q3D02 requires identical Q3D01 measurement configs')
    measurements = {}
    for name, unit in _UNITS.items():
        left, right = (a['measurements'][name], b['measurements'][name])
        row = dict(unit=unit, status='VALID', reason=None, a=deepcopy(left), b=deepcopy(right))
        if left['status'] != 'VALID' or right['status'] != 'VALID':
            row.update(status='INSUFFICIENT_EVIDENCE', reason='INPUT_MEASUREMENT_UNAVAILABLE', signed_a_minus_b=None, absolute_difference=None, ratio_a_over_b=None, ratio_status='INSUFFICIENT_EVIDENCE', ratio_reason='INPUT_MEASUREMENT_UNAVAILABLE')
        else:
            av, bv = (left['value'], right['value'])
            diff = av - bv
            if not math.isfinite(diff):
                raise ValueError('measurement difference is not representable as finite')
            row.update(signed_a_minus_b=diff, absolute_difference=abs(diff))
            if bv == 0:
                row.update(ratio_a_over_b=None, ratio_status='NOT_DEFINED', ratio_reason='ZERO_DENOMINATOR')
            else:
                ratio = av / bv
                if not math.isfinite(ratio):
                    raise ValueError('measurement ratio is not representable as finite')
                row.update(ratio_a_over_b=ratio, ratio_status='VALID', ratio_reason=None)
        measurements[name] = row
    return dict(tool_id='Q3D02', definition_version=_COMPARISON_SCHEMA, direction_estimator=deepcopy(a['direction_estimator']), profile='GT_DISPLAY', evidence_scope='GT_DISPLAY_POLYLINE_GEOMETRY', geometry_comparison='A_MINUS_B', reference_point_a=a['reference_point'], reference_point_b=b['reference_point'], reference_comparability='NOT_ESTABLISHED', source_continuity_a='UNKNOWN', source_continuity_b='UNKNOWN', time_status_a=a['time_status'], time_status_b=b['time_status'], config=deepcopy(a['config']), measurements=measurements)

def _comparison(value):
    required = {'tool_id', 'definition_version', 'profile', 'evidence_scope', 'geometry_comparison', 'reference_point_a', 'reference_point_b', 'reference_comparability', 'source_continuity_a', 'source_continuity_b', 'time_status_a', 'time_status_b', 'config', 'measurements', 'direction_estimator'}
    if type(value) is not dict or set(value) != required or value.get('tool_id') != 'Q3D02' or (value.get('definition_version') != _COMPARISON_SCHEMA):
        raise ValueError('comparison must be a complete Q3D02 result')
    if value['profile'] != 'GT_DISPLAY' or value['evidence_scope'] != 'GT_DISPLAY_POLYLINE_GEOMETRY' or value['geometry_comparison'] != 'A_MINUS_B' or (value['reference_comparability'] != 'NOT_ESTABLISHED') or (value['source_continuity_a'] != 'UNKNOWN') or (value['source_continuity_b'] != 'UNKNOWN'):
        raise ValueError('Q3D02 semantics differ')
    if type(value['measurements']) is not dict or set(value['measurements']) != set(_UNITS):
        raise ValueError('Q3D02 measurement allowlist differs')
    if value['direction_estimator'] != _ESTIMATOR:
        raise ValueError('Q3D02 direction estimator differs')
    if type(value['config']) is not dict or set(value['config']) != {'continuous', 'sample_step_m', 'turn_window_m', 'angle_deadband_rad', 'max_samples'}:
        raise ValueError('Q3D02 measurement config differs')
    fields = {'unit', 'status', 'reason', 'a', 'b', 'signed_a_minus_b', 'absolute_difference', 'ratio_a_over_b', 'ratio_status', 'ratio_reason'}
    for name, unit in _UNITS.items():
        row = value['measurements'][name]
        if type(row) is not dict or set(row) != fields or row['unit'] != unit:
            raise ValueError('Q3D02 measurement contract differs')
        for side in ('a', 'b'):
            _measurement(row[side], name, unit, side)
        if row['status'] == 'VALID':
            for field in ('signed_a_minus_b', 'absolute_difference'):
                _finite(row[field], 'Q3D02.' + field)
            if row['absolute_difference'] < 0:
                raise ValueError('Q3D02 absolute difference is negative')
            if row['ratio_status'] == 'VALID':
                _finite(row['ratio_a_over_b'], 'Q3D02.ratio')
                if row['ratio_reason'] is not None:
                    raise ValueError('valid Q3D02 ratio has a reason')
            elif row['ratio_status'] != 'NOT_DEFINED' or row['ratio_a_over_b'] is not None or row['ratio_reason'] != 'ZERO_DENOMINATOR':
                raise ValueError('Q3D02 ratio status differs')
        elif row['status'] != 'INSUFFICIENT_EVIDENCE' or row['signed_a_minus_b'] is not None or row['absolute_difference'] is not None or (row['ratio_a_over_b'] is not None):
            raise ValueError('Q3D02 unavailable measurement differs')
    return value

def _pattern_measurement(value, unit, label):
    if type(value) is not dict or set(value) != {'value', 'status', 'reason', 'unit'} or value['unit'] != unit:
        raise ValueError(label + ' measurement contract differs')
    if value['status'] == 'VALID':
        _finite(value['value'], label)
        if value['reason'] is not None:
            raise ValueError(label + ' valid measurement has a reason')
    elif value['status'] in ('INSUFFICIENT_EVIDENCE', 'NOT_DEFINED'):
        if value['value'] is not None or type(value['reason']) is not str or (not value['reason']):
            raise ValueError(label + ' unavailable measurement contract differs')
    else:
        raise ValueError(label + ' measurement status is invalid')
    return value

def _unavailable_pattern(unit, reason, status='INSUFFICIENT_EVIDENCE'):
    return dict(value=None, status=status, reason=reason, unit=unit)

def _pattern_peak(values, progress, unit, label):
    if type(values) is not list or type(progress) is not list or (not values) or (len(values) != len(progress)):
        raise ValueError(label + ' profile samples differ')
    for index, value in enumerate(values):
        _finite(value, label + '.value')
        _finite(progress[index], label + '.progress')
        if not 0 <= progress[index] <= 1 or (index and progress[index] <= progress[index - 1]):
            raise ValueError(label + ' progress must increase within [0, 1]')
    peak = max(values)
    positions = [p for value, p in zip(values, progress) if value == peak]
    return (_pattern_measurement(dict(value=peak, status='VALID', reason=None, unit=unit), unit, label), positions)

def _pattern_aggregate(rows):
    result = {}
    for name, unit in _PATTERN_UNITS.items():
        measurements = [row['measurements'][name] for row in rows]
        values = [measurement['value'] for measurement in measurements if measurement['status'] == 'VALID']
        invalid = len(measurements) - len(values)
        valid = bool(measurements) and invalid == 0
        result[name] = dict(status='VALID' if valid else 'INSUFFICIENT_EVIDENCE', min=min(values) if valid else None, max=max(values) if valid else None, count=len(measurements), invalid_count=invalid, unit=unit)
    return result

def _nearest_peak_progress_distance(left, right):
    """Exact Cartesian minimum for two nonempty, sorted peak-position lists."""
    i = j = 0
    nearest = math.inf
    while i < len(left) and j < len(right):
        nearest = min(nearest, abs(left[i] - right[j]))
        if nearest == 0.0:
            return nearest
        if left[i] < right[j]:
            i += 1
        else:
            j += 1
    return nearest

def _alignment_pattern_evidence(row, ordinal):
    if type(row) is not dict or type(row.get('position')) is not dict:
        raise ValueError('U07 retained alignment position evidence is missing')
    measurements = {}
    for name, unit in _PATTERN_POSITION_UNITS.items():
        if name not in row['position']:
            raise ValueError('U07 position measurement is missing: ' + name)
        measurements[name] = deepcopy(_pattern_measurement(row['position'][name], unit, name))
    centered, position = (measurements['centered_rms_m'], measurements['position_rms_m'])
    if centered['status'] != 'VALID' or position['status'] != 'VALID':
        ratio = _unavailable_pattern('RATIO', 'INPUT_MEASUREMENT_UNAVAILABLE')
    elif position['value'] == 0:
        ratio = _unavailable_pattern('RATIO', 'ZERO_POSITION_RMS', 'NOT_DEFINED')
    else:
        ratio_value = centered['value'] / position['value']
        ratio = _pattern_measurement(dict(value=_finite(ratio_value, 'centered_to_position_rms_ratio'), status='VALID', reason=None, unit='RATIO'), 'RATIO', 'centered_to_position_rms_ratio')
    measurements['centered_to_position_rms_ratio'] = ratio
    ordered = row.get('ordered_profile')
    if type(ordered) is not dict or ordered.get('unit') != 'METER':
        raise ValueError('U07 ordered profile contract differs')
    if ordered.get('status') == 'VALID':
        if set(ordered) != {'status', 'unit', 'samples'} or type(ordered['samples']) is not list or (not ordered['samples']):
            raise ValueError('U07 valid ordered profile contract differs')
        distances, progress = ([], [])
        for sample in ordered['samples']:
            if type(sample) is not dict or set(sample) != {'progress', 'distance_m'}:
                raise ValueError('U07 ordered profile sample contract differs')
            distance = _finite(sample['distance_m'], 'ordered distance')
            if distance < 0:
                raise ValueError('U07 ordered distance is negative')
            distances.append(distance)
            progress.append(_finite(sample['progress'], 'ordered progress'))
        ordered_peak, ordered_positions = _pattern_peak(distances, progress, 'METER', 'ordered_distance_peak_m')
    elif ordered.get('status') == 'INSUFFICIENT_EVIDENCE':
        if set(ordered) != {'status', 'reason', 'unit'} or type(ordered['reason']) is not str or (not ordered['reason']):
            raise ValueError('U07 unavailable ordered profile contract differs')
        ordered_peak, ordered_positions = (_unavailable_pattern('METER', ordered['reason']), [])
    else:
        raise ValueError('U07 ordered profile status is invalid')
    measurements['ordered_distance_peak_m'] = ordered_peak
    measurements['ordered_distance_peak_progress_min'] = dict(value=min(ordered_positions), status='VALID', reason=None, unit='NORMALIZED_ARC_PROGRESS') if ordered_positions else _unavailable_pattern('NORMALIZED_ARC_PROGRESS', 'ORDERED_PROFILE_UNAVAILABLE')
    measurements['ordered_distance_peak_progress_max'] = dict(value=max(ordered_positions), status='VALID', reason=None, unit='NORMALIZED_ARC_PROGRESS') if ordered_positions else _unavailable_pattern('NORMALIZED_ARC_PROGRESS', 'ORDERED_PROFILE_UNAVAILABLE')
    measurements['ordered_distance_peak_progress_count'] = dict(value=len(ordered_positions), status='VALID', reason=None, unit='COUNT') if ordered_positions else _unavailable_pattern('COUNT', 'ORDERED_PROFILE_UNAVAILABLE')
    local = row.get('local_turn_profile')
    local_keys = {'reason', 'progress', 'direction_difference_rad', 'direction_reason', 'signed_turn_a_rad', 'signed_turn_a_reason', 'signed_turn_b_rad', 'signed_turn_b_reason', 'signed_turn_difference_rad'}
    if type(local) is not dict or set(local) != local_keys:
        raise ValueError('U07 local turn profile contract differs')
    arrays = [local[name] for name in local_keys - {'reason'}]
    if any((type(values) is not list for values in arrays)) or len({len(values) for values in arrays}) != 1:
        raise ValueError('U07 local turn profile arrays differ')
    local_progress = local['progress']
    if local['reason'] is not None:
        if type(local['reason']) is not str or not local['reason'] or local_progress:
            raise ValueError('U07 unavailable local turn profile contract differs')
        turn_peak, turn_positions = (_unavailable_pattern('RADIAN', local['reason']), [])
    else:
        if not local_progress:
            raise ValueError('U07 available local turn profile is empty')
        for index, progress_value in enumerate(local_progress):
            _finite(progress_value, 'local turn progress')
            if not 0 <= progress_value <= 1 or (index and progress_value <= local_progress[index - 1]):
                raise ValueError('U07 local turn progress must increase within [0, 1]')
        differences = local['signed_turn_difference_rad']
        for index, difference in enumerate(differences):
            direction_reason = local['direction_reason'][index]
            direction_difference = local['direction_difference_rad'][index]
            if direction_reason is None:
                _finite(direction_difference, 'local direction difference')
            elif type(direction_reason) is not str or not direction_reason or direction_difference is not None:
                raise ValueError('U07 local direction sample contract differs')
            reason_a, reason_b = (local['signed_turn_a_reason'][index], local['signed_turn_b_reason'][index])
            value_a, value_b = (local['signed_turn_a_rad'][index], local['signed_turn_b_rad'][index])
            for value, reason, label in ((value_a, reason_a, 'turn_a'), (value_b, reason_b, 'turn_b')):
                if reason is None:
                    _finite(value, 'local ' + label)
                elif type(reason) is not str or not reason or value is not None:
                    raise ValueError('U07 local turn sample contract differs')
            if reason_a is None and reason_b is None:
                _finite(difference, 'local signed turn difference')
                if difference != value_a - value_b:
                    raise ValueError('U07 signed turn difference does not match its sides')
            elif difference is not None:
                raise ValueError('U07 invalid local turn difference must be null')
        if any((reason is not None for reason in local['signed_turn_a_reason'] + local['signed_turn_b_reason'])) or any((value is None for value in differences)):
            turn_peak = _unavailable_pattern('RADIAN', 'LOCAL_TURN_SAMPLE_UNAVAILABLE')
            turn_positions = []
        else:
            turn_values = [abs(value) for value in differences]
            turn_peak, turn_positions = _pattern_peak(turn_values, local_progress, 'RADIAN', 'turn_difference_peak_abs_rad')
    measurements['turn_difference_peak_abs_rad'] = turn_peak
    measurements['turn_difference_peak_progress_min'] = dict(value=min(turn_positions), status='VALID', reason=None, unit='NORMALIZED_ARC_PROGRESS') if turn_positions else _unavailable_pattern('NORMALIZED_ARC_PROGRESS', 'LOCAL_TURN_PROFILE_UNAVAILABLE')
    measurements['turn_difference_peak_progress_max'] = dict(value=max(turn_positions), status='VALID', reason=None, unit='NORMALIZED_ARC_PROGRESS') if turn_positions else _unavailable_pattern('NORMALIZED_ARC_PROGRESS', 'LOCAL_TURN_PROFILE_UNAVAILABLE')
    measurements['turn_difference_peak_progress_count'] = dict(value=len(turn_positions), status='VALID', reason=None, unit='COUNT') if turn_positions else _unavailable_pattern('COUNT', 'LOCAL_TURN_PROFILE_UNAVAILABLE')
    if ordered_positions and turn_positions:
        separation = _nearest_peak_progress_distance(ordered_positions, turn_positions)
        measurements['peak_progress_min_abs_difference'] = dict(value=separation, status='VALID', reason=None, unit='NORMALIZED_ARC_PROGRESS')
    else:
        measurements['peak_progress_min_abs_difference'] = _unavailable_pattern('NORMALIZED_ARC_PROGRESS', 'PEAK_PROGRESS_UNAVAILABLE')
    if set(measurements) != set(_PATTERN_UNITS):
        raise ValueError('Q3D03 geometry pattern measurement allowlist differs')
    return dict(ordinal=ordinal, measurements=measurements, ordered_distance_peak_progress=ordered_positions, turn_difference_peak_progress=turn_positions)

def summarize_explanation_evidence(comparison, discrepancy):
    """Q3D03: bundle geometric demand and retained U07 alignment evidence, without labels."""
    comparison = _comparison(comparison)
    if type(discrepancy) is not dict or discrepancy.get('tool_id') != 'U07' or discrepancy.get('definition_version') != 'q2-gt-display-discrepancy/0.2.0':
        raise ValueError('discrepancy must be a U07 GT_DISPLAY result')
    if discrepancy.get('direction_estimator') != comparison['direction_estimator']:
        raise ValueError('Q3D02 and U07 direction estimator differs')
    if discrepancy.get('profile') != 'GT_DISPLAY' or discrepancy.get('evidence_scope') != 'GT_DISPLAY_POLYLINE_GEOMETRY' or discrepancy.get('reference_comparability') != 'NOT_ESTABLISHED':
        raise ValueError('U07 GT_DISPLAY semantics differ')
    if (discrepancy.get('reference_point_a'), discrepancy.get('reference_point_b')) != (comparison['reference_point_a'], comparison['reference_point_b']) or (discrepancy.get('time_status_a'), discrepancy.get('time_status_b')) != (comparison['time_status_a'], comparison['time_status_b']) or (discrepancy.get('source_continuity_a'), discrepancy.get('source_continuity_b')) != ('UNKNOWN', 'UNKNOWN'):
        raise ValueError('Q3D02 and U07 display metadata differ')
    q3_config = comparison['config']
    u07_config = discrepancy.get('config')
    measurement_keys = ('sample_step_m', 'turn_window_m', 'angle_deadband_rad', 'max_samples')
    if type(u07_config) is not dict or any((u07_config.get(key) != q3_config[key] for key in measurement_keys)):
        raise ValueError('Q3D02 and U07 measurement configs differ')
    summary = discrepancy.get('alignment_summary')
    if type(discrepancy.get('per_alignment')) is not list or type(discrepancy.get('search_complete')) is not bool or type(summary) is not dict or (set(summary) != {'retained_count', 'measurements'}) or (type(summary['retained_count']) is not int) or (summary['retained_count'] < 1) or (summary['retained_count'] != len(discrepancy['per_alignment'])) or (type(summary['measurements']) is not dict):
        raise ValueError('U07 all-retained-range summary is missing')
    patterns = [_alignment_pattern_evidence(row, ordinal) for ordinal, row in enumerate(discrepancy['per_alignment'])]
    geometry_pattern_evidence = dict(scope='GT_DISPLAY_NORMALIZED_ARC_PROGRESS', progress_semantics='INDEPENDENT_NORMALIZED_ARC_LENGTH_NOT_TIME_OR_PHYSICAL_CORRESPONDENCE', search_complete=discrepancy['search_complete'], retained_count=summary['retained_count'], per_alignment=patterns, measurements=_pattern_aggregate(patterns))
    return dict(tool_id='Q3D03', definition_version=_EVIDENCE_SCHEMA, direction_estimator=deepcopy(comparison['direction_estimator']), scope='QUALITATIVE_GT_DISPLAY_EXPLANATION', capability_evidence='NOT_PROVIDED', causal_attribution='NOT_ESTABLISHED', demand_comparison=deepcopy(comparison), alignment_summary=dict(scope='ALL_RETAINED_U07_ALIGNMENTS', search_complete=discrepancy['search_complete'], **deepcopy(summary)), geometry_pattern_evidence=geometry_pattern_evidence, reference_point_a=comparison['reference_point_a'], reference_point_b=comparison['reference_point_b'], reference_comparability='NOT_ESTABLISHED', source_continuity_a='UNKNOWN', source_continuity_b='UNKNOWN', time_status_a=comparison['time_status_a'], time_status_b=comparison['time_status_b'])
