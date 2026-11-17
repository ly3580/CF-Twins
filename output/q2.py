import numpy as np
from shapely.geometry import LineString

POSITION_MEASURES = ('position_rms_m', 'centered_rms_m', 'chord_residual_rms_m', 'signed_turn_profile_rms_rad', 'excess_length_difference_m', 'reverse_direction', 'local_direction_difference_mean_rad')
ROUTE_MEASURES = ('length_m', 'chord_m', 'excess_length_m', 'length_over_chord', 'width_m', 'height_m', 'centroid_x_m', 'centroid_y_m', 'start_x_m', 'start_y_m', 'last_x_m', 'last_y_m', 'absolute_vertex_turn_sum_rad', 'maximum_vertex_turn_rad')

def _number(value):
    if not isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
        return None
    value = float(value)
    return value if np.isfinite(value) else None

def _measurement(value):
    if not isinstance(value, dict) or value.get('status') != 'VALID':
        return None
    return _number(value.get('value'))

def _stats(values):
    known = [value for value in values if value is not None]
    if not known:
        return [None, None, None]
    return [float(min(known)), float(sum(known) / len(known)), float(max(known))]

def _mean(values):
    return float(sum(values) / len(values)) if values else None

def _in_bin(progress, index, count):
    return index / count <= progress and (progress < (index + 1) / count or (index == count - 1 and progress <= 1.0))

def _ordered_samples(row):
    profile = row.get('ordered_profile', {})
    if profile.get('status') != 'VALID':
        return []
    result = []
    for sample in profile.get('samples', []):
        progress = _number(sample.get('progress'))
        distance = _number(sample.get('distance_m'))
        if progress is not None and 0.0 <= progress <= 1.0 and distance is not None and distance >= 0.0:
            result.append((progress, distance))
    return result

def _turn_samples(row, name):
    profile = row.get('local_turn_profile', {})
    progress = profile.get('progress')
    values = profile.get(name)
    if progress is None or values is None or len(progress) != len(values):
        return []
    result = []
    for index in range(len(progress)):
        p = _number(progress[index])
        value = _number(values[index])
        if p is not None and 0.0 <= p <= 1.0 and value is not None:
            result.append((p, value))
    return result

def _scale_features(discrepancy):
    rows = discrepancy.get('per_alignment', [])
    result = []
    for name in POSITION_MEASURES:
        result.extend(_stats([_measurement(row.get('position', {}).get(name)) for row in rows]))
    ordered = [sample for row in rows for sample in _ordered_samples(row)]
    for index in range(8):
        distances = [distance for progress, distance in ordered if _in_bin(progress, index, 8)]
        result.extend([_mean(distances), float(max(distances)) if distances else None])
    differences = [sample for row in rows for sample in _turn_samples(row, 'signed_turn_difference_rad')]
    for index in range(8):
        result.append(_mean([abs(value) for progress, value in differences if _in_bin(progress, index, 8)]))
    for name in ('direction_difference_rad', 'signed_turn_a_rad', 'signed_turn_b_rad'):
        result.append(_mean([abs(value) for row in rows for progress, value in _turn_samples(row, name)]))
    if len(result) != 48:
        raise ValueError('Q2 scale feature schema mismatch')
    return result

def _points(trajectory):
    points = trajectory.get('points')
    if not isinstance(points, list) or not points:
        return None
    result = []
    for point in points:
        x = _number(point.get('x_m'))
        y = _number(point.get('y_m'))
        if x is None or y is None:
            return None
        result.append([x, y])
    return np.asarray(result, dtype=float)

def _last_inside(trajectory, points):
    end = trajectory.get('observation_end')
    if not isinstance(end, dict) or points is None:
        return None
    last = end.get('last_point')
    if not isinstance(last, dict):
        return None
    x = _number(last.get('x_m'))
    y = _number(last.get('y_m'))
    if x is None or y is None:
        return None
    result = np.asarray([x, y], dtype=float)
    if not np.array_equal(result, points[-1]):
        return None
    return result

def _directional_coverage(own_line, other_line):
    positions = np.arange(0.0, float(own_line.length), 0.1).tolist()
    positions.append(float(own_line.length))
    covered = 0
    for position in positions:
        if own_line.interpolate(position).distance(other_line) <= 0.5:
            covered += 1
    return float(covered / len(positions))

def _native_route_features(trajectory_a, trajectory_b, points_a, points_b):
    last_a = _last_inside(trajectory_a, points_a)
    last_b = _last_inside(trajectory_b, points_b)
    endpoint_distance = float(np.linalg.norm(last_a - last_b)) if last_a is not None and last_b is not None else None
    if points_a is None or points_b is None or len(points_a) < 2 or len(points_b) < 2:
        return [None, None, None, endpoint_distance]
    line_a = LineString(points_a)
    line_b = LineString(points_b)
    if not line_a.is_valid or not line_b.is_valid or line_a.length <= 0.0 or line_b.length <= 0.0:
        return [None, None, None, endpoint_distance]
    coverage_a = _directional_coverage(line_a, line_b)
    coverage_b = _directional_coverage(line_b, line_a)
    return [coverage_a, coverage_b, float(min(coverage_a, coverage_b)), endpoint_distance]

def _route_summary(points):
    values = dict.fromkeys(ROUTE_MEASURES)
    if points is None or len(points) < 2:
        return [values[name] for name in ROUTE_MEASURES]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    length = float(np.sum(lengths))
    chord = float(np.linalg.norm(points[-1] - points[0]))
    span = np.max(points, axis=0) - np.min(points, axis=0)
    if length > 0.0:
        midpoints = (points[:-1] + points[1:]) / 2.0
        centroid = np.sum(midpoints * lengths[:, None], axis=0) / length
    else:
        centroid = points[0]
    nonzero = lengths > 0.0
    directions = segments[nonzero] / lengths[nonzero, None]
    turn_sum, turn_max = None, None
    if len(directions) == 1:
        turn_sum, turn_max = 0.0, 0.0
    elif len(directions) > 1:
        cross = directions[:-1, 0] * directions[1:, 1] - directions[:-1, 1] * directions[1:, 0]
        dot = np.sum(directions[:-1] * directions[1:], axis=1)
        turns = np.abs(np.arctan2(cross, dot))
        turn_sum, turn_max = float(np.sum(turns)), float(np.max(turns))
    values.update({
        'length_m': length, 'chord_m': chord, 'excess_length_m': length - chord,
        'length_over_chord': length / chord if chord > 0.0 else None,
        'width_m': float(span[0]), 'height_m': float(span[1]),
        'centroid_x_m': float(centroid[0]), 'centroid_y_m': float(centroid[1]),
        'start_x_m': float(points[0, 0]), 'start_y_m': float(points[0, 1]),
        'last_x_m': float(points[-1, 0]), 'last_y_m': float(points[-1, 1]),
        'absolute_vertex_turn_sum_rad': turn_sum, 'maximum_vertex_turn_rad': turn_max
    })
    return [_number(values[name]) for name in ROUTE_MEASURES]

def extract(inputs, atoms):
    region_a = atoms['get_local_region'](inputs['structure_a'], scope='comparison')
    region_b = atoms['get_local_region'](inputs['structure_b'], scope='comparison')
    alignment = atoms['align_region_structures'](region_a, region_b, config=inputs['alignment_config'], layer_pairs=inputs['layer_pairs'])
    trajectory_a = atoms['get_local_trajectory'](inputs['decision_a'], instance_id=inputs['instance_a'], continuous=inputs['continuous_a'])
    trajectory_b = atoms['get_local_trajectory'](inputs['decision_b'], instance_id=inputs['instance_b'], continuous=inputs['continuous_b'])
    aligned = atoms['align_trajectories'](trajectory_a, trajectory_b, alignment)
    fine = atoms['trajectory_discrepancy'](aligned, **inputs['measurement_config_fine'])
    coarse = atoms['trajectory_discrepancy'](aligned, **inputs['measurement_config_coarse'])
    atoms['ct_evidence']()
    result = _scale_features(fine) + _scale_features(coarse)
    points_a, points_b = _points(trajectory_a), _points(trajectory_b)
    result.extend(_native_route_features(trajectory_a, trajectory_b, points_a, points_b))
    result.extend(_route_summary(points_a))
    result.extend(_route_summary(points_b))
    if len(result) != 128:
        raise ValueError('Q2 feature schema mismatch')
    return result

def fit(features, labels):
    raise RuntimeError('Only joint fit_multiclass is used')

def predict(features, state):
    raise RuntimeError('Only joint predict_multiclass is used')
