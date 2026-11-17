import bisect
import math
from numbers import Integral, Real
_NAMES = ('length_m', 'chord_m', 'excess_length_m', 'chord_ratio', 'lateral_rms_m', 'lateral_max_m', 'mean_abs_turn_rate_rad_per_m', 'max_abs_turn_rad', 'turn_sign_changes')
_TURN_NAMES = _NAMES[6:]
_ANGLE_TOLERANCE_RAD = 1e-12

def _finite(value):
    if not math.isfinite(value):
        raise ValueError('geometry or measurement cannot be represented as a finite number')
    return value

def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('{} must be a finite real number, not bool'.format(name))
    try:
        return _finite(float(value))
    except (OverflowError, TypeError) as error:
        raise ValueError('{} must be a representable finite number'.format(name)) from error

def _sequence(value, name):
    if isinstance(value, (str, bytes, dict, set, frozenset)):
        raise ValueError('{} must be an ordered sequence'.format(name))
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError('{} must be an ordered sequence'.format(name)) from error

def _points(points):
    result = []
    for point in _sequence(points, 'points'):
        pair = _sequence(point, 'point')
        if len(pair) != 2:
            raise ValueError('each point must contain exactly two XY coordinates')
        result.append((_number(pair[0], 'x'), _number(pair[1], 'y')))
    return result

def _sum(values):
    try:
        return _finite(math.fsum(values))
    except OverflowError as error:
        raise ValueError('geometry sum cannot be represented as a finite number') from error

def _difference(a, b):
    return (_finite(a[0] - b[0]), _finite(a[1] - b[1]))

def _norm(vector):
    return _finite(math.hypot(*vector))

def _valid(value):
    return {'value': _finite(value), 'status': 'VALID', 'reason': None}

def _unavailable(status, reason):
    return {'value': None, 'status': status, 'reason': reason}

def _position_grid(length, step, tolerance, budget):
    """Keep both endpoints; snap a near-terminal grid point only once."""
    if length == 0:
        return [0.0]
    grid = [0.0]
    for index in range(1, budget + 1):
        position = index * step
        if position >= length or length - position <= tolerance:
            if len(grid) == budget:
                raise ValueError('position grid exceeds max_samples')
            grid.append(length)
            return grid
        if len(grid) == budget:
            raise ValueError('position grid exceeds max_samples')
        grid.append(position)
    raise ValueError('position grid exceeds max_samples')

def _center_grid(length, step, window, tolerance, budget):
    """No off-grid last center is appended."""
    end = length - window
    grid = []
    for index in range(budget + 1):
        center = window + index * step
        if center > end and center - end > tolerance:
            return grid
        if len(grid) == budget:
            raise ValueError('turn-center grid exceeds max_samples')
        if abs(center - end) <= tolerance:
            grid.append(end)
            return grid
        grid.append(center)
    raise ValueError('turn-center grid exceeds max_samples')

def _polyline(points):
    local = [_difference(point, points[0]) for point in points]
    vertices = [local[0]]
    edge_lengths = []
    for point in local[1:]:
        edge = _norm(_difference(point, vertices[-1]))
        if edge != 0:
            edge_lengths.append(edge)
            vertices.append(point)
    length = _sum(edge_lengths)
    cumulative = [0.0]
    running = 0.0
    compensation = 0.0
    for edge in edge_lengths:
        adjusted = edge - compensation
        next_running = _finite(running + adjusted)
        compensation = next_running - running - adjusted
        running = next_running
        cumulative.append(running)
    cumulative[-1] = length
    return (vertices, cumulative, length)

def _interpolate(vertices, cumulative, position):
    if position <= 0:
        return vertices[0]
    if position >= cumulative[-1]:
        return vertices[-1]
    right = bisect.bisect_right(cumulative, position)
    left = right - 1
    fraction = (position - cumulative[left]) / (cumulative[right] - cumulative[left])
    delta = _difference(vertices[right], vertices[left])
    return (_finite(vertices[left][0] + fraction * delta[0]), _finite(vertices[left][1] + fraction * delta[1]))

def measure_route_shape(points, *, continuous, sample_step_m, turn_window_m, angle_deadband_rad, max_samples=10000):
    """Measure point-reference shape without files, labels, or source identity.

    Each output has its own ``value/status/reason``. VALID zero is a measured
    zero; unavailable values are null. The absolute geometric tolerance is
    ``1e-12 * max(1, length_m)`` metres and is not a matching/noise threshold.
    Angles within ``1e-12`` radians of zero or pi are normalized before the
    user deadband. A retained pi turn has no determinate left/right sign.
    ``max_samples`` separately bounds the position and turn-center grids.
    """
    if type(continuous) is not bool:
        raise ValueError('continuous must be an explicit bool')
    step = _number(sample_step_m, 'sample_step_m')
    window = _number(turn_window_m, 'turn_window_m')
    deadband = _number(angle_deadband_rad, 'angle_deadband_rad')
    if step <= 0 or window <= 0:
        raise ValueError('sample_step_m and turn_window_m must be strictly positive')
    if not 0 <= deadband <= math.pi:
        raise ValueError('angle_deadband_rad must lie in [0, pi]')
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral) or max_samples < 2:
        raise ValueError('max_samples must be a non-bool integer at least 2')
    budget = int(max_samples)
    points = _points(points)
    reason = 'CONTINUITY_NOT_PROVEN' if not continuous else 'TOO_FEW_POINTS'
    measurements = {name: _unavailable('INSUFFICIENT_EVIDENCE', reason) for name in _NAMES}
    result = {'schema': 'route-shape-measurements/0.1.0', 'reference': 'POINT_REFERENCE', 'config': {'continuous': continuous, 'sample_step_m': step, 'turn_window_m': window, 'angle_deadband_rad': deadband, 'max_samples': budget}, 'measurements': measurements}
    if not continuous or len(points) < 2:
        return result
    vertices, cumulative, length = _polyline(points)
    chord = _norm(vertices[-1])
    tolerance = 1e-12 * max(1.0, length)
    excess = length - chord
    if excess < -tolerance:
        raise ValueError('polyline length is numerically inconsistent with its chord')
    measurements['length_m'] = _valid(length)
    measurements['chord_m'] = _valid(chord)
    measurements['excess_length_m'] = _valid(max(0.0, excess))
    positions = _position_grid(length, step, tolerance, budget)
    if chord <= tolerance:
        for name in ('chord_ratio', 'lateral_rms_m', 'lateral_max_m'):
            measurements[name] = _unavailable('NOT_DEFINED', 'ZERO_CHORD')
    else:
        measurements['chord_ratio'] = _valid(length / chord)
        direction = (vertices[-1][0] / chord, vertices[-1][1] / chord)
        distances = []
        for position in positions:
            x, y = _interpolate(vertices, cumulative, position)
            distances.append(abs(_sum((x * direction[1], -y * direction[0]))))
        maximum = max(distances)
        rms = maximum * math.sqrt(_sum(((distance / maximum) ** 2 for distance in distances)) / len(distances)) if maximum else 0.0
        measurements['lateral_rms_m'] = _valid(rms)
        measurements['lateral_max_m'] = _valid(maximum)
    if length == 0:
        turn_reason = 'ZERO_LENGTH'
    elif window - length / 2 > tolerance / 2:
        turn_reason = 'WINDOW_TOO_LONG'
    else:
        turn_reason = None
        centers = _center_grid(length, step, window, tolerance, budget)
        turns = []
        for center in centers:
            before = _interpolate(vertices, cumulative, center - window)
            middle = _interpolate(vertices, cumulative, center)
            after = _interpolate(vertices, cumulative, center + window)
            u, v = (_difference(middle, before), _difference(after, middle))
            u_length, v_length = (_norm(u), _norm(v))
            if min(u_length, v_length) <= tolerance:
                turn_reason = 'DEGENERATE_WINDOW_CHORD'
                break
            u = (u[0] / u_length, u[1] / u_length)
            v = (v[0] / v_length, v[1] / v_length)
            turn = math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
            if abs(turn) <= _ANGLE_TOLERANCE_RAD:
                turn = 0.0
            elif math.pi - abs(turn) <= _ANGLE_TOLERANCE_RAD:
                turn = math.copysign(math.pi, turn)
            turns.append(turn)
        if turn_reason is None:
            magnitudes = [abs(turn) for turn in turns]
            retained_turns = [turn for turn in turns if abs(turn) > deadband]
            measurements['mean_abs_turn_rate_rad_per_m'] = _valid(_sum(magnitudes) / len(magnitudes) / window)
            measurements['max_abs_turn_rad'] = _valid(max(magnitudes))
            if any((abs(turn) == math.pi for turn in retained_turns)):
                measurements['turn_sign_changes'] = _unavailable('NOT_DEFINED', 'AMBIGUOUS_TURN_SIDE')
            else:
                signs = [1 if turn > 0 else -1 for turn in retained_turns]
                measurements['turn_sign_changes'] = _valid(sum((first != second for first, second in zip(signs, signs[1:]))))
    if turn_reason is not None:
        for name in _TURN_NAMES:
            measurements[name] = _unavailable('INSUFFICIENT_EVIDENCE', turn_reason)
    return result
