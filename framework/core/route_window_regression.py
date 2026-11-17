import bisect
import math
from . import route_pair_atomics as _pair
from . import route_shape_atomics as _shape
from .route_pair_direction import measure_route_pair_v2
_DIRECTION = 'local_direction_difference_mean_rad'
_TURN = 'signed_turn_profile_rms_rad'
_ESTIMATOR = {'name': 'CONTINUOUS_ARC_LENGTH_LINEAR_REGRESSION', 'version': '0.1.0', 'domain': 'UNCHANGED_OBSERVED_POLYLINE', 'weighting': 'UNIFORM_CONTINUOUS_ARC_LENGTH', 'integration': 'EXACT_PIECEWISE_LINEAR_FIRST_MOMENT', 'source_clock': 'NOT_USED', 'legacy_unavailable_samples': 'PRESERVED'}

def _tangent(polyline, lower, upper, tolerance):
    """Direction of the linear LS fit on one closed arc-length interval.

    In normalized coordinates u in [-1,1], its slope is proportional to
    integral(u*p(u) du). Each clipped source segment is integrated exactly.
    Subtracting p(0) does not change that moment, and scaling its offsets avoids
    products of large metre values. The fitted endpoint displacement is 3*I.
    """
    vertices, cumulative, length = polyline
    lower, upper = (max(0.0, lower), min(length, upper))
    if upper <= lower:
        return None
    half = (upper - lower) / 2
    middle = lower + half
    origin = _pair._interpolate(polyline, middle)
    first = bisect.bisect_right(cumulative, lower)
    last = bisect.bisect_left(cumulative, upper)
    positions = [lower, *cumulative[first:last], upper]
    offsets = [_pair._difference(_pair._interpolate(polyline, s), origin) for s in positions]
    scale = max((abs(value) for offset in offsets for value in offset))
    if scale == 0:
        return None
    points = [(x / scale, y / scale) for x, y in offsets]
    u = [(s - middle) / half for s in positions]
    u[0], u[-1] = (-1.0, 1.0)
    moments = []
    for axis in (0, 1):
        moments.append(math.fsum(((b - a) / 6 * ((2 * a + b) * pa[axis] + (a + 2 * b) * pb[axis]) for a, b, pa, pb in zip(u, u[1:], points, points[1:]))))
    norm = math.hypot(*moments)
    if norm <= tolerance / scale / 3:
        return None
    return (moments[0] / norm, moments[1] / norm)

def _angle(a, b, *, signed):
    cross, dot = (a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    angle = math.atan2(cross if signed else abs(cross), dot)
    if abs(angle) <= _pair._ANGLE_TOLERANCE:
        return 0.0
    if math.pi - abs(angle) <= _pair._ANGLE_TOLERANCE:
        return math.copysign(math.pi, angle) if signed else math.pi
    return angle

def _turn(polyline, center, window, tolerance, *, signed_pi_unknown):
    before = _tangent(polyline, center - window, center, tolerance)
    after = _tangent(polyline, center, center + window, tolerance)
    if before is None or after is None:
        return (None, 'DEGENERATE_WINDOW_REGRESSION')
    angle = _angle(before, after, signed=True)
    if signed_pi_unknown and abs(angle) == math.pi:
        return (None, 'AMBIGUOUS_TURN_SIDE')
    return (angle, None)

def _turn_failure(reasons):
    """Keep degeneracy independent of input order when both sides fail."""
    reason = 'DEGENERATE_WINDOW_REGRESSION' if 'DEGENERATE_WINDOW_REGRESSION' in reasons else 'AMBIGUOUS_TURN_SIDE'
    return _pair._unavailable(reason, 'NOT_DEFINED' if reason == 'AMBIGUOUS_TURN_SIDE' else 'INSUFFICIENT_EVIDENCE')

def measure_route_pair_windowed(a, b, *, continuous_a, continuous_b, turn_window_m, progress_samples=101):
    """Replace v2 local directions/turns, preserving every original XY metric.

    Full spatial windows estimate direction, and two adjacent half-windows
    estimate signed turn. No original invalid local sample becomes valid and
    no sample is dropped. Newly degenerate regression intervals stay unknown.
    """
    points_a, points_b = (_pair._points(a), _pair._points(b))
    result = measure_route_pair_v2(points_a, points_b, continuous_a=continuous_a, continuous_b=continuous_b, turn_window_m=turn_window_m, progress_samples=progress_samples)
    result['schema'] = 'route-pair-window-regression/0.1.0'
    result['estimator'] = dict(_ESTIMATOR)
    local = result['diagnostics']['local_profile']
    if not local['progress']:
        return result
    polylines = (_pair._polyline(points_a), _pair._polyline(points_b))
    tolerances = [1e-12 * max(1.0, polyline[2]) for polyline in polylines]
    window = result['config']['turn_window_m']
    for index, progress in enumerate(local['progress']):
        centers = [progress * polyline[2] for polyline in polylines]
        if local['direction_reason'][index] is None:
            tangents = [_tangent(polyline, center - window, center + window, tolerance) for polyline, center, tolerance in zip(polylines, centers, tolerances)]
            if any((tangent is None for tangent in tangents)):
                local['direction_difference_rad'][index] = None
                local['direction_reason'][index] = 'DEGENERATE_DIRECTION_WINDOW_REGRESSION'
            else:
                local['direction_difference_rad'][index] = _angle(*tangents, signed=False)
        for side, polyline, center, tolerance in zip(('a', 'b'), polylines, centers, tolerances):
            if local['signed_turn_' + side + '_reason'][index] is None:
                angle, reason = _turn(polyline, center, window, tolerance, signed_pi_unknown=True)
                local['signed_turn_' + side + '_rad'][index] = angle
                local['signed_turn_' + side + '_reason'][index] = reason
        a_turn, b_turn = (local['signed_turn_a_rad'][index], local['signed_turn_b_rad'][index])
        local['signed_turn_difference_rad'][index] = None if a_turn is None or b_turn is None else _pair._finite(a_turn - b_turn)
    measurements = result['measurements']
    if measurements[_DIRECTION]['status'] == 'VALID':
        if any(local['direction_reason']):
            measurements[_DIRECTION] = _pair._unavailable('DEGENERATE_DIRECTION_WINDOW_REGRESSION')
        else:
            count = len(local['progress'])
            measurements[_DIRECTION] = _pair._valid(_pair._sum((value / count for value in local['direction_difference_rad'])))
    if measurements[_TURN]['status'] == 'VALID':
        reasons = [reason for side in ('a', 'b') for reason in local['signed_turn_' + side + '_reason'] if reason is not None]
        measurements[_TURN] = _turn_failure(reasons) if reasons else _pair._valid(_pair._rms(local['signed_turn_difference_rad']))
    return result

def measure_route_shape_windowed(points, *, continuous, sample_step_m, turn_window_m, angle_deadband_rad, max_samples=10000):
    """Preserve shape geometry and replace only its three turn summaries.

    Samples use the original exact centre grid and physical half-window.
    Magnitude of a pi turn remains measurable; its sign changes remain
    undefined. Original unavailable shape measurements are never promoted.
    """
    points = _shape._points(points)
    result = _shape.measure_route_shape(points, continuous=continuous, sample_step_m=sample_step_m, turn_window_m=turn_window_m, angle_deadband_rad=angle_deadband_rad, max_samples=max_samples)
    result['schema'] = 'route-shape-window-regression/0.1.0'
    result['estimator'] = dict(_ESTIMATOR)
    measurements = result['measurements']
    if not any((measurements[name]['status'] == 'VALID' for name in _shape._TURN_NAMES)):
        return result
    polyline = _shape._polyline(points)
    length, config = (polyline[2], result['config'])
    tolerance = 1e-12 * max(1.0, length)
    window, deadband = (config['turn_window_m'], config['angle_deadband_rad'])
    centers = _shape._center_grid(length, config['sample_step_m'], window, tolerance, config['max_samples'])
    turns = []
    for center in centers:
        angle, reason = _turn(polyline, center, window, tolerance, signed_pi_unknown=False)
        if reason is not None:
            for name in _shape._TURN_NAMES:
                if measurements[name]['status'] == 'VALID':
                    measurements[name] = _shape._unavailable('INSUFFICIENT_EVIDENCE', reason)
            return result
        turns.append(angle)
    magnitudes = [abs(turn) for turn in turns]
    estimates = {'mean_abs_turn_rate_rad_per_m': _shape._valid(_shape._sum(magnitudes) / len(magnitudes) / window), 'max_abs_turn_rad': _shape._valid(max(magnitudes))}
    retained = [turn for turn in turns if abs(turn) > deadband]
    if any((abs(turn) == math.pi for turn in retained)):
        estimates['turn_sign_changes'] = _shape._unavailable('NOT_DEFINED', 'AMBIGUOUS_TURN_SIDE')
    else:
        signs = [1 if turn > 0 else -1 for turn in retained]
        estimates['turn_sign_changes'] = _shape._valid(sum((a != b for a, b in zip(signs, signs[1:]))))
    for name, estimate in estimates.items():
        if measurements[name]['status'] == 'VALID':
            measurements[name] = estimate
    return result
