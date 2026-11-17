import math
from . import route_pair_atomics as _legacy
_DIRECTION_NAME = 'local_direction_difference_mean_rad'

def _direction_difference(pa, pb, progress, window, tolerance_a, tolerance_b):
    directions = []
    for polyline, tolerance in ((pa, tolerance_a), (pb, tolerance_b)):
        center = progress * polyline[2]
        before = _legacy._interpolate(polyline, center - window)
        after = _legacy._interpolate(polyline, center + window)
        chord = _legacy._difference(after, before)
        length = _legacy._norm(chord)
        if length <= tolerance:
            return (None, 'DEGENERATE_DIRECTION_WINDOW_CHORD')
        directions.append((chord[0] / length, chord[1] / length))
    a, b = directions
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    angle = math.atan2(abs(cross), dot)
    if angle <= _legacy._ANGLE_TOLERANCE:
        angle = 0.0
    elif math.pi - angle <= _legacy._ANGLE_TOLERANCE:
        angle = math.pi
    return (angle, None)

def measure_route_pair_v2(a, b, *, continuous_a, continuous_b, turn_window_m, progress_samples=101):
    """Extend v1 with mean absolute circular local direction difference.

    All v1 measurements, configuration and diagnostics are preserved exactly.
    The v2 development profile consumes ``centered_rms_m``, the new direction
    mean, and ``signed_turn_profile_rms_rad``. Legacy chord-residual and excess
    length quantities remain diagnostics, not newly assigned matching labels.

    Use the v1 common full-half-window support [r, 1-r], where
    r=max(h/La, h/Lb), with its numerical boundary tolerance and sampling rule.
    At each shared progress t, normalize g=p(t*L+h)-p(t*L-h) on each side,
    then compute atan2(abs(cross(unit_a,unit_b)), dot(unit_a,unit_b)). The
    arithmetic mean of these angles is in [0, pi] radians. No independent
    rotation, scale, reversal, or entrance-specific crop is fitted. A single
    degenerate direction vector makes the whole direction mean unavailable;
    it is never omitted from an otherwise valid mean.

    ``diagnostics.local_profile`` stores parallel progress/direction/turn
    arrays. Null samples retain individual reason codes, including signed pi
    turn ambiguity. Its scalar ``reason`` describes the absence of a common
    support interval only; an available grid may still contain null samples.
    The original whole-turn measurement and its status remain authoritative.
    Direction and signed-turn validity are independent. A singleton common
    interval contributes exactly one sample. Unknown continuity, short/zero
    geometry and all original input/sample budgets retain their v1 boundary.

    Source iterables and coordinate iterators are consumed once into bounded,
    finite float points; only that stable representation is reused by v1 and
    the local profile. This avoids consuming a caller's generator twice.
    """
    points_a, points_b = (_legacy._points(a), _legacy._points(b))
    result = _legacy.measure_route_pair(points_a, points_b, continuous_a=continuous_a, continuous_b=continuous_b, turn_window_m=turn_window_m, progress_samples=progress_samples)
    result['schema'] = 'route-pair-measurements/0.2.0'
    measurements = result['measurements']
    diagnostics = result['diagnostics']
    local_profile = {'reason': None, 'progress': [], 'direction_difference_rad': [], 'direction_reason': [], 'signed_turn_a_rad': [], 'signed_turn_a_reason': [], 'signed_turn_b_rad': [], 'signed_turn_b_reason': [], 'signed_turn_difference_rad': []}
    diagnostics['local_profile'] = local_profile
    diagnostics['direction_sample_count'] = 0
    interval = diagnostics['turn_progress_interval']
    if interval is None:
        reason = measurements['signed_turn_profile_rms_rad']['reason']
        measurements[_DIRECTION_NAME] = _legacy._unavailable(reason)
        local_profile['reason'] = reason
        return result
    pa, pb = (_legacy._polyline(points_a), _legacy._polyline(points_b))
    tolerance_a = 1e-12 * max(1.0, pa[2])
    tolerance_b = 1e-12 * max(1.0, pb[2])
    window = result['config']['turn_window_m']
    lower, upper = interval
    count = diagnostics['turn_sample_count']
    diagnostics['direction_sample_count'] = count
    for index in range(count):
        progress = lower if count == 1 else lower + (upper - lower) * index / (count - 1)
        direction, reason = _direction_difference(pa, pb, progress, window, tolerance_a, tolerance_b)
        turn_a, reason_a = _legacy._turn(pa, progress * pa[2], window, tolerance_a)
        turn_b, reason_b = _legacy._turn(pb, progress * pb[2], window, tolerance_b)
        local_profile['progress'].append(progress)
        local_profile['direction_difference_rad'].append(direction)
        local_profile['direction_reason'].append(reason)
        local_profile['signed_turn_a_rad'].append(turn_a)
        local_profile['signed_turn_a_reason'].append(reason_a)
        local_profile['signed_turn_b_rad'].append(turn_b)
        local_profile['signed_turn_b_reason'].append(reason_b)
        local_profile['signed_turn_difference_rad'].append(None if reason_a or reason_b else _legacy._finite(turn_a - turn_b))
    if any(local_profile['direction_reason']):
        measurements[_DIRECTION_NAME] = _legacy._unavailable('DEGENERATE_DIRECTION_WINDOW_CHORD')
    else:
        measurements[_DIRECTION_NAME] = _legacy._valid(_legacy._sum((angle / count for angle in local_profile['direction_difference_rad'])))
    return result
