import bisect
import math
from numbers import Integral, Real
import numpy as np
from shapely.errors import ShapelyError
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union
from shapely.prepared import prep
_MAX_POINTS = 10000
_MAX_PROGRESS_SAMPLES = 10000
_MAX_MASK_CELLS = 65536
_MAX_CORRIDOR_WORK = 5000000
_ANGLE_TOLERANCE = 1e-12
_NAMES = ('position_rms_m', 'centered_rms_m', 'chord_residual_rms_m', 'signed_turn_profile_rms_rad', 'excess_length_difference_m', 'reverse_direction')

class _BudgetError(ValueError):
    pass

def _finite(value):
    if not math.isfinite(value):
        raise ValueError('geometry or measurement is not representable as finite')
    return value

def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('{} must be a finite real number, not bool'.format(name))
    try:
        return _finite(float(value))
    except (OverflowError, TypeError) as error:
        raise ValueError('{} is not a representable finite number'.format(name)) from error

def _sequence(value, name, limit):
    if isinstance(value, (str, bytes, dict, set, frozenset)):
        raise ValueError('{} must be an ordered sequence'.format(name))
    try:
        iterator = iter(value)
    except TypeError as error:
        raise ValueError('{} must be an ordered sequence'.format(name)) from error
    result = []
    for element in iterator:
        if len(result) == limit:
            raise _BudgetError('{} exceeds its input budget'.format(name))
        result.append(element)
    return result

def _point(point):
    pair = _sequence(point, 'point', 3)
    if len(pair) != 2:
        raise ValueError('each point must have exactly two XY coordinates')
    return (_number(pair[0], 'x'), _number(pair[1], 'y'))

def _points(points):
    return [_point(point) for point in _sequence(points, 'points', _MAX_POINTS)]

def _difference(a, b):
    return (_finite(a[0] - b[0]), _finite(a[1] - b[1]))

def _norm(vector):
    return _finite(math.hypot(*vector))

def _sum(values):
    try:
        return _finite(math.fsum(values))
    except OverflowError as error:
        raise ValueError('geometry sum is not representable as finite') from error

def _polyline(points):
    vertices = [(0.0, 0.0)]
    edges = []
    for point in points[1:]:
        local = _difference(point, points[0])
        edge = _norm(_difference(local, vertices[-1]))
        if edge:
            vertices.append(local)
            edges.append(edge)
    length = _sum(edges)
    cumulative = [0.0]
    running, compensation = (0.0, 0.0)
    for edge in edges:
        adjusted = edge - compensation
        next_running = _finite(running + adjusted)
        compensation = next_running - running - adjusted
        running = next_running
        if running <= cumulative[-1]:
            raise ValueError('positive segment cannot be resolved in cumulative arc length')
        cumulative.append(running)
    cumulative[-1] = length
    return (vertices, cumulative, length)

def _interpolate(polyline, position):
    vertices, cumulative, length = polyline
    if position <= 0:
        return vertices[0]
    if position >= length:
        return vertices[-1]
    right = bisect.bisect_right(cumulative, position)
    left = right - 1
    fraction = (position - cumulative[left]) / (cumulative[right] - cumulative[left])
    delta = _difference(vertices[right], vertices[left])
    return (_finite(vertices[left][0] + fraction * delta[0]), _finite(vertices[left][1] + fraction * delta[1]))

def _rms(values):
    maximum = max((abs(value) for value in values))
    if maximum == 0:
        return 0.0
    return _finite(maximum * math.sqrt(_sum(((value / maximum) ** 2 for value in values)) / len(values)))

def _valid(value):
    if not isinstance(value, bool):
        _finite(value)
    return {'value': value, 'status': 'VALID', 'reason': None}

def _unavailable(reason, status='INSUFFICIENT_EVIDENCE'):
    return {'value': None, 'status': status, 'reason': reason}

def _turn(polyline, center, window, tolerance):
    before = _interpolate(polyline, center - window)
    middle = _interpolate(polyline, center)
    after = _interpolate(polyline, center + window)
    u, v = (_difference(middle, before), _difference(after, middle))
    u_length, v_length = (_norm(u), _norm(v))
    if min(u_length, v_length) <= tolerance:
        return (None, 'DEGENERATE_WINDOW_CHORD')
    u, v = ((u[0] / u_length, u[1] / u_length), (v[0] / v_length, v[1] / v_length))
    angle = math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
    if math.pi - abs(angle) <= _ANGLE_TOLERANCE:
        return (None, 'AMBIGUOUS_TURN_SIDE')
    return (0.0 if abs(angle) <= _ANGLE_TOLERANCE else angle, None)

def measure_route_pair(a, b, *, continuous_a, continuous_b, turn_window_m, progress_samples=101):
    """Compare the original XY polylines at equal normalized arc progress.

    Position/centered/chord-residual RMS use ``t=i/(N-1)`` including endpoints.
    The chord residual is the XY vector ``p(t)-[(1-t)*p(0)+t*p(1)]``;
    subtracting it does not rotate or rescale either route. Centering removes
    only the mean pair displacement. Length excess means length minus chord;
    its difference is absolute and still measured in metres.

    Signed turns use physical half-window h on both sides. Their N progress
    samples cover the common supported interval [max(h/La,h/Lb), 1-that]; a
    singleton interval has one sample. A pi turn has no left/right sign.
    ``reverse_direction`` only reports numerically antiparallel nonzero chords
    (cosine <= -1+1e-12), not a general turn-angle classification.

    Budgets are 10,000 source points per route and 2..10,000 progress samples.
    Invalid arguments or unrepresentable arithmetic raise ValueError. Unknown
    continuity, short windows and degenerate geometry return named nulls.
    All quantities describe POINT_REFERENCE geometry, not robot body motion.
    """
    if type(continuous_a) is not bool or type(continuous_b) is not bool:
        raise ValueError('continuous_a and continuous_b must be explicit bools')
    window = _number(turn_window_m, 'turn_window_m')
    if window <= 0:
        raise ValueError('turn_window_m must be strictly positive')
    if isinstance(progress_samples, (bool, np.bool_)) or not isinstance(progress_samples, Integral) or (not 2 <= progress_samples <= _MAX_PROGRESS_SAMPLES):
        raise ValueError('progress_samples must be an integer in [2, 10000]')
    count = int(progress_samples)
    a, b = (_points(a), _points(b))
    reason = 'CONTINUITY_NOT_PROVEN' if not (continuous_a and continuous_b) else 'TOO_FEW_POINTS'
    measurements = {name: _unavailable(reason) for name in _NAMES}
    diagnostics = {'length_a_m': None, 'length_b_m': None, 'chord_a_m': None, 'chord_b_m': None, 'chord_direction_cosine': None, 'turn_progress_interval': None, 'turn_sample_count': 0}
    result = {'schema': 'route-pair-measurements/0.1.0', 'reference': 'POINT_REFERENCE', 'config': {'continuous_a': continuous_a, 'continuous_b': continuous_b, 'turn_window_m': window, 'progress_samples': count, 'max_points_per_route': _MAX_POINTS}, 'measurements': measurements, 'diagnostics': diagnostics}
    if not (continuous_a and continuous_b) or min(len(a), len(b)) < 2:
        return result
    pa, pb = (_polyline(a), _polyline(b))
    la, lb = (pa[2], pb[2])
    ca, cb = (_norm(pa[0][-1]), _norm(pb[0][-1]))
    ta, tb = (1e-12 * max(1.0, la), 1e-12 * max(1.0, lb))
    diagnostics.update(length_a_m=la, length_b_m=lb, chord_a_m=ca, chord_b_m=cb)
    if la - ca < -ta or lb - cb < -tb:
        raise ValueError('polyline length is numerically less than its chord')
    measurements['excess_length_difference_m'] = _valid(abs(max(0.0, la - ca) - max(0.0, lb - cb)))
    zero_chord = ca <= ta or cb <= tb
    if zero_chord:
        for name in ('chord_residual_rms_m', 'reverse_direction'):
            measurements[name] = _unavailable('ZERO_CHORD', 'NOT_DEFINED')
    else:
        ua, ub = ((pa[0][-1][0] / ca, pa[0][-1][1] / ca), (pb[0][-1][0] / cb, pb[0][-1][1] / cb))
        cosine = max(-1.0, min(1.0, _sum((ua[0] * ub[0], ua[1] * ub[1]))))
        diagnostics['chord_direction_cosine'] = cosine
        measurements['reverse_direction'] = _valid(cosine <= -1.0 + _ANGLE_TOLERANCE)
    if la == 0 or lb == 0:
        for name in ('position_rms_m', 'centered_rms_m', 'signed_turn_profile_rms_rad'):
            measurements[name] = _unavailable('ZERO_LENGTH')
        return result
    offset = _difference(a[0], b[0])
    differences, residuals = ([], [])
    for index in range(count):
        progress = index / (count - 1)
        va, vb = (_interpolate(pa, progress * la), _interpolate(pb, progress * lb))
        local_delta = _difference(va, vb)
        differences.append((_finite(offset[0] + local_delta[0]), _finite(offset[1] + local_delta[1])))
        if not zero_chord:
            ra = _difference(va, (progress * pa[0][-1][0], progress * pa[0][-1][1]))
            rb = _difference(vb, (progress * pb[0][-1][0], progress * pb[0][-1][1]))
            residuals.append(_norm(_difference(ra, rb)))
    measurements['position_rms_m'] = _valid(_rms([_norm(v) for v in differences]))
    center = (_sum((v[0] / count for v in differences)), _sum((v[1] / count for v in differences)))
    measurements['centered_rms_m'] = _valid(_rms([_norm(_difference(v, center)) for v in differences]))
    if not zero_chord:
        measurements['chord_residual_rms_m'] = _valid(_rms(residuals))
    if window - la / 2 > ta / 2 or window - lb / 2 > tb / 2:
        measurements['signed_turn_profile_rms_rad'] = _unavailable('WINDOW_TOO_LONG')
        return result
    lower = min(0.5, max(window / la, window / lb))
    upper = 1.0 - lower
    turn_count = 1 if upper == lower else count
    diagnostics['turn_progress_interval'] = [lower, upper]
    diagnostics['turn_sample_count'] = turn_count
    turns = []
    for index in range(turn_count):
        progress = lower if turn_count == 1 else lower + (upper - lower) * index / (turn_count - 1)
        angle_a, reason_a = _turn(pa, progress * la, window, ta)
        angle_b, reason_b = _turn(pb, progress * lb, window, tb)
        if reason_a or reason_b:
            reason = 'DEGENERATE_WINDOW_CHORD' if 'DEGENERATE_WINDOW_CHORD' in (reason_a, reason_b) else 'AMBIGUOUS_TURN_SIDE'
            status = 'NOT_DEFINED' if reason == 'AMBIGUOUS_TURN_SIDE' else 'INSUFFICIENT_EVIDENCE'
            measurements['signed_turn_profile_rms_rad'] = _unavailable(reason, status)
            return result
        turns.append(angle_a - angle_b)
    measurements['signed_turn_profile_rms_rad'] = _valid(_rms(turns))
    return result

def _mask(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return (None, 0, 'INVALID_MASK')
    if array.ndim != 2 or min(array.shape) == 0:
        return (None, 0, 'INVALID_MASK')
    if array.size > _MAX_MASK_CELLS:
        return (None, 0, 'BUDGET_EXCEEDED')
    support = np.zeros(array.shape, dtype=bool)
    unknown = 0
    for index, item in np.ndenumerate(array):
        if isinstance(item, (bool, np.bool_)):
            support[index] = bool(item)
        elif isinstance(item, Real) and math.isfinite(item) and (item in (0, 1)):
            support[index] = item == 1
        else:
            unknown += 1
    return (support, unknown, None)

def _mask_geometry(mask, cell):
    rectangles = []
    for row, values in enumerate(mask):
        start = None
        for column in range(len(values) + 1):
            supported = column < len(values) and values[column]
            if supported and start is None:
                start = column
            elif not supported and start is not None:
                rectangles.append(box(start * cell, row * cell, column * cell, (row + 1) * cell))
                start = None
    return unary_union(rectangles)

def shared_display_corridor(a, b, mask_a, mask_b, *, cell_m, origin_xy):
    """Certify a whole piecewise-convex connection band inside shared support.

    Masks share an origin and resolution: [row, column] covers the closed cell
    [ox+column*c, ox+(column+1)*c] x [oy+row*c, oy+(row+1)*c]. Only boolean True
    or numeric 1 supports a cell; unknowns are unsupported. The caller must
    provide TG road AND observed as its TG mask, and prove input continuity.

    Every original arc-length breakpoint of either polyline is retained in a
    union of normalized progress values. For each consecutive interval, the
    convex hull of its four route endpoints must be covered by the common
    mask. This checks the full band, including its interior and both routes;
    it is conservative when the convex hull exceeds the ruled connection.
    Degenerate line bands still need full coverage. No sparse point test is
    substituted. Coverage uses only a 1e-12 * max(1, grid extent) metre buffer
    for numerical roundoff, never a matching or physical clearance threshold.

    VERIFIED means only DISPLAY_GEOMETRY_ONLY. UNKNOWN never proves different
    passages. Budgets: 10,000 points/route, 65,536 cells/mask and 5,000,000 for
    (number of bands * number of grid cells). Budget refusal returns UNKNOWN.
    """
    cell = _number(cell_m, 'cell_m')
    if cell <= 0:
        raise ValueError('cell_m must be strictly positive')
    origin = _point(origin_xy)
    diagnostics = {'checked_bands': 0, 'breakpoint_count': 0, 'support_cells': 0, 'unknown_cells_a': 0, 'unknown_cells_b': 0, 'failed_band_progress': None, 'numerical_tolerance_m': None}
    result = {'schema': 'shared-display-corridor/0.1.0', 'reference': 'DISPLAY_GEOMETRY_ONLY', 'status': 'UNKNOWN', 'reason': None, 'config': {'cell_m': cell, 'origin_xy': list(origin), 'max_points_per_route': _MAX_POINTS, 'max_mask_cells': _MAX_MASK_CELLS, 'max_band_cell_work': _MAX_CORRIDOR_WORK}, 'diagnostics': diagnostics}
    if mask_a is None or mask_b is None:
        result['reason'] = 'MISSING_MASK'
        return result
    ma, unknown_a, error_a = _mask(mask_a)
    mb, unknown_b, error_b = _mask(mask_b)
    diagnostics.update(unknown_cells_a=unknown_a, unknown_cells_b=unknown_b)
    if error_a or error_b:
        result['reason'] = error_a or error_b
        return result
    if ma.shape != mb.shape:
        result['reason'] = 'MASK_SHAPE_MISMATCH'
        return result
    try:
        a, b = (_points(a), _points(b))
    except _BudgetError:
        result['reason'] = 'BUDGET_EXCEEDED'
        return result
    if min(len(a), len(b)) < 2:
        result['reason'] = 'TOO_FEW_POINTS'
        return result
    pa, pb = (_polyline(a), _polyline(b))
    if min(pa[2], pb[2]) == 0:
        result['reason'] = 'ZERO_LENGTH'
        return result
    progress = sorted(set([0.0, 1.0] + [s / pa[2] for s in pa[1]] + [s / pb[2] for s in pb[1]]))
    diagnostics['breakpoint_count'] = len(progress)
    if (len(progress) - 1) * int(ma.size) > _MAX_CORRIDOR_WORK:
        result['reason'] = 'BUDGET_EXCEEDED'
        return result
    common = ma & mb
    diagnostics['support_cells'] = int(common.sum())
    if not diagnostics['support_cells']:
        result['reason'] = 'NO_COMMON_SUPPORT'
        return result
    extent = _finite(cell * max(ma.shape))
    tolerance = 1e-12 * max(1.0, extent)
    diagnostics['numerical_tolerance_m'] = tolerance
    offset_a, offset_b = (_difference(a[0], origin), _difference(b[0], origin))

    def position(polyline, offset, value):
        local = _interpolate(polyline, value * polyline[2])
        return (_finite(local[0] + offset[0]), _finite(local[1] + offset[1]))
    try:
        support = prep(_mask_geometry(common, cell).buffer(tolerance))
        for start, end in zip(progress, progress[1:]):
            band = MultiPoint([position(pa, offset_a, start), position(pa, offset_a, end), position(pb, offset_b, start), position(pb, offset_b, end)]).convex_hull
            diagnostics['checked_bands'] += 1
            if not support.covers(band):
                result['reason'] = 'BAND_NOT_COVERED'
                diagnostics['failed_band_progress'] = [start, end]
                return result
    except ShapelyError:
        result['reason'] = 'GEOMETRY_OPERATION_FAILED'
        return result
    result['status'] = 'VERIFIED'
    return result
