from numbers import Integral
POLICY_VERSION = 'single-region-observation/1.0.0'
CAP_NS = 20000000000

def _integers(values, name):
    if any((isinstance(x, bool) or not isinstance(x, Integral) for x in values)):
        raise ValueError(name + ' must contain integers, not booleans/floats')
    return [int(x) for x in values]

def segment_samples(sample_indices, times_ns, relations):
    """Segment one subject/window/region in source order.

    Relations: -1 missing, 0 exterior, 1 strict interior, 2 boundary.
    Input times are integer clock ticks in nanoseconds; their provenance
    (absolute native or derived relative) belongs to the caller's source record.
    Gaps terminate censored observations and lock until actual exterior evidence.
    Locked observations remain in the source data, with suppression counters.
    """
    indices = _integers(list(sample_indices), 'sample_indices')
    times = _integers(list(times_ns), 'times_ns')
    rel = _integers(list(relations), 'relations')
    if not len(indices) == len(times) == len(rel):
        raise ValueError('input lengths differ')
    if any((x < 0 for x in indices)) or any((b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError('sample_indices must be nonnegative and strictly increasing')
    if any((b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('times_ns must be strictly increasing')
    if any((x not in (-1, 0, 1, 2) for x in rel)):
        raise ValueError('invalid point relation')
    result = dict(episodes=[], suppressed_after_limit_samples=0, suppressed_after_gap_samples=0, boundary_only_samples=0, missing_samples=rel.count(-1), sample_index_gaps=0)
    active = None
    last_outside = None
    locked = None

    def close(reason, witness=None):
        nonlocal active
        first, last = (active['sample_positions'][0], active['sample_positions'][-1])
        t_last = times[last]
        deadline = times[first] + CAP_NS
        exit_bounds = [t_last, times[witness] if witness is not None else None]
        stop_bounds = [t_last, t_last]
        if reason == 'OBSERVED_EXIT':
            stop_bounds = exit_bounds[:]
        elif reason == 'TIME_LIMIT':
            stop_bounds = [deadline, deadline]
        elif reason == 'EXIT_OR_TIME_LIMIT':
            stop_bounds = [t_last, deadline]
        active.update(first_sample_index=indices[first], last_sample_index=indices[last], first_observed_time_ns=times[first], last_observed_time_ns=t_last, observed_span_ns=t_last - times[first], deadline_ns=deadline, administrative_end_ns=deadline if reason in ('TIME_LIMIT', 'EXIT_OR_TIME_LIMIT') else None, end_reason=reason, right_censored=reason != 'OBSERVED_EXIT', exit_time_bounds_ns=exit_bounds, stop_time_bounds_ns=stop_bounds, exit_witness_position=witness)
        result['episodes'].append(active)
        active = None
    for pos, (idx, time, relation) in enumerate(zip(indices, times, rel)):
        gap = pos > 0 and idx != indices[pos - 1] + 1
        if gap:
            result['sample_index_gaps'] += 1
        if gap or relation == -1:
            last_outside = None
            if active is not None:
                close('OBSERVATION_GAP')
                locked = 'gap'
            if relation == -1:
                continue
        if active is not None:
            deadline = times[active['sample_positions'][0]] + CAP_NS
            if relation == 0:
                close('OBSERVED_EXIT' if time <= deadline else 'EXIT_OR_TIME_LIMIT', pos)
                last_outside, locked = (pos, None)
                continue
            if time <= deadline:
                active['sample_positions'].append(pos)
            if time >= deadline:
                close('TIME_LIMIT')
                locked = 'limit'
                if time > deadline:
                    result['suppressed_after_limit_samples'] += 1
            continue
        if relation == 0:
            locked, last_outside = (None, pos)
        elif locked is not None:
            result['suppressed_after_' + locked + '_samples'] += 1
        elif relation == 2:
            result['boundary_only_samples'] += 1
        else:
            active = dict(sample_positions=[pos], start_kind='OBSERVED_ENTRY' if last_outside is not None else 'LEFT_CENSORED', entry_time_bounds_ns=[times[last_outside] if last_outside is not None else None, time], entry_witness_position=last_outside)
            last_outside = None
    if active is not None:
        close('SOURCE_END')
    return result
