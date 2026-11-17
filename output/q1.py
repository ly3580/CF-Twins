import numpy as np
from shapely.geometry import shape

TASK_MEASURES = ('entry_distance_m', 'entry_direction_difference_rad', 'observed_exit_distance_m', 'observed_exit_direction_difference_rad')
CENSORED_REASONS = ('TIME_LIMIT', 'EXIT_OR_TIME_LIMIT', 'SOURCE_END', 'OBSERVATION_GAP')

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
    known = [v for v in values if v is not None]
    if not known:
        return [None, None, None]
    return [float(min(known)), float(sum(known) / len(known)), float(max(known))]

def _observed_exit(endpoint):
    if not isinstance(endpoint, dict):
        return None
    status = endpoint.get('status')
    reason = endpoint.get('end_reason')
    censored = endpoint.get('right_censored')
    if status == 'VALID' and reason == 'OBSERVED_EXIT' and censored is False:
        return 1.0
    if status == 'INSUFFICIENT_EVIDENCE' and censored is True and reason in CENSORED_REASONS:
        return 0.0
    return None

def _dilate_one_cell(mask):
    result = mask.copy()
    result[1:, :] |= mask[:-1, :]
    result[:-1, :] |= mask[1:, :]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result

def _static_coverage(region_a, region_b):
    first = region_a.get('static_grid')
    second = region_b.get('static_grid')
    if not isinstance(first, dict) or not isinstance(second, dict):
        return [None, None, None, None]
    if first.get('base_mask_semantics') == 'TG_FROZEN_COMPARISON_GROUND_CANDIDATE' and second.get('base_mask_semantics') == 'NUPLAN_SOURCE_ROAD_MASK':
        tg, np_grid = first, second
    elif second.get('base_mask_semantics') == 'TG_FROZEN_COMPARISON_GROUND_CANDIDATE' and first.get('base_mask_semantics') == 'NUPLAN_SOURCE_ROAD_MASK':
        tg, np_grid = second, first
    else:
        return [None, None, None, None]
    for grid in (tg, np_grid):
        if grid.get('geometry_frame') != 'STATIC_REGION_LOCAL_METRES' or _number(grid.get('cell_m')) != 0.5:
            return [None, None, None, None]
        origin = np.asarray(grid.get('origin_xy_m'), dtype=float)
        if origin.shape != (2,) or not np.array_equal(origin, np.asarray([-10.0, -10.0])):
            return [None, None, None, None]
    if tg.get('observed_mask_semantics') != 'TG_ALL_RAW_SEMANTIC_CELL_SUPPORT':
        return [None, None, None, None]
    if np_grid.get('observed_mask_semantics') != 'NOT_PROVIDED_NUPLAN' or np_grid.get('observed_mask') is not None:
        return [None, None, None, None]
    tg_base = np.asarray(tg.get('base_mask'))
    np_base = np.asarray(np_grid.get('base_mask'))
    observed = np.asarray(tg.get('observed_mask'))
    for mask in (tg_base, np_base, observed):
        if mask.shape != (40, 40) or mask.dtype != np.dtype(bool):
            return [None, None, None, None]
    a = tg_base & observed
    b = np_base & observed
    count_a = int(np.count_nonzero(a))
    count_b = int(np.count_nonzero(b))
    area_a, area_b = 0.25 * count_a, 0.25 * count_b
    if count_a == 0 or count_b == 0:
        return [None, None, float(area_a), float(area_b)]
    coverage_a = float(np.count_nonzero(a & _dilate_one_cell(b)) / count_a)
    coverage_b = float(np.count_nonzero(b & _dilate_one_cell(a)) / count_b)
    return [coverage_a, coverage_b, float(area_a), float(area_b)]

def _geometry(local_map):
    if not isinstance(local_map, dict):
        return None
    geometry = local_map.get('geometry')
    if not isinstance(geometry, dict):
        return None
    result = shape(geometry)
    if result.is_empty or not result.is_valid:
        return None
    return result

def _component_summary(local_map):
    components = local_map.get('road_components') if isinstance(local_map, dict) else None
    if not isinstance(components, list):
        return [None, None]
    areas = []
    for component in components:
        geometry = _geometry(component)
        if geometry is None:
            return [float(len(components)), None]
        area = _number(geometry.area)
        if area is None:
            return [float(len(components)), None]
        areas.append(area)
    return [float(len(components)), float(sum(areas))]

def _features(region_a, region_b, spatial, task_comparison):
    sides = spatial.get('sides', {})
    side_a = sides.get('a', {})
    side_b = sides.get('b', {})
    map_a = side_a.get('local_map')
    map_b = side_b.get('local_map')
    geometry_a = _geometry(map_a)
    geometry_b = _geometry(map_b)
    features = _static_coverage(region_a, region_b)
    features.extend([
        _number(geometry_a.area) if geometry_a is not None else None,
        _number(geometry_b.area) if geometry_b is not None else None,
        _number(geometry_a.length) if geometry_a is not None else None,
        _number(geometry_b.length) if geometry_b is not None else None
    ])
    overlaps, distances = [], []
    for retained in spatial.get('per_alignment', []):
        transformed_b = _geometry(retained.get('b_local_map_in_a'))
        if geometry_a is None or transformed_b is None:
            overlaps.append(None)
            distances.append(None)
        else:
            union_area = float(geometry_a.union(transformed_b).area)
            overlaps.append(_number(geometry_a.intersection(transformed_b).area / union_area) if union_area > 0.0 else None)
            distances.append(_number(geometry_a.hausdorff_distance(transformed_b)))
    features.extend(_stats(overlaps))
    features.extend(_stats(distances))
    comparisons = task_comparison.get('comparisons', [])
    for name in TASK_MEASURES:
        features.extend(_stats([_measurement(row.get(name)) for row in comparisons]))
    components_a = _component_summary(map_a)
    components_b = _component_summary(map_b)
    features.extend([components_a[0], components_b[0], components_a[1], components_b[1]])
    features.extend([_observed_exit(side_a.get('observed_exit')), _observed_exit(side_b.get('observed_exit'))])
    if len(features) != 32:
        raise ValueError('Q1 feature schema mismatch')
    return features

def extract(inputs, atoms):
    region_a = atoms['get_local_region'](inputs['structure_a'], scope='comparison')
    region_b = atoms['get_local_region'](inputs['structure_b'], scope='comparison')
    alignment = atoms['align_region_structures'](region_a, region_b, config=inputs['alignment_config'], layer_pairs=inputs['layer_pairs'])
    spatial = atoms['get_local_task_geometry'](inputs['context_a'], inputs['context_b'], instance_a=inputs['instance_a'], instance_b=inputs['instance_b'], alignment=alignment, profile='GT_POSTHOC', posthoc=inputs['posthoc'])
    task_comparison = atoms['compare_local_tasks'](inputs['context_a'], inputs['context_b'], instance_a=inputs['instance_a'], instance_b=inputs['instance_b'], alignment=alignment, profile='GT_POSTHOC', posthoc=inputs['posthoc'])
    atoms['ct_evidence']()
    return _features(region_a, region_b, spatial, task_comparison)

def fit(features, labels):
    raise RuntimeError('Only joint fit_multiclass is used')

def predict(features, state):
    raise RuntimeError('Only joint predict_multiclass is used')
