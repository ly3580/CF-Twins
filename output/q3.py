import numpy as np

DEMAND_MEASURES = ('length_m', 'chord_m', 'excess_length_m', 'chord_ratio', 'lateral_rms_m', 'lateral_max_m', 'mean_abs_turn_rate_rad_per_m', 'max_abs_turn_rad', 'turn_sign_changes')
PATTERN_MEASURES = ('position_rms_m', 'centered_rms_m', 'chord_residual_rms_m', 'signed_turn_profile_rms_rad', 'centered_to_position_rms_ratio', 'ordered_distance_peak_m', 'ordered_distance_peak_progress_min', 'ordered_distance_peak_progress_max', 'ordered_distance_peak_progress_count', 'turn_difference_peak_abs_rad', 'turn_difference_peak_progress_min', 'turn_difference_peak_progress_max', 'turn_difference_peak_progress_count', 'peak_progress_min_abs_difference')

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

def _turn_samples(row):
    profile = row.get('local_turn_profile', {})
    progress = profile.get('progress')
    values = profile.get('signed_turn_difference_rad')
    if progress is None or values is None or len(progress) != len(values):
        return []
    result = []
    for index in range(len(progress)):
        p = _number(progress[index])
        value = _number(values[index])
        if p is not None and 0.0 <= p <= 1.0 and value is not None:
            result.append((p, value))
    return result

def _scale_features(demand_a, demand_b, comparison, discrepancy, explanation):
    result = []
    for demand in (demand_a, demand_b):
        measurements = demand.get('measurements', {})
        for name in DEMAND_MEASURES:
            result.append(_measurement(measurements.get(name)))
    comparison_measurements = comparison.get('measurements', {})
    for name in DEMAND_MEASURES:
        row = comparison_measurements.get(name, {})
        valid = row.get('status') == 'VALID'
        result.extend([
            _number(row.get('signed_a_minus_b')) if valid else None,
            _number(row.get('absolute_difference')) if valid else None,
            _number(row.get('ratio_a_over_b')) if valid and row.get('ratio_status') == 'VALID' else None
        ])
    pattern_rows = explanation.get('geometry_pattern_evidence', {}).get('per_alignment')
    patterns = pattern_rows if isinstance(pattern_rows, list) else []
    for name in PATTERN_MEASURES:
        result.extend(_stats([_measurement(row.get('measurements', {}).get(name)) for row in patterns]))
    alignments = discrepancy.get('per_alignment', [])
    ordered = [sample for row in alignments for sample in _ordered_samples(row)]
    for index in range(8):
        distances = [distance for progress, distance in ordered if _in_bin(progress, index, 8)]
        result.extend([_mean(distances), float(max(distances)) if distances else None])
    turn_samples = [sample for row in alignments for sample in _turn_samples(row)]
    for index in range(8):
        turns = [value for progress, value in turn_samples if _in_bin(progress, index, 8)]
        result.extend([_mean(turns), _mean([abs(value) for value in turns])])
    for field in ('ordered_distance_peak_progress', 'turn_difference_peak_progress'):
        peaks = []
        for row in patterns:
            values = row.get(field)
            if isinstance(values, list):
                for value in values:
                    p = _number(value)
                    if p is not None and 0.0 <= p <= 1.0:
                        peaks.append(p)
        for index in range(4):
            result.append(float(sum(1 for p in peaks if _in_bin(p, index, 4)) / len(peaks)) if peaks else None)
    result.append(float(len(patterns)) if isinstance(pattern_rows, list) else None)
    if len(result) != 128:
        raise ValueError('Q3 scale feature schema mismatch')
    return result

def extract(inputs, atoms):
    region_a = atoms['get_local_region'](inputs['structure_a'], scope='comparison')
    region_b = atoms['get_local_region'](inputs['structure_b'], scope='comparison')
    alignment = atoms['align_region_structures'](region_a, region_b, config=inputs['alignment_config'], layer_pairs=inputs['layer_pairs'])
    trajectory_a = atoms['get_local_trajectory'](inputs['decision_a'], instance_id=inputs['instance_a'], continuous=inputs['continuous_a'])
    trajectory_b = atoms['get_local_trajectory'](inputs['decision_b'], instance_id=inputs['instance_b'], continuous=inputs['continuous_b'])
    aligned = atoms['align_trajectories'](trajectory_a, trajectory_b, alignment)
    bundles = []
    for config_name in ('measurement_config_fine', 'measurement_config_coarse'):
        config = inputs[config_name]
        discrepancy = atoms['trajectory_discrepancy'](aligned, **config)
        demand_a = atoms['describe_display_demands'](trajectory_a, sample_step_m=config['sample_step_m'], turn_window_m=config['turn_window_m'], angle_deadband_rad=config['angle_deadband_rad'], max_samples=config['max_samples'])
        demand_b = atoms['describe_display_demands'](trajectory_b, sample_step_m=config['sample_step_m'], turn_window_m=config['turn_window_m'], angle_deadband_rad=config['angle_deadband_rad'], max_samples=config['max_samples'])
        comparison = atoms['compare_display_demands'](a=demand_a, b=demand_b)
        explanation = atoms['summarize_explanation_evidence'](comparison=comparison, discrepancy=discrepancy)
        bundles.append((demand_a, demand_b, comparison, discrepancy, explanation))
    atoms['ct_evidence']()
    result = []
    for bundle in bundles:
        result.extend(_scale_features(*bundle))
    if len(result) != 256:
        raise ValueError('Q3 feature schema mismatch')
    return result

def fit(features, labels):
    raise RuntimeError('Only joint fit_multiclass is used')

def predict(features, state):
    raise RuntimeError('Only joint predict_multiclass is used')
