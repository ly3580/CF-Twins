import numpy as np

CLASSES = ('U', 'C', 'I_E', 'I_P')
GROUPS = ('q1', 'q2', 'q3')
DIMENSIONS = (32, 128, 256)

def fit_multiclass(features, labels):
    n = len(labels)
    if n == 0:
        raise ValueError('Training requires usable labelled cases')
    if any(label not in CLASSES for label in labels):
        raise ValueError('Unsupported training label')
    blocks = []
    for name, dimension in zip(GROUPS, DIMENSIONS):
        rows = features[name]
        if len(rows) != n or any(len(row) != dimension for row in rows):
            raise ValueError('Training feature dimensions do not match the contract')
        if any(value is not None and not np.isfinite(float(value)) for row in rows for value in row):
            raise ValueError('Raw training features must be finite numbers or None')
        blocks.append(np.asarray(rows, dtype=float))
    raw = np.concatenate(blocks, axis=1)
    observed = np.isfinite(raw)
    medians = np.zeros(416, dtype=float)
    scales = np.ones(416, dtype=float)
    for column in range(416):
        values = raw[observed[:, column], column]
        if len(values) > 0:
            medians[column] = float(np.median(values))
            quartiles = np.percentile(values, [25.0, 75.0])
            scales[column] = max(float((quartiles[1] - quartiles[0]) / 1.349), float(0.25 * np.std(values)), 0.000001)
    if not np.all(np.isfinite(medians)) or not np.all(np.isfinite(scales)):
        raise ValueError('Nonfinite preprocessing parameters')
    filled = np.where(observed, raw, medians[None, :])
    transformed = (2.0 / np.pi) * np.arctan((filled - medians[None, :]) / scales[None, :])
    parts = []
    start = 0
    for dimension in DIMENSIONS:
        stop = start + dimension
        group_weight = float(np.sqrt(1.0 / (3.0 * dimension)))
        parts.append(transformed[:, start:stop] * group_weight)
        parts.append((~observed[:, start:stop]).astype(float) * (0.5 * group_weight))
        start = stop
    centers = np.concatenate(parts, axis=1)
    squared_norms = np.sum(centers * centers, axis=1)
    distance_squared = np.maximum(squared_norms[:, None] + squared_norms[None, :] - 2.0 * centers.dot(centers.T), 0.0)
    kernel = (np.exp(-distance_squared / 0.125) + np.exp(-distance_squared / 0.5) + np.exp(-distance_squared / 2.0)) / 3.0
    y = np.asarray([CLASSES.index(label) for label in labels], dtype=int)
    counts = np.bincount(y, minlength=4).astype(float)
    if np.any(counts == 0.0):
        raise ValueError('Each of the four classes must occur in training')
    target = np.eye(4, dtype=float)[y]
    sample_weights = 1.0 / np.sqrt(counts[y])
    sample_weights *= n / float(np.sum(sample_weights))
    prior = np.log((counts + 1.0) / (n + 4.0))
    prior -= np.mean(prior)
    coefficients = np.zeros((n, 4), dtype=float)
    intercept = prior.copy()
    extrapolated_coefficients = coefficients.copy()
    extrapolated_intercept = intercept.copy()
    momentum_time = 1.0
    regularization = 0.01
    intercept_regularization = 0.0001
    kernel_bound = float(np.max(np.sum(kernel, axis=1)))
    lipschitz_bound = 0.5 * float(np.max(sample_weights)) * (kernel_bound * kernel_bound / n + 1.0) + regularization
    step = 1.0 / lipschitz_bound
    for iteration in range(1600):
        logits = kernel.dot(extrapolated_coefficients) + extrapolated_intercept[None, :]
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        residual = (probabilities - target) * (sample_weights[:, None] / n)
        coefficient_gradient = kernel.T.dot(residual) + regularization * extrapolated_coefficients
        intercept_gradient = np.sum(residual, axis=0) + intercept_regularization * (extrapolated_intercept - prior)
        next_coefficients = extrapolated_coefficients - step * coefficient_gradient
        next_intercept = extrapolated_intercept - step * intercept_gradient
        next_time = (1.0 + np.sqrt(1.0 + 4.0 * momentum_time * momentum_time)) / 2.0
        momentum = (momentum_time - 1.0) / next_time
        extrapolated_coefficients = next_coefficients + momentum * (next_coefficients - coefficients)
        extrapolated_intercept = next_intercept + momentum * (next_intercept - intercept)
        coefficients = next_coefficients
        intercept = next_intercept
        momentum_time = next_time
    logits = kernel.dot(coefficients) + intercept[None, :]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    residual = (probabilities - target) * (sample_weights[:, None] / n)
    coefficient_gradient = kernel.T.dot(residual) + regularization * coefficients
    intercept_gradient = np.sum(residual, axis=0) + intercept_regularization * (intercept - prior)
    objective = -float(np.sum(sample_weights[:, None] * target * log_probabilities) / n)
    objective += 0.5 * regularization * float(np.sum(coefficients * coefficients))
    objective += 0.5 * intercept_regularization * float(np.sum((intercept - prior) ** 2))
    gradient_norm = float(np.sqrt(np.sum(coefficient_gradient ** 2) + np.sum(intercept_gradient ** 2)))
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(intercept)) or not np.isfinite(objective) or not np.isfinite(gradient_norm):
        raise ValueError('Nonfinite joint fitted state')
    return {
        'fit_contract': 'ct-parallel-classification/1.0',
        'classes': list(CLASSES),
        'dimensions': list(DIMENSIONS),
        'medians': medians.tolist(),
        'scales': scales.tolist(),
        'centers': centers.tolist(),
        'coefficients': coefficients.tolist(),
        'intercept': intercept.tolist(),
        'training_count': int(n),
        'class_counts': {CLASSES[index]: int(counts[index]) for index in range(4)},
        'optimizer_iterations': 1600,
        'training_objective': float(objective),
        'training_gradient_norm': float(gradient_norm),
    }

def predict_multiclass(features, state):
    rows = []
    for name, dimension in zip(GROUPS, DIMENSIONS):
        values = features[name]
        if len(values) != dimension:
            raise ValueError('Prediction feature dimensions do not match the contract')
        if any(value is not None and not np.isfinite(float(value)) for value in values):
            raise ValueError('Raw prediction features must be finite numbers or None')
        rows.append(values)
    q1, q2, q3 = rows
    gate_values = (q1[0], q1[1], q1[30], q1[31], q2[96], q2[97], q2[98], q2[99])
    gate_known = all(value is not None for value in gate_values)
    if gate_known:
        static_a, static_b, exit_a, exit_b, route_a, route_b, route_min, last_distance = [float(value) for value in gate_values]
        static_pass = 0.8 <= static_a <= 1.0 and 0.8 <= static_b <= 1.0
        route_pass = 0.95 <= route_a <= 1.0 and 0.95 <= route_b <= 1.0 and 0.95 <= route_min <= 1.0 and route_min == min(route_a, route_b) and 0.0 <= last_distance <= 3.0
        if static_pass and exit_a == 1.0 and exit_b == 1.0 and route_pass:
            return {'class_scores': {'U': 0.0, 'C': 1.0, 'I_E': 0.0, 'I_P': 0.0}, 'reason_codes': ['NATIVE_STATIC_SUPPORT_AND_TWO_OBSERVED_EXIT_ROUTE_GATE']}
    raw = np.concatenate([np.asarray(values, dtype=float) for values in rows])
    observed = np.isfinite(raw)
    medians = np.asarray(state['medians'], dtype=float)
    scales = np.asarray(state['scales'], dtype=float)
    filled = np.where(observed, raw, medians)
    transformed = (2.0 / np.pi) * np.arctan((filled - medians) / scales)
    parts = []
    start = 0
    for dimension in DIMENSIONS:
        stop = start + dimension
        group_weight = float(np.sqrt(1.0 / (3.0 * dimension)))
        parts.append(transformed[start:stop] * group_weight)
        parts.append((~observed[start:stop]).astype(float) * (0.5 * group_weight))
        start = stop
    vector = np.concatenate(parts)
    centers = np.asarray(state['centers'], dtype=float)
    distance_squared = np.sum((centers - vector[None, :]) ** 2, axis=1)
    similarities = (np.exp(-distance_squared / 0.125) + np.exp(-distance_squared / 0.5) + np.exp(-distance_squared / 2.0)) / 3.0
    logits = similarities.dot(np.asarray(state['coefficients'], dtype=float)) + np.asarray(state['intercept'], dtype=float)
    exponentials = np.exp(logits - np.max(logits))
    probabilities = exponentials / np.sum(exponentials)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('Nonfinite joint class probabilities')
    reasons = ['JOINT_FOUR_CLASS_GEOMETRY_MODEL', 'FIXED_TEMPERATURE_1']
    reasons.append('NATIVE_ROUTE_GATE_NOT_SATISFIED' if gate_known else 'NATIVE_ROUTE_GATE_EVIDENCE_UNKNOWN')
    return {'class_scores': {CLASSES[index]: float(probabilities[index]) for index in range(4)}, 'reason_codes': reasons}
