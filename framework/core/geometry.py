from __future__ import annotations
import numpy as np

def wrap_to_pi(angle_rad: float) -> np.float64:
    """Normalize one finite radian angle to the half-open range [-pi, pi)."""
    angle = np.float64(angle_rad)
    if not np.isfinite(angle):
        raise ValueError('angle_rad must be finite')
    return np.float64((angle + np.pi) % (2.0 * np.pi) - np.pi)

def _validate_xy(value: np.ndarray, field_name: str, require_single: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError(f'{field_name} last dimension must have size 2')
    if require_single and array.shape != (2,):
        raise ValueError(f'{field_name} must have shape (2,)')
    if not np.isfinite(array).all():
        raise ValueError(f'{field_name} must contain finite values')
    return array

def _rotation(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)

def source_to_task_local_xy(point_xy: np.ndarray, start_xy: np.ndarray, start_yaw_rad: float) -> np.ndarray:
    """Translate by the task start, then apply the inverse start rotation.

    ``point_xy`` may have shape ``(2,)`` or ``(..., 2)``. Positions are metres.
    """
    points = _validate_xy(point_xy, 'point_xy', require_single=False)
    start = _validate_xy(start_xy, 'start_xy', require_single=True)
    yaw = np.float64(start_yaw_rad)
    if not np.isfinite(yaw):
        raise ValueError('start_yaw_rad must be finite')
    return np.einsum('ij,...j->...i', _rotation(-yaw), points - start)

def task_local_to_source_xy(point_xy: np.ndarray, start_xy: np.ndarray, start_yaw_rad: float) -> np.ndarray:
    """Apply the start rotation, then add the task-start source position."""
    points = _validate_xy(point_xy, 'point_xy', require_single=False)
    start = _validate_xy(start_xy, 'start_xy', require_single=True)
    yaw = np.float64(start_yaw_rad)
    if not np.isfinite(yaw):
        raise ValueError('start_yaw_rad must be finite')
    return np.einsum('ij,...j->...i', _rotation(yaw), points) + start

def source_to_task_local_yaw(source_yaw_rad: float, start_yaw_rad: float) -> np.float64:
    """Express a source-world yaw relative to the frozen task-start heading."""
    source_yaw = np.float64(source_yaw_rad)
    start_yaw = np.float64(start_yaw_rad)
    if not np.isfinite(source_yaw) or not np.isfinite(start_yaw):
        raise ValueError('source_yaw_rad and start_yaw_rad must be finite')
    return wrap_to_pi(source_yaw - start_yaw)

def source_to_task_local_anchor_xy(point_xy: np.ndarray, anchor_xy: np.ndarray, anchor_yaw_rad: float) -> np.ndarray:
    """Project source points into the task frame fixed by a map anchor."""
    return source_to_task_local_xy(point_xy, anchor_xy, anchor_yaw_rad)

def task_local_to_source_anchor_xy(point_xy: np.ndarray, anchor_xy: np.ndarray, anchor_yaw_rad: float) -> np.ndarray:
    """Restore task-local points to the source frame of a map anchor."""
    return task_local_to_source_xy(point_xy, anchor_xy, anchor_yaw_rad)

def source_to_task_local_anchor_yaw(source_yaw_rad: float, anchor_yaw_rad: float) -> np.float64:
    """Express a source yaw relative to the task's map-anchor heading."""
    return source_to_task_local_yaw(source_yaw_rad, anchor_yaw_rad)
