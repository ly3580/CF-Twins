from __future__ import annotations
import math
from numbers import Integral, Real
import numpy as np
SOURCE_FIELDS = frozenset('source_sample_id source_sample_index sample_time_ns time_basis time_evidence_status pose_time_ns pose_time_status reference_point source_pose_reference source_frame source_x_m source_y_m map_x_m map_y_m source_z_m source_z_status source_yaw_rad source_yaw_status source_qw source_qx source_qy source_qz source_orientation_status body_pose_status'.split())
LOCAL_FIELDS = frozenset('z_m z_status z_unknown_reason yaw_rad yaw_status qw qx qy qz orientation_status'.split())
_Q = ('qw', 'qx', 'qy', 'qz')

def _int(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(label + ' must be an integer')
    return int(value)

def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or (not math.isfinite(value)):
        raise ValueError(label + ' must be finite')
    return float(value)

def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ' must be nonempty text')
    return value

def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _quaternion(values, unit=False):
    q = np.array([_number(v, 'quaternion') for v in values], dtype=np.float64)
    scale = float(np.max(np.abs(q)))
    if scale == 0:
        raise ValueError('zero source quaternion')
    norm = scale * float(np.linalg.norm(q / scale))
    if unit and abs(norm - 1.0) > 1e-05:
        raise ValueError('source quaternion norm differs from unit')
    return q / norm

def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w * v - x * i - y * j - z * k, w * i + x * v + y * k - z * j, w * j - x * k + y * v + z * i, w * k + x * j - y * i + z * v])

def _heading(q):
    w, x, y, z = q
    xx, yy = (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z))
    return None if math.hypot(xx, yy) <= 1e-12 else math.atan2(yy, xx)

def project_sample_fields(source_record, region):
    """Project orientation into a static frame; unknown anchor Z stays null."""
    validate_source_fields(source_record)
    heading = _number(region.get('heading_rad'), 'region heading')
    result = dict(z_m=None, z_status='UNKNOWN_ANCHOR_ELEVATION', z_unknown_reason='ANCHOR_Z_NOT_ESTABLISHED', yaw_rad=None, yaw_status='UNKNOWN', qw=None, qx=None, qy=None, qz=None, orientation_status=source_record['source_orientation_status'])
    is_tg = source_record['source_frame'] == 'TARTANAIR_NED'
    is_ego = source_record['source_pose_reference'] == 'NATIVE_DATABASE_EGO_POSE_NOT_VERIFIED_REAR_AXLE_SE3'
    anchor_z = region.get('anchor_z_m')
    if is_ego:
        result.update(z_status='UNKNOWN_REFERENCE_POINT', z_unknown_reason='SOURCE_NOT_VERIFIED_REAR_AXLE_SE3', orientation_status='UNKNOWN_REFERENCE_POINT')
    elif anchor_z is not None:
        height = -source_record['source_z_m'] if is_tg else source_record['source_z_m']
        result.update(z_m=height - _number(anchor_z, 'anchor Z'), z_status='KNOWN', z_unknown_reason=None)
    if source_record['source_yaw_status'] == 'KNOWN':
        yaw = -source_record['source_yaw_rad'] if is_tg else source_record['source_yaw_rad']
        result.update(yaw_rad=_wrap(yaw - heading), yaw_status='KNOWN')
    if is_tg:
        q = _quaternion([source_record['source_' + k] for k in _Q], unit=True)
        nwu = _multiply([0.0, 1.0, 0.0, 0.0], q)
        local = _multiply([math.cos(heading / 2), 0.0, 0.0, -math.sin(heading / 2)], nwu)
        for v in local:
            if abs(v) > 1e-14:
                if v < 0:
                    local = -local
                break
        result.update(**{k: float(v) for k, v in zip(_Q, local)})
    return result

def validate_source_fields(record):
    """Validate one closed, region-independent source record without source IO.

    This validates recorded capabilities, not a claim of physical calibration.
    Source XY/time joins are checked when the row is made by SourceFieldIndex.
    """
    if not isinstance(record, dict) or set(record) != SOURCE_FIELDS:
        raise ValueError('source field record shape differs')
    for key in ('source_sample_id', 'reference_point', 'source_pose_reference', 'source_frame'):
        _text(record[key], key)
    for key in ('source_sample_index', 'sample_time_ns'):
        if not 0 <= _int(record[key], key) < 2 ** 63:
            raise ValueError('source time/index outside nonnegative int64')
    for key in ('source_x_m', 'source_y_m', 'map_x_m', 'map_y_m', 'source_z_m'):
        _number(record[key], key)
    if record['source_z_status'] != 'KNOWN_SOURCE_REFERENCE':
        raise ValueError('source Z capability differs')
    if record['source_yaw_status'] == 'KNOWN':
        _number(record['source_yaw_rad'], 'source yaw')
    elif record['source_yaw_status'] != 'UNKNOWN_VERTICAL_HEADING' or record['source_yaw_rad'] is not None:
        raise ValueError('source yaw status/value differs')
    reference = record['source_pose_reference']
    is_tg = reference == 'lcam_front'
    is_ego = reference == 'NATIVE_DATABASE_EGO_POSE_NOT_VERIFIED_REAR_AXLE_SE3'
    if not is_tg and (not is_ego) and (reference != 'SOURCE_VEHICLE_BOX_CENTER'):
        raise ValueError('source reference point unsupported')
    expected_orientation = 'KNOWN_CAMERA_REFERENCE' if is_tg else 'SOURCE_REFERENCE_ONLY' if is_ego else 'SOURCE_UNAVAILABLE'
    expected_body = 'UNKNOWN_EXTRINSIC' if is_tg else 'UNKNOWN_REFERENCE_POINT' if is_ego else 'NOT_APPLICABLE_VEHICLE_BOX'
    if record['source_orientation_status'] != expected_orientation or record['body_pose_status'] != expected_body:
        raise ValueError('source orientation/reference capability differs')
    q = [record['source_' + k] for k in _Q]
    if is_tg or is_ego:
        _quaternion(q, unit=is_tg)
    elif any((v is not None for v in q)):
        raise ValueError('vehicle full orientation unavailable')
    if is_tg:
        if record['reference_point'] != 'lcam_front; robot body extrinsic UNKNOWN' or record['source_frame'] != 'TARTANAIR_NED' or record['time_basis'] != 'DERIVED_RELATIVE_NS_FROM_SECONDS' or (record['time_evidence_status'] not in ('USER_CONFIRMED', 'NOMINAL_DERIVED')) or (record['pose_time_ns'] is not None) or (record['pose_time_status'] != 'ABSOLUTE_TIME_UNKNOWN') or (record['map_x_m'] != record['source_x_m']) or (record['map_y_m'] != -record['source_y_m']):
            raise ValueError('TG source clock/reference/map axes differ')
        expected_yaw = _heading(_quaternion(q, unit=True))
        if (expected_yaw is None) != (record['source_yaw_rad'] is None) or (expected_yaw is not None and abs(_wrap(record['source_yaw_rad'] - expected_yaw)) > 1e-12):
            raise ValueError('TG source heading differs from camera orientation')
    else:
        expected_reference = 'SDK_PARSED_REAR_AXLE_SE2' if is_ego else 'SOURCE_VEHICLE_BOX_CENTER'
        expected_pose_time = 'NATIVE_EGO_POSE_TIMESTAMP' if is_ego else 'NATIVE_LIDAR_TIMESTAMP'
        if record['reference_point'] != expected_reference or record['source_frame'] == 'TARTANAIR_NED' or record['time_basis'] != 'ABSOLUTE_NS' or (record['time_evidence_status'] != 'NATIVE_LIDAR_TIMESTAMP') or (record['pose_time_status'] != expected_pose_time) or (record['source_yaw_status'] != 'KNOWN') or (record['map_x_m'] != record['source_x_m']) or (record['map_y_m'] != record['source_y_m']):
            raise ValueError('nuPlan source clock/reference/map axes differ')
        if not 0 <= _int(record['pose_time_ns'], 'pose time') < 2 ** 63:
            raise ValueError('pose time outside nonnegative int64')
        if not is_ego and record['pose_time_ns'] != record['sample_time_ns']:
            raise ValueError('vehicle pose time differs from native frame')
    return dict(status='PASS_SOURCE_FIELD_RECORD')
