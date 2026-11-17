from __future__ import annotations
import math
from dataclasses import dataclass
from numbers import Real
import numpy as np
from shapely import wkb
from shapely.errors import GEOSException
from shapely.ops import transform

def _real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number, not bool')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result

def _points(xy: object) -> np.ndarray:
    raw = np.asarray(xy, dtype=object)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError('xy must have shape (n, 2)')
    values = [_real(value, 'xy coordinate') for value in raw.flat]
    return np.asarray(values, dtype=np.float64).reshape(raw.shape)

def _planar_road_part(geometry) -> bool:
    if geometry.is_empty or geometry.has_z or (not geometry.is_valid):
        return False
    if geometry.geom_type == 'GeometryCollection':
        return all((_planar_road_part(part) for part in geometry.geoms))
    return geometry.geom_type in ('Polygon', 'MultiPolygon', 'LineString', 'MultiLineString')

@dataclass(frozen=True)
class RegionFrame:
    """Static source-frame centre and x-axis heading; angles are radians."""
    center_x: float
    center_y: float
    heading_rad: float

    def __post_init__(self) -> None:
        for name in ('center_x', 'center_y', 'heading_rad'):
            _real(getattr(self, name), name)

    def to_local(self, xy: object) -> np.ndarray:
        source = _points(xy)
        dx = source[:, 0] - self.center_x
        dy = source[:, 1] - self.center_y
        cosine, sine = (math.cos(self.heading_rad), math.sin(self.heading_rad))
        return np.column_stack((cosine * dx + sine * dy, -sine * dx + cosine * dy))

    def to_source(self, xy: object) -> np.ndarray:
        local = _points(xy)
        cosine, sine = (math.cos(self.heading_rad), math.sin(self.heading_rad))
        return np.column_stack((self.center_x + cosine * local[:, 0] - sine * local[:, 1], self.center_y + sine * local[:, 0] + cosine * local[:, 1]))

    def project_wkb(self, hexstr: str) -> str:
        """Project polygonal road area, preserving attached planar boundary lines."""
        if not isinstance(hexstr, str):
            raise TypeError('WKB must be a hex string')
        try:
            geometry = wkb.loads(bytes.fromhex(hexstr))
        except (ValueError, TypeError, GEOSException) as exc:
            raise ValueError('invalid WKB hex string') from exc
        if geometry.geom_type not in ('Polygon', 'MultiPolygon', 'GeometryCollection') or geometry.area <= 0 or (not _planar_road_part(geometry)):
            raise ValueError('WKB requires valid nonempty planar area with optional boundary lines')

        def project(x: object, y: object, z: object=None):
            if z is not None:
                raise ValueError('three-dimensional WKB is not supported')
            x_values = np.asarray(x, dtype=np.float64)
            y_values = np.asarray(y, dtype=np.float64)
            if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
                raise ValueError('WKB coordinates must be finite')
            cosine, sine = (math.cos(self.heading_rad), math.sin(self.heading_rad))
            dx, dy = (x_values - self.center_x, y_values - self.center_y)
            return (cosine * dx + sine * dy, -sine * dx + cosine * dy)
        return wkb.dumps(transform(project, geometry), hex=True)
