from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Tuple
from shapely import wkb as shapely_wkb
from shapely.geometry import MultiPolygon, Polygon
from .identifiers import stable_id
from .versioning import M1R_GEOMETRY_CANONICALIZATION_VERSION, RegionRole
Point2D = Tuple[float, float]
RingCoordinates = Tuple[Point2D, ...]
PolygonCoordinates = Tuple[RingCoordinates, Tuple[RingCoordinates, ...]]
CanonicalCoordinates = Tuple[PolygonCoordinates, ...]
_TASK_ID_PATTERN = re.compile('^task_[0-9a-f]{64}$')

class TaskRegionGeometryError(ValueError):
    """Raised when a geometry cannot enter the frozen TaskRegion contract."""

@dataclass(frozen=True)
class CanonicalTaskRegion:
    """Immutable identity coordinates plus fixed storage WKB."""
    geometry_type: str
    canonical_coordinates: CanonicalCoordinates
    wkb: bytes

def _normalize_coordinate(coordinate: Any) -> Point2D:
    if len(coordinate) != 2:
        raise TaskRegionGeometryError('TaskRegion coordinates must be exactly 2D')
    x = float(coordinate[0])
    y = float(coordinate[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise TaskRegionGeometryError('TaskRegion coordinates must be finite float64')
    if x == 0.0:
        x = 0.0
    if y == 0.0:
        y = 0.0
    return (x, y)

def _signed_area(ring: RingCoordinates) -> float:
    return 0.5 * sum((x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])))

def _minimum_rotation(ring: RingCoordinates) -> RingCoordinates:
    minimum_vertex = min(ring)
    candidates = []
    for index, point in enumerate(ring):
        if point == minimum_vertex:
            candidates.append(ring[index:] + ring[:index])
    return min(candidates)

def _canonical_ring(coordinates: Any, counter_clockwise: bool) -> RingCoordinates:
    points = [_normalize_coordinate(coordinate) for coordinate in coordinates]
    without_consecutive_duplicates = []
    for point in points:
        if not without_consecutive_duplicates or point != without_consecutive_duplicates[-1]:
            without_consecutive_duplicates.append(point)
    while len(without_consecutive_duplicates) > 1 and without_consecutive_duplicates[-1] == without_consecutive_duplicates[0]:
        without_consecutive_duplicates.pop()
    ring = tuple(without_consecutive_duplicates)
    if len(set(ring)) < 3:
        raise TaskRegionGeometryError('TaskRegion rings must contain at least three distinct vertices')
    signed_area = _signed_area(ring)
    if signed_area == 0.0:
        raise TaskRegionGeometryError('TaskRegion rings must have non-zero area')
    if (signed_area > 0.0) != counter_clockwise:
        ring = tuple(reversed(ring))
    return _minimum_rotation(ring)

def _canonical_polygon(polygon: Polygon) -> PolygonCoordinates:
    exterior = _canonical_ring(polygon.exterior.coords, counter_clockwise=True)
    holes = tuple(sorted((_canonical_ring(interior.coords, counter_clockwise=False) for interior in polygon.interiors)))
    return (exterior, holes)

def _closed(ring: RingCoordinates) -> Tuple[Point2D, ...]:
    return ring + (ring[0],)

def _geometry_from_coordinates(geometry_type: str, coordinates: CanonicalCoordinates) -> Any:
    polygons = [Polygon(_closed(exterior), [_closed(hole) for hole in holes]) for exterior, holes in coordinates]
    if geometry_type == 'Polygon':
        return polygons[0]
    return MultiPolygon(polygons)

def _fixed_wkb(geometry: Any) -> bytes:
    try:
        return shapely_wkb.dumps(geometry, hex=False, big_endian=False, output_dimension=2, include_srid=False)
    except TypeError:
        return shapely_wkb.dumps(geometry, hex=False, big_endian=False, output_dimension=2)

def canonicalize_task_region(geometry: Any) -> CanonicalTaskRegion:
    """Validate and canonicalize one non-empty 2D Polygon/MultiPolygon."""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TaskRegionGeometryError('TaskRegion geometry must be a Polygon or MultiPolygon')
    if geometry.is_empty:
        raise TaskRegionGeometryError('TaskRegion geometry must be non-empty')
    if getattr(geometry, 'has_z', False):
        raise TaskRegionGeometryError('TaskRegion geometry must be exactly 2D')
    if not geometry.is_valid:
        raise TaskRegionGeometryError('TaskRegion geometry must be valid')
    if not math.isfinite(float(geometry.area)) or float(geometry.area) <= 0.0:
        raise TaskRegionGeometryError('TaskRegion geometry must have positive finite area')
    geometry_type = geometry.geom_type
    source_polygons = [geometry] if geometry_type == 'Polygon' else list(geometry.geoms)
    if not source_polygons:
        raise TaskRegionGeometryError('TaskRegion MultiPolygon must contain a polygon')
    coordinates = tuple(sorted((_canonical_polygon(polygon) for polygon in source_polygons)))
    canonical_geometry = _geometry_from_coordinates(geometry_type, coordinates)
    if not canonical_geometry.is_valid or canonical_geometry.is_empty:
        raise TaskRegionGeometryError('canonical TaskRegion geometry is invalid')
    return CanonicalTaskRegion(geometry_type=geometry_type, canonical_coordinates=coordinates, wkb=_fixed_wkb(canonical_geometry))

def make_region_id(task_id: str, region_role: Any, region_index: int, geometry: Any) -> str:
    """Create a stable region ID from task scope and canonical geometry."""
    if not isinstance(task_id, str) or not _TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskRegionGeometryError('task_id must be a namespaced task_<64 lowercase hex> ID')
    try:
        role = region_role if isinstance(region_role, RegionRole) else RegionRole(region_role)
    except (TypeError, ValueError) as error:
        raise TaskRegionGeometryError('region_role is not in the closed vocabulary') from error
    if isinstance(region_index, bool) or not isinstance(region_index, int) or region_index < 0:
        raise TaskRegionGeometryError('region_index must be a non-negative integer')
    canonical = canonicalize_task_region(geometry)
    canonical_coordinates_as_lists = [[[[x, y] for x, y in exterior], [[[x, y] for x, y in hole] for hole in holes]] for exterior, holes in canonical.canonical_coordinates]
    payload = {'task_id': task_id, 'region_role': role.value, 'region_index': region_index, 'geometry_canonicalization_version': M1R_GEOMETRY_CANONICALIZATION_VERSION, 'geometry_type': canonical.geometry_type, 'canonical_coordinates': canonical_coordinates_as_lists}
    return stable_id('region', M1R_GEOMETRY_CANONICALIZATION_VERSION, payload)
