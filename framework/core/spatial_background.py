import math
from numbers import Real
import numpy as np
from shapely import wkb
from shapely.geometry import Polygon, box
from .observation_projection import RegionFrame
from .observation_contract import _fields, _text, _int, _number, _geometry
from .observation_contract import TABLE_FIELDS as OBSERVATION_FIELDS
AGENT_CATEGORIES = {'vehicle', 'pedestrian', 'bicycle', 'ego'}
NON_AGENT_CATEGORIES = {'traffic_cone', 'barrier', 'czone_sign', 'generic_object'}
BOX_ATTRIBUTES = set('x y yaw width length z height'.split())
CELL_ATTRIBUTES = set('grid_x grid_y raw_id count z_min_m z_max_m z_mean_m cell_size_m'.split())
ITEM_FIELDS = set('source_id source_track category_name source_class geometry_model source_geometry_wkb_hex observation_wkb_hex road_wkb_hex observation_relation road_relation source_attributes'.split())
UNLOCATED_FIELDS = set('source_id source_track category_name source_class reason source_attributes'.split())
QUERY_FIELDS = set('items unlocated outside_count source_record_count query_status'.split())
TABLE_FIELDS = {'regions': OBSERVATION_FIELDS['regions'], 'instances': set('instance_id region_id first_time_ns time_basis subject_source_track'.split()), 'snapshots': set('snapshot_id region_id kind source_time_ns source_binding'.split()) | QUERY_FIELDS, 'links': set('instance_id ordinal relative_time_ns snapshot_id'.split())}

def source_class(category):
    if category in AGENT_CATEGORIES:
        return 'AGENT_CATEGORY'
    if category in NON_AGENT_CATEGORIES:
        return 'NON_AGENT_CATEGORY'
    return 'UNKNOWN_CATEGORY'

def _audit_number(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return 'INVALID_TYPE:' + type(value).__name__
    v = float(value)
    return v if math.isfinite(v) else 'NONFINITE:' + v.hex()

def _valid_number(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and math.isfinite(float(value))

def _relation(geometry):
    return 'AREA' if geometry.area > 0 else 'BOUNDARY_TOUCH'

def _box_polygon(center, angle, length, width):
    c, s = (math.cos(angle), math.sin(angle))
    return Polygon([(center[0] + c * u - s * v, center[1] + s * u + c * v) for u, v in ((length, width), (-length, width), (-length, -width), (length, -width))])

class RegionQuery:

    def __init__(self, region):
        _fields(region, TABLE_FIELDS['regions'], 'region')
        self.region = region
        self.frame = RegionFrame(region['center_x'], region['center_y'], region['heading_rad'])
        self.square = _geometry(region['observation_wkb_hex'], 'square')
        self.road = _geometry(region['structure_wkb_hex'], 'road', allow_collection=True)
        _number(region['size_m'], 'size_m')
        if region['size_m'] not in (25.0, 50.0, 75.0, 100.0):
            raise ValueError('unsupported region size')
        h = region['size_m'] / 2
        if self.square.symmetric_difference(box(-h, -h, h, h)).area > 0.0001 or not self.square.buffer(1e-06).covers(self.road):
            raise ValueError('region scopes differ from the static square')
        if region['anchor_type'] != 'STATIC_REGION_FRAME' or region['structure_status'] != 'ROAD_SCOPE_ONLY' or region['map_role'] not in ('SOURCE_NATIVE', 'DERIVED_ROAD_SCOPE'):
            raise ValueError('unsupported region evidence')

    def _item(self, geometry, source_id, track, category, classification, model, attributes):
        if not self.square.intersects(geometry):
            return None
        observation = geometry.intersection(self.square)
        road = observation.intersection(self.road)
        return dict(source_id=source_id, source_track=track, category_name=category, source_class=classification, geometry_model=model, source_geometry_wkb_hex=geometry.wkb_hex, observation_wkb_hex=observation.wkb_hex, road_wkb_hex=None if road.is_empty else road.wkb_hex, observation_relation=_relation(observation), road_relation='OUTSIDE' if road.is_empty else _relation(road), source_attributes=attributes)

    def boxes(self, rows):
        items, unlocated, seen, outside = ([], [], set(), 0)
        prepared, coordinates = ([], [])
        for row in rows:
            _fields(row, BOX_ATTRIBUTES | {'source_id', 'source_track', 'category_name'}, 'native box')
            sid, track, category = (row['source_id'], row['source_track'], row['category_name'])
            _text(sid, 'source_id')
            _text(track, 'source_track')
            _text(category, 'category')
            if sid in seen:
                raise ValueError('duplicate source identity')
            seen.add(sid)
            attributes = {k: _audit_number(row[k]) for k in sorted(BOX_ATTRIBUTES)}
            classification = source_class(category)
            if any((not _valid_number(row[k]) for k in ('x', 'y', 'yaw', 'width', 'length'))) or row['width'] <= 0 or row['length'] <= 0:
                dimensions = None
            else:
                dimensions = (row['yaw'], row['width'], row['length'])
                coordinates.append([row['x'], row['y']])
            prepared.append((sid, track, category, attributes, classification, dimensions))
        centers = self.frame.to_local(coordinates) if coordinates else ()
        bounds = self.square.bounds if coordinates else None
        center_index = 0
        for sid, track, category, attributes, classification, dimensions in prepared:
            if dimensions is None:
                unlocated.append(dict(source_id=sid, source_track=track, category_name=category, source_class=classification, reason='INVALID_PLANAR_BOX', source_attributes=attributes))
                continue
            center = centers[center_index]
            center_index += 1
            yaw, width, length = dimensions
            angle = yaw - self.region['heading_rad']
            c, s = (math.cos(angle), math.sin(angle))
            length, width = (length / 2, width / 2)
            ex, ey = (abs(c) * length + abs(s) * width, abs(s) * length + abs(c) * width)
            xmin, ymin, xmax, ymax = bounds
            if center[0] + ex < xmin or center[0] - ex > xmax or center[1] + ey < ymin or (center[1] - ey > ymax):
                outside += 1
                continue
            geometry = _box_polygon(center, angle, length, width)
            if not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
                unlocated.append(dict(source_id=sid, source_track=track, category_name=category, source_class=classification, reason='INVALID_PLANAR_BOX', source_attributes=attributes))
                continue
            item = self._item(geometry, sid, track, category, classification, 'ORIENTED_BOX_FOOTPRINT', attributes)
            if item is None:
                outside += 1
            else:
                items.append(item)
        return dict(items=items, unlocated=unlocated, outside_count=outside, source_record_count=len(seen), query_status='PARTIALLY_UNLOCATED' if unlocated else 'ALL_SOURCE_RECORDS_LOCATED')

    def cells(self, arrays, cell_size_m, labels):
        required = CELL_ATTRIBUTES - {'cell_size_m'}
        if not isinstance(arrays, dict) or set(arrays) != required:
            raise ValueError('grid fields differ')
        if any((not isinstance(a, np.ndarray) or a.ndim != 1 for a in arrays.values())) or len({a.shape for a in arrays.values()}) != 1:
            raise ValueError('invalid grid shapes')
        if not _valid_number(cell_size_m) or cell_size_m <= 0 or any((arrays[k].dtype.kind not in 'iu' for k in ('grid_x', 'grid_y', 'raw_id', 'count'))) or np.any(arrays['count'] <= 0):
            raise ValueError('invalid grid indices/count/size')
        for k in ('z_min_m', 'z_max_m', 'z_mean_m'):
            if arrays[k].dtype.kind not in 'fiu' or not np.isfinite(arrays[k]).all():
                raise ValueError('invalid grid heights')
        if np.any(arrays['z_min_m'] > arrays['z_mean_m']) or np.any(arrays['z_mean_m'] > arrays['z_max_m']):
            raise ValueError('invalid grid height bounds')
        if not isinstance(labels, list):
            raise ValueError('labels require list')
        names = {}
        for label in labels:
            if not isinstance(label, dict) or not {'raw_id', 'name'} <= set(label):
                raise ValueError('invalid label')
            _int(label['raw_id'], 'raw_id')
            _text(label['name'], 'label name')
            if label['raw_id'] in names:
                raise ValueError('duplicate label identity')
            names[label['raw_id']] = label['name']
        if not set((int(v) for v in arrays['raw_id'])) <= set(names):
            raise ValueError('grid category absent from labels')
        size = float(cell_size_m)
        items, seen, outside = ([], set(), 0)
        for i in range(len(arrays['count'])):
            gx, gy, raw = (int(arrays[k][i]) for k in ('grid_x', 'grid_y', 'raw_id'))
            sid = f'{gx}:{gy}:{raw}'
            if sid in seen:
                raise ValueError('duplicate grid source identity')
            seen.add(sid)
            corners = self.frame.to_local([[gx * size, gy * size], [(gx + 1) * size, gy * size], [(gx + 1) * size, (gy + 1) * size], [gx * size, (gy + 1) * size]])
            geometry = Polygon(corners)
            attributes = {k: int(arrays[k][i]) for k in ('grid_x', 'grid_y', 'raw_id', 'count')}
            attributes.update({k: float(arrays[k][i]) for k in ('z_min_m', 'z_max_m', 'z_mean_m')})
            attributes['cell_size_m'] = size
            item = self._item(geometry, sid, None, names[raw], 'STATIC_SEMANTIC_SAMPLE', 'GRID_SUPPORT_APPROXIMATION', attributes)
            if item is None:
                outside += 1
            else:
                items.append(item)
        return dict(items=items, unlocated=[], outside_count=outside, source_record_count=len(seen), query_status='ALL_SOURCE_RECORDS_LOCATED')

def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all((c in '0123456789abcdef' for c in value))

def _clip_geometry(value):
    _text(value, 'geometry')
    try:
        g = wkb.loads(bytes.fromhex(value))
    except Exception as exc:
        raise ValueError('invalid clipped geometry') from exc
    if g.is_empty or not g.is_valid or g.has_z or (not all((math.isfinite(v) for v in g.bounds))) or (g.geom_type not in ('Polygon', 'MultiPolygon', 'LineString', 'MultiLineString', 'Point', 'MultiPoint', 'GeometryCollection')):
        raise ValueError('invalid clipped planar geometry')
    if g.geom_type == 'GeometryCollection':
        for part in g.geoms:
            _clip_geometry(part.wkb_hex)
    return g

def _box_attributes(attributes, *, located):
    _fields(attributes, BOX_ATTRIBUTES, 'box attributes')
    for value in attributes.values():
        if value is None:
            continue
        if isinstance(value, str) and (value.startswith('NONFINITE:') or value.startswith('INVALID_TYPE:')):
            continue
        _number(value, 'box attribute')
    valid = all((_valid_number(attributes[k]) for k in ('x', 'y', 'yaw', 'width', 'length'))) and attributes['width'] > 0 and (attributes['length'] > 0)
    if valid != located:
        raise ValueError('box geometry status differs from source attributes')

def _validate_query(snapshot, query):
    if not isinstance(snapshot['items'], list) or not isinstance(snapshot['unlocated'], list):
        raise ValueError('query records require lists')
    for field in ('outside_count', 'source_record_count'):
        _int(snapshot[field], field)
        if snapshot[field] < 0:
            raise ValueError('negative query count')
    if len(snapshot['items']) + len(snapshot['unlocated']) + snapshot['outside_count'] != snapshot['source_record_count']:
        raise ValueError('source query count differs')
    expected = 'PARTIALLY_UNLOCATED' if snapshot['unlocated'] else 'ALL_SOURCE_RECORDS_LOCATED'
    if snapshot['query_status'] != expected:
        raise ValueError('false source localization claim')
    seen = set()
    for row in snapshot['items'] + snapshot['unlocated']:
        _text(row.get('source_id'), 'source identity')
        _text(row.get('category_name'), 'category')
        if row['source_id'] in seen:
            raise ValueError('duplicate query source identity')
        seen.add(row['source_id'])
    native = snapshot['kind'] == 'NUPLAN_LIDAR_FRAME'
    for row in snapshot['unlocated']:
        _fields(row, UNLOCATED_FIELDS, 'unlocated source')
        if not native or row['reason'] != 'INVALID_PLANAR_BOX' or row['source_class'] != source_class(row['category_name']):
            raise ValueError('unsupported unlocated evidence')
        _text(row['source_track'], 'track')
        _box_attributes(row['source_attributes'], located=False)
    for row in snapshot['items']:
        _fields(row, ITEM_FIELDS, 'query item')
        if native:
            _text(row['source_track'], 'track')
            _box_attributes(row['source_attributes'], located=True)
            if row['source_class'] != source_class(row['category_name']) or row['geometry_model'] != 'ORIENTED_BOX_FOOTPRINT':
                raise ValueError('false native category/geometry evidence')
        else:
            if row['source_track'] is not None or row['source_class'] != 'STATIC_SEMANTIC_SAMPLE' or row['geometry_model'] != 'GRID_SUPPORT_APPROXIMATION':
                raise ValueError('false static cell evidence')
            attrs = row['source_attributes']
            _fields(attrs, CELL_ATTRIBUTES, 'cell attributes')
            for key in ('grid_x', 'grid_y', 'raw_id', 'count'):
                _int(attrs[key], key)
            for key in ('z_min_m', 'z_max_m', 'z_mean_m', 'cell_size_m'):
                _number(attrs[key], key)
            if attrs['count'] <= 0 or attrs['cell_size_m'] <= 0 or (not attrs['z_min_m'] <= attrs['z_mean_m'] <= attrs['z_max_m']):
                raise ValueError('invalid original cell statistics')
        source = _geometry(row['source_geometry_wkb_hex'], 'source footprint')
        attrs = row['source_attributes']
        if native:
            center = query.frame.to_local([[attrs['x'], attrs['y']]])[0]
            expected_source = _box_polygon(center, attrs['yaw'] - query.region['heading_rad'], attrs['length'] / 2, attrs['width'] / 2)
        else:
            gx, gy, size = (attrs['grid_x'], attrs['grid_y'], attrs['cell_size_m'])
            if row['source_id'] != f"{gx}:{gy}:{attrs['raw_id']}":
                raise ValueError('grid source identity differs from attributes')
            expected_source = Polygon(query.frame.to_local([[gx * size, gy * size], [(gx + 1) * size, gy * size], [(gx + 1) * size, (gy + 1) * size], [gx * size, (gy + 1) * size]]))
        if not source.equals(expected_source):
            raise ValueError('source footprint differs from original source attributes')
        observation = _clip_geometry(row['observation_wkb_hex'])
        if not observation.equals(source.intersection(query.square)) or row['observation_relation'] != _relation(observation):
            raise ValueError('observation clip differs from full footprint')
        expected_road = observation.intersection(query.road)
        if row['road_wkb_hex'] is None:
            if not expected_road.is_empty or row['road_relation'] != 'OUTSIDE':
                raise ValueError('missing road intersection')
        else:
            road = _clip_geometry(row['road_wkb_hex'])
            if not road.equals(expected_road) or row['road_relation'] != _relation(road):
                raise ValueError('road clip differs from observation/road intersection')
