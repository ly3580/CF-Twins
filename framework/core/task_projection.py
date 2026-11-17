from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from shapely import wkb as shapely_wkb
from shapely.geometry import box
from shapely.ops import transform
from .geometry import source_to_task_local_anchor_xy
from .task_region_geometry import canonicalize_task_region, make_region_id
from .versioning import MapRepresentationRole, ProvenanceStatus
_STATIC_GEOMETRY_REQUIRED_COLUMNS = {'geometry_id', 'map_id', 'geometry_type', 'static_class', 'geometry_wkb', 'geometry_status'}
_TOPOLOGY_NODE_REQUIRED_COLUMNS = {'node_id', 'map_id', 'x_m', 'y_m', 'static_class', 'node_status'}
_TOPOLOGY_EDGE_REQUIRED_COLUMNS = {'edge_id', 'map_id', 'source_node_id', 'target_node_id', 'length_m', 'directed', 'edge_status'}

@dataclass(frozen=True)
class TaskFrame:
    """The platform-neutral topology anchor that defines one task-local frame."""
    source_x_m: float
    source_y_m: float
    source_yaw_rad: float
    status: str

    def __post_init__(self) -> None:
        coordinates = np.asarray([self.source_x_m, self.source_y_m, self.source_yaw_rad], dtype=np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError('TaskFrame source pose must be finite')
        if not -math.pi <= float(self.source_yaw_rad) < math.pi:
            raise ValueError('TaskFrame source_yaw_rad must be in [-pi, pi)')
        if self.status != 'VALID':
            raise ValueError('TaskFrame status must be VALID')

    @property
    def source_xy(self) -> np.ndarray:
        """Return the two-dimensional anchor location in source metres."""
        return np.asarray([self.source_x_m, self.source_y_m], dtype=np.float64)

@dataclass(frozen=True)
class ProjectedTaskRegion:
    """A TaskRegion canonicalized after its source-to-task-local projection."""
    region_id: str
    geometry_type: str
    geometry_wkb: bytes

@dataclass(frozen=True)
class ProjectedCommonMap:
    """In-memory COMMON_NEUTRAL map content expressed in one task frame."""
    static_geometry: pd.DataFrame
    topology_nodes: pd.DataFrame
    topology_edges: pd.DataFrame

def _require_columns(frame: pd.DataFrame, required: set, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{name} must be a pandas DataFrame')
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')

def _common_role(role: MapRepresentationRole) -> MapRepresentationRole:
    try:
        resolved = role if isinstance(role, MapRepresentationRole) else MapRepresentationRole(role)
    except (TypeError, ValueError) as error:
        raise ValueError('representation_role must be a known map role') from error
    if resolved is not MapRepresentationRole.COMMON_NEUTRAL:
        raise ValueError(f'{resolved.value} cannot enter a COMMON_NEUTRAL StructureView')
    return resolved

def _task_transform(task_frame: TaskFrame):

    def project(x, y, z=None):
        points = np.stack([np.asarray(x), np.asarray(y)], axis=-1)
        local = source_to_task_local_anchor_xy(points, task_frame.source_xy, task_frame.source_yaw_rad)
        return (local[..., 0], local[..., 1])
    return project

def _project_geometry(geometry, task_frame: TaskFrame):
    return transform(_task_transform(task_frame), geometry)

def project_task_region_to_task_local(*, task_id: str, region_role: str, region_index: int, source_geometry, task_frame: TaskFrame) -> ProjectedTaskRegion:
    """Project then canonicalize a TaskRegion using Task 1's frozen interface."""
    if not isinstance(task_frame, TaskFrame):
        raise TypeError('task_frame must be a TaskFrame')
    projected = _project_geometry(source_geometry, task_frame)
    canonical = canonicalize_task_region(projected)
    canonical_geometry = shapely_wkb.loads(canonical.wkb)
    return ProjectedTaskRegion(region_id=make_region_id(task_id, region_role, region_index, canonical_geometry), geometry_type=canonical.geometry_type, geometry_wkb=canonical.wkb)

def _crop_bounds(crop_bounds_task_local: Optional[Tuple[float, float, float, float]]):
    if crop_bounds_task_local is None:
        return None
    if len(crop_bounds_task_local) != 4:
        raise ValueError('crop_bounds_task_local must have four values')
    bounds = tuple((float(value) for value in crop_bounds_task_local))
    if not np.isfinite(np.asarray(bounds, dtype=np.float64)).all():
        raise ValueError('crop_bounds_task_local must be finite')
    min_x, min_y, max_x, max_y = bounds
    if min_x >= max_x or min_y >= max_y:
        raise ValueError('crop_bounds_task_local must have positive area')
    return box(*bounds)

def _project_static_geometry(static_geometry: pd.DataFrame, task_frame: TaskFrame, common_map_id: str, crop) -> pd.DataFrame:
    projected_rows = []
    if static_geometry['geometry_id'].duplicated().any():
        raise ValueError('static_geometry.geometry_id must be unique')
    for _, row in static_geometry.iterrows():
        geometry = shapely_wkb.loads(row['geometry_wkb'])
        projected = _project_geometry(geometry, task_frame)
        if crop is not None and (not projected.intersects(crop)):
            continue
        output = row.to_dict()
        output['map_id'] = common_map_id
        output['geometry_type'] = projected.geom_type.upper()
        output['geometry_wkb'] = shapely_wkb.dumps(projected, hex=False, output_dimension=2)
        output['geometry_status'] = ProvenanceStatus.DETERMINISTIC_DERIVED.value
        projected_rows.append(output)
    projected_rows.sort(key=lambda row: (str(row['geometry_type']), str(row['static_class']), bytes(row['geometry_wkb'])))
    for index, row in enumerate(projected_rows):
        row['geometry_id'] = f'static_geometry_{index:06d}'
    return pd.DataFrame(projected_rows, columns=static_geometry.columns).reset_index(drop=True)

def _compressed_labels(signatures: dict[str, tuple]) -> dict[str, int]:
    ordered = sorted(set(signatures.values()))
    label_by_signature = {signature: index for index, signature in enumerate(ordered)}
    return {node_id: label_by_signature[signature] for node_id, signature in signatures.items()}

def _canonical_topology_labels(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, int]:
    """Refine authorized node/edge content without consulting source IDs."""
    base_signatures = {str(row['node_id']): (float(row['x_m']), float(row['y_m']), str(row['static_class'])) for _, row in nodes.iterrows()}
    labels = _compressed_labels(base_signatures)
    for _ in range(len(nodes)):
        refined: dict[str, tuple] = {}
        for node_id in base_signatures:
            incidents = []
            for _, edge in edges.iterrows():
                source_id = str(edge['source_node_id'])
                target_id = str(edge['target_node_id'])
                length_m = float(edge['length_m'])
                directed = bool(edge['directed'])
                if directed:
                    if source_id == node_id:
                        incidents.append(('OUT', length_m, labels[target_id]))
                    if target_id == node_id:
                        incidents.append(('IN', length_m, labels[source_id]))
                elif source_id == node_id or target_id == node_id:
                    other_id = target_id if source_id == node_id else source_id
                    incidents.append(('UNDIRECTED', length_m, labels[other_id]))
            refined[node_id] = (labels[node_id], tuple(sorted(incidents)))
        labels = _compressed_labels(refined)
    if len(set(labels.values())) != len(labels):
        raise ValueError('ambiguous projected topology nodes cannot use a source ID tiebreak')
    return labels

def _project_topology(topology_nodes: pd.DataFrame, topology_edges: pd.DataFrame, task_frame: TaskFrame, common_map_id: str, crop) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = topology_nodes.copy(deep=True).reset_index(drop=True)
    if nodes['node_id'].duplicated().any():
        raise ValueError('topology_nodes.node_id must be unique')
    input_node_ids = set(nodes['node_id'])
    missing_endpoint = ~topology_edges['source_node_id'].isin(input_node_ids) | ~topology_edges['target_node_id'].isin(input_node_ids)
    if missing_endpoint.any():
        raise ValueError('topology_edges reference a missing input node')
    local_xy = source_to_task_local_anchor_xy(nodes[['x_m', 'y_m']].to_numpy(dtype=np.float64), task_frame.source_xy, task_frame.source_yaw_rad)
    nodes['x_m'] = pd.Series(local_xy[:, 0], index=nodes.index, dtype='float64')
    nodes['y_m'] = pd.Series(local_xy[:, 1], index=nodes.index, dtype='float64')
    if crop is not None:
        bounds = crop.bounds
        nodes = nodes[nodes['x_m'].between(bounds[0], bounds[2]) & nodes['y_m'].between(bounds[1], bounds[3])].copy()
    edges = topology_edges.copy(deep=True).reset_index(drop=True)
    if edges['edge_id'].duplicated().any():
        raise ValueError('topology_edges.edge_id must be unique')
    retained_source_ids = set(nodes['node_id'])
    edges = edges[edges['source_node_id'].isin(retained_source_ids) & edges['target_node_id'].isin(retained_source_ids)].copy()
    local_node_xy = {str(row['node_id']): (float(row['x_m']), float(row['y_m'])) for _, row in nodes.iterrows()}
    edges['length_m'] = pd.Series([math.hypot(local_node_xy[str(row['target_node_id'])][0] - local_node_xy[str(row['source_node_id'])][0], local_node_xy[str(row['target_node_id'])][1] - local_node_xy[str(row['source_node_id'])][1]) for _, row in edges.iterrows()], index=edges.index, dtype='float64')
    canonical_labels = _canonical_topology_labels(nodes, edges)
    ordered_source_ids = sorted(canonical_labels, key=canonical_labels.__getitem__)
    node_id_map = {source_id: f'topology_node_{index:06d}' for index, source_id in enumerate(ordered_source_ids)}
    nodes = nodes.set_index('node_id', drop=False).loc[ordered_source_ids].reset_index(drop=True)
    nodes['node_id'] = [node_id_map[source_id] for source_id in ordered_source_ids]
    nodes['map_id'] = common_map_id
    nodes['node_status'] = ProvenanceStatus.DETERMINISTIC_DERIVED.value
    edges['source_node_id'] = edges['source_node_id'].map(node_id_map)
    edges['target_node_id'] = edges['target_node_id'].map(node_id_map)
    undirected = ~edges['directed'].astype(bool)
    if undirected.any():
        endpoints = edges.loc[undirected, ['source_node_id', 'target_node_id']].apply(lambda row: sorted(row.tolist()), axis=1, result_type='expand')
        edges.loc[undirected, 'source_node_id'] = endpoints.iloc[:, 0].to_numpy()
        edges.loc[undirected, 'target_node_id'] = endpoints.iloc[:, 1].to_numpy()
    edges = edges.sort_values(['source_node_id', 'target_node_id', 'length_m', 'directed'], kind='mergesort').reset_index(drop=True)
    edges['edge_id'] = [f'topology_edge_{index:06d}' for index in range(len(edges))]
    edges['map_id'] = common_map_id
    edges['edge_status'] = ProvenanceStatus.DETERMINISTIC_DERIVED.value
    return (nodes.reset_index(drop=True), edges.reset_index(drop=True))

def project_map_to_task_local(*, task_frame: TaskFrame, representation_role: MapRepresentationRole, common_map_id: str, static_geometry: pd.DataFrame, topology_nodes: pd.DataFrame, topology_edges: pd.DataFrame, crop_bounds_task_local: Optional[Tuple[float, float, float, float]]=None) -> ProjectedCommonMap:
    """Return a task-local common map; source-native maps are hard-rejected."""
    if not isinstance(task_frame, TaskFrame):
        raise TypeError('task_frame must be a TaskFrame')
    _common_role(representation_role)
    if not isinstance(common_map_id, str) or not common_map_id.strip():
        raise ValueError('common_map_id must be a nonempty string')
    _require_columns(static_geometry, _STATIC_GEOMETRY_REQUIRED_COLUMNS, 'static_geometry')
    _require_columns(topology_nodes, _TOPOLOGY_NODE_REQUIRED_COLUMNS, 'topology_nodes')
    _require_columns(topology_edges, _TOPOLOGY_EDGE_REQUIRED_COLUMNS, 'topology_edges')
    crop = _crop_bounds(crop_bounds_task_local)
    projected_geometry = _project_static_geometry(static_geometry, task_frame, common_map_id, crop)
    projected_nodes, projected_edges = _project_topology(topology_nodes, topology_edges, task_frame, common_map_id, crop)
    return ProjectedCommonMap(static_geometry=projected_geometry, topology_nodes=projected_nodes, topology_edges=projected_edges)
