from __future__ import annotations
from copy import deepcopy
import json
from typing import Dict, List
from .observation_contract import validate_observation_bundle

class _ProgramView:
    __slots__ = ('_snapshot', '_names')

    def __init__(self, tables: Dict[str, List[dict]]) -> None:
        object.__setattr__(self, '_snapshot', json.dumps(tables, allow_nan=False).encode('utf-8'))
        object.__setattr__(self, '_names', tuple(tables))

    def __getattribute__(self, name):
        if name.startswith('_') and name != '__class__':
            raise AttributeError(f'private view attribute {name!r} is inaccessible')
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        raise AttributeError('program views are immutable')

    def __delattr__(self, name):
        raise AttributeError('program views are immutable')

    @property
    def table_names(self):
        return object.__getattribute__(self, '_names')

    def table(self, name: str) -> List[dict]:
        if not isinstance(name, str):
            raise ValueError('table name must be text')
        if name not in object.__getattribute__(self, '_names'):
            raise KeyError(f'table {name!r} is not allowed in this view')
        return json.loads(object.__getattribute__(self, '_snapshot'))[name]

class StructureView(_ProgramView):
    """Static, region-local geometry and its evidence status."""
    __slots__ = ()

class TaskView(_ProgramView):
    """Task conditions available at the first internal observation."""
    __slots__ = ()

class TaskContextView(_ProgramView):
    """The union of static structure and task conditions."""
    __slots__ = ()

class DecisionView(_ProgramView):
    """Task context and the complete observed, region-local decision."""
    __slots__ = ()

def _relative(value, first_time_ns):
    if value is None:
        return None
    return value - first_time_ns

class ObservationScene:
    """Build allowlisted snapshots; source-audit fields are discarded."""

    def __init__(self, bundle: dict) -> None:
        validate_observation_bundle(bundle)
        self._structures = {}
        self._tasks = {}
        self._decisions = {}
        for region in bundle['regions']:
            region_id = region['region_id']
            if region_id in self._structures:
                raise ValueError(f'duplicate region_id: {region_id}')
            self._structures[region_id] = {'region_spec': [{key: region[key] for key in ('anchor_type', 'size_m', 'observation_wkb_hex', 'structure_wkb_hex', 'map_role', 'structure_status')}], 'static_geometry': [{'geometry_wkb_hex': region['structure_wkb_hex']}]}
        samples_by_instance = {}
        for sample in bundle['samples']:
            samples_by_instance.setdefault(sample['instance_id'], []).append(sample)
        events_by_instance = {}
        for event in bundle['events']:
            events_by_instance.setdefault(event['instance_id'], []).append(event)
        for instance in bundle['instances']:
            instance_id = instance['instance_id']
            if instance_id in self._tasks:
                raise ValueError(f'duplicate instance_id: {instance_id}')
            region_id = instance['region_id']
            if region_id not in self._structures:
                raise ValueError(f'unknown region_id: {region_id}')
            samples = sorted(samples_by_instance.get(instance_id, []), key=lambda row: row['ordinal'])
            if not samples or samples[0]['ordinal'] != 0:
                raise ValueError(f'instance {instance_id} lacks its first sample')
            first = instance['first_time_ns']
            self._tasks[instance_id] = {'task_spec': [{'region_id': region_id, 'policy_version': bundle['policy_version'], 'cap_ns': bundle['cap_ns'], 'start_kind': instance['start_kind'], 'entry_road_status': instance['entry_road_status'], 'reference_point': instance['reference_point'], 'time_basis': instance['time_basis'], 'first_x_m': samples[0]['x_m'], 'first_y_m': samples[0]['y_m'], 'entry_lower_relative_ns': _relative(instance['entry_lower_ns'], first), 'entry_upper_relative_ns': _relative(instance['entry_upper_ns'], first)}]}
            self._decisions[instance_id] = {'observed_trajectory': [{key: sample[key] for key in ('ordinal', 'relative_time_ns', 'x_m', 'y_m', 'road_relation', 'square_relation')} for sample in samples], 'task_events': [{key: event[key] for key in ('event_type', 'lower_relative_ns', 'upper_relative_ns', 'censored')} for event in events_by_instance.get(instance_id, [])], 'observation_end': [{'end_reason': instance['end_reason'], 'right_censored': instance['right_censored'], 'last_relative_ns': _relative(instance['last_time_ns'], first), 'deadline_relative_ns': _relative(instance['deadline_ns'], first), 'stop_lower_relative_ns': _relative(instance['stop_lower_ns'], first), 'stop_upper_relative_ns': _relative(instance['stop_upper_ns'], first), 'outcome_status': instance['outcome_status']}]}
            self._decisions[instance_id].update(self._structures[region_id])
            self._decisions[instance_id].update(self._tasks[instance_id])
        self._structures = deepcopy(self._structures)
        self._tasks = deepcopy(self._tasks)
        self._decisions = deepcopy(self._decisions)
        self._instance_regions = {row['instance_id']: row['region_id'] for row in bundle['instances']}

    def structure_view(self, region_id: str) -> StructureView:
        return StructureView(self._structures[region_id])

    def task_view(self, instance_id: str) -> TaskView:
        return TaskView(self._tasks[instance_id])

    def task_context_view(self, instance_id: str) -> TaskContextView:
        tables = dict(self._structures[self._instance_regions[instance_id]])
        tables.update(self._tasks[instance_id])
        return TaskContextView(tables)

    def decision_view(self, instance_id: str) -> DecisionView:
        return DecisionView(self._decisions[instance_id])
