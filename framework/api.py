"""Bind generated extractors to the scientific atom implementations."""
from types import MappingProxyType

from .core import q1_atomics, q1_spatial_evidence, q2_atomics, q3_atomics

_COMMON = {
    'get_local_region': q1_atomics.get_local_region,
    'align_region_structures': q1_atomics.align_region_structures,
}
_AXIS_ATOMS = {
    'q1': {
        **_COMMON,
        'get_local_task_geometry': q1_spatial_evidence.collect_spatial_evidence,
        'compare_local_tasks': q1_atomics.compare_local_tasks,
    },
    'q2': {
        **_COMMON,
        'get_local_trajectory': q2_atomics.get_local_trajectory,
        'align_trajectories': q2_atomics.align_trajectories,
        'trajectory_discrepancy': q2_atomics.trajectory_discrepancy,
    },
    'q3': {
        **_COMMON,
        'get_local_trajectory': q2_atomics.get_local_trajectory,
        'align_trajectories': q2_atomics.align_trajectories,
        'trajectory_discrepancy': q2_atomics.trajectory_discrepancy,
        'describe_display_demands': q3_atomics.describe_display_demands,
        'compare_display_demands': q3_atomics.compare_display_demands,
        'summarize_explanation_evidence': q3_atomics.summarize_explanation_evidence,
    },
}


def atoms_for(axis, trace_callback):
    """Return the read-only atom dictionary expected by output/<axis>.py."""
    if axis not in _AXIS_ATOMS:
        raise ValueError("axis must be 'q1', 'q2', or 'q3'")
    if not callable(trace_callback):
        raise TypeError('trace_callback must be callable')
    atoms = {**_AXIS_ATOMS[axis], 'ct_evidence': trace_callback}
    assert all(callable(function) for function in atoms.values())
    return MappingProxyType(atoms)
