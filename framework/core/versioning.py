from __future__ import annotations
from enum import Enum
M1R_STABLE_ID_VERSION = 'cross_embodiment_stable_id/1.0.0'
M1R_GEOMETRY_CANONICALIZATION_VERSION = 'cross_embodiment_geometry_canonicalization/1.0.0'

class ProvenanceStatus(str, Enum):
    """Closed vocabulary for native, derived, missing, and failed values."""
    NATIVE = 'NATIVE'
    PARSED = 'PARSED'
    DETERMINISTIC_DERIVED = 'DETERMINISTIC_DERIVED'
    RULE_GENERATED = 'RULE_GENERATED'
    MODEL_ESTIMATED = 'MODEL_ESTIMATED'
    PLACEHOLDER = 'PLACEHOLDER'
    UNKNOWN = 'UNKNOWN'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    NOT_OBSERVED = 'NOT_OBSERVED'
    MODEL_FAILED = 'MODEL_FAILED'
    NATIVE_NULL = 'NATIVE_NULL'

class RegionRole(str, Enum):
    ENTRY = 'ENTRY'
    GOAL = 'GOAL'
    APPROACH = 'APPROACH'
    ALTERNATIVE_EXIT = 'ALTERNATIVE_EXIT'

class MapRepresentationRole(str, Enum):
    SOURCE_NATIVE = 'SOURCE_NATIVE'
    COMMON_NEUTRAL = 'COMMON_NEUTRAL'
    PLATFORM_FEASIBILITY = 'PLATFORM_FEASIBILITY'
