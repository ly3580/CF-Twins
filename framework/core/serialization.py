from __future__ import annotations
import json
import math
import unicodedata
from collections.abc import Mapping
from typing import Any

class CanonicalEncodingError(ValueError):
    """Raised when a value has no representation in a frozen encoding."""

def _normalize_text(value: str) -> str:
    return unicodedata.normalize('NFC', value)

def _canonical_hash_value(value: Any) -> Any:
    if value is None:
        return ['null']
    if isinstance(value, bool):
        return ['bool', value]
    if isinstance(value, int):
        return ['int', str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalEncodingError('floating-point values must be finite')
        return ['float', value.hex()]
    if isinstance(value, str):
        return ['string', _normalize_text(value)]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, list):
        return ['list', [_canonical_hash_value(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        raise CanonicalEncodingError('unordered set and frozenset inputs are not supported')
    if isinstance(value, Mapping):
        normalized_items = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalEncodingError('mapping keys must be strings')
            normalized_key = _normalize_text(key)
            if normalized_key in normalized_items:
                raise CanonicalEncodingError('mapping keys have a collision after Unicode NFC normalization')
            normalized_items[normalized_key] = _canonical_hash_value(item)
        return ['mapping', [[key, normalized_items[key]] for key in sorted(normalized_items)]]
    raise CanonicalEncodingError('unsupported value type for canonical hash encoding: ' + type(value).__name__)

def canonical_hash_bytes(value: Any) -> bytes:
    """Encode ``value`` into type-tagged deterministic UTF-8 bytes.

    This representation is only for stable IDs and logical content digests.
    It must never be written in place of a real JSON manifest.
    """
    canonical = _canonical_hash_value(value)
    try:
        return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except UnicodeEncodeError as error:
        raise CanonicalEncodingError('strings must be encodable as valid UTF-8') from error

def _validated_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalEncodingError('JSON floating-point values must be finite')
        return value
    if isinstance(value, bytes):
        raise CanonicalEncodingError('bytes are not supported by deterministic JSON')
    if isinstance(value, list):
        return [_validated_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalEncodingError('JSON mapping keys must be strings')
            result[key] = _validated_json_value(item)
        return result
    raise CanonicalEncodingError('unsupported value type for deterministic JSON: ' + type(value).__name__)

def deterministic_json_bytes(value: Any) -> bytes:
    """Return compact, key-sorted UTF-8 JSON ending in exactly one LF."""
    validated = _validated_json_value(value)
    try:
        encoded = json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except UnicodeEncodeError as error:
        raise CanonicalEncodingError('strings must be encodable as valid UTF-8') from error
    return encoded + b'\n'
