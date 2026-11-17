from __future__ import annotations
import hashlib
import re
from collections.abc import Mapping
from .serialization import CanonicalEncodingError, canonical_hash_bytes
from .versioning import M1R_STABLE_ID_VERSION
_ENTITY_KIND_PATTERN = re.compile('^[a-z][a-z0-9_]*$')

class StableIdentityError(ValueError):
    """Raised when an identity envelope or one of its inputs is invalid."""

def _require_nonempty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StableIdentityError(f'{field_name} must be a non-empty string')
    return value

def stable_id(entity_kind: str, identity_version: str, payload: Mapping) -> str:
    """Hash a payload under distinct algorithm and entity identity versions."""
    if not isinstance(entity_kind, str) or not _ENTITY_KIND_PATTERN.fullmatch(entity_kind):
        raise StableIdentityError('entity_kind must match ^[a-z][a-z0-9_]*$')
    _require_nonempty_text(identity_version, 'identity_version')
    if not isinstance(payload, Mapping):
        raise StableIdentityError('payload must be a mapping')
    envelope = {'entity_kind': entity_kind, 'stable_id_version': M1R_STABLE_ID_VERSION, 'identity_version': identity_version, 'payload': payload}
    try:
        digest = hashlib.sha256(canonical_hash_bytes(envelope)).hexdigest()
    except CanonicalEncodingError as error:
        raise StableIdentityError(str(error)) from error
    return f'{entity_kind}_{digest}'
