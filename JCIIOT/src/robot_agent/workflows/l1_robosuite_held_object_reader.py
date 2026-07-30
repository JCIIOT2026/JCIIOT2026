"""Minimal read-only adapter for RobosuiteBackend held-object observations.

This module is the only L1 workflow layer allowed to inspect the backend's
private ``_held_crate_name`` field. It deliberately does not import the
backend, MuJoCo, robosuite, skills, dispatcher, or execution stack.

The field value is raw runtime evidence, never task truth:

* a non-null value after reset can be stale and must fail an EMPTY checkpoint;
* the expected instance after pick-up and during transport is supporting
  evidence only;
* null after place-down is necessary but cannot prove stable placement,
  release completion, bounds, or task success.

The session identifier is externally injected. This Reader owns only its local
observation-order counter; backend-state sequence is unavailable and explicit
unknown. Runtime session continuity, freshness, and checkpoint semantics remain
owned by ``l1_held_object_provider``. This adapter never writes or clears
backend state.
"""

from __future__ import annotations

import math
import re
import time
import weakref
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Final

from .l1_held_object_provider import HeldObjectClock


READER_VERSION: Final = "2.0"
PRIVATE_BACKEND_FIELD: Final = "_held_crate_name"
SOURCE_NAME: Final = "robosuite_backend_private_held_object_adapter"
OFFLINE_READINESS_STATUS: Final = "REAL_HELD_OBJECT_READER_OFFLINE_READY"
SESSION_SOURCE: Final = "external"
BACKEND_STATE_SEQUENCE_STATUS: Final = "unknown"
_ERROR_MESSAGE_UNAVAILABLE: Final = "<error-message-unavailable>"
_REDACTED_PATH: Final = "<redacted-path>"
_REDACTED_CREDENTIAL: Final = "<redacted-credential>"
_REDACTED_ADDRESS: Final = "<redacted-address>"
_WINDOWS_PATH = re.compile(
    r"(?i)\b[a-z]:[\\/](?:[^\\/\s:'\"<>]+[\\/])*[^\\/\s:'\"<>]*"
)
_POSIX_PATH = re.compile(
    r"(?<![\w/])/(?:[^/\s:'\"<>]+/)+[^/\s:'\"<>]+"
)
_AUTHORIZATION_VALUE = re.compile(
    r"(?i)\bauthorization\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_NAMED_CREDENTIAL = re.compile(
    r"(?i)\b(?:token|api[_-]?key|secret)\s*[:=]\s*[^\s,;]+"
)
_SK_CREDENTIAL = re.compile(r"(?i)(?<![a-z0-9])sk-[a-z0-9._-]+")
_OBJECT_REPR_ADDRESS = re.compile(r"(?i)<[^>\r\n]*\bat\s+0x[0-9a-f]+>")
_MEMORY_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


class RobosuiteHeldObjectReaderReasonCode:
    """Stable adapter-local error categories passed to the Provider."""

    BACKEND_REFERENCE_EXPIRED = "BACKEND_REFERENCE_EXPIRED"
    BACKEND_FIELD_MISSING = "BACKEND_FIELD_MISSING"
    BACKEND_FIELD_READ_FAILED = "BACKEND_FIELD_READ_FAILED"
    BACKEND_READER_CLOCK_FAILED = "BACKEND_READER_CLOCK_FAILED"


@dataclass(frozen=True)
class RobosuiteHeldObjectReadResult:
    """Immutable Reader-local evidence envelope.

    ``read_sequence`` is only the observation order within this Reader
    instance and externally supplied session. The backend exposes no official
    state-version sequence, so that value is always null and explicitly
    marked unknown. This envelope intentionally has no generic ``sequence``
    compatibility alias.
    """

    status: str
    held_object: object
    session_id: str
    session_source: str
    read_sequence: int
    backend_state_sequence: None
    backend_state_sequence_status: str
    captured_at: str | None
    monotonic_time: float | None
    source: str
    source_kind: str
    error_type: str | None = None
    exception_type: str | None = None
    error_message_sanitized: str | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


def sanitize_reader_error_message(
    message: object,
    *,
    max_length: int = 256,
) -> str:
    """Return a deterministic, single-line, bounded, sanitized message."""

    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
    ):
        max_length = 256
    try:
        text = str(message)
    except BaseException:
        return _ERROR_MESSAGE_UNAVAILABLE[:max_length]
    text = text.replace("\r", " ").replace("\n", " ")
    text = _AUTHORIZATION_VALUE.sub(
        f"Authorization: {_REDACTED_CREDENTIAL}",
        text,
    )
    text = _BEARER_VALUE.sub(
        f"Bearer {_REDACTED_CREDENTIAL}",
        text,
    )
    text = _NAMED_CREDENTIAL.sub(_REDACTED_CREDENTIAL, text)
    text = _SK_CREDENTIAL.sub(_REDACTED_CREDENTIAL, text)
    text = _WINDOWS_PATH.sub(_REDACTED_PATH, text)
    text = _POSIX_PATH.sub(_REDACTED_PATH, text)
    text = _OBJECT_REPR_ADDRESS.sub(_REDACTED_ADDRESS, text)
    text = _MEMORY_ADDRESS.sub(_REDACTED_ADDRESS, text)
    text = " ".join(text.split())
    if not text:
        text = _ERROR_MESSAGE_UNAVAILABLE
    return text[:max_length]


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def _timestamp(clock: HeldObjectClock) -> tuple[str, float]:
    captured = clock.now()
    if not isinstance(captured, datetime):
        raise TypeError("clock.now() must return datetime")
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    monotonic_value = float(clock.monotonic())
    if not math.isfinite(monotonic_value):
        raise ValueError("clock.monotonic() must be finite")
    return captured.isoformat(), monotonic_value


class RobosuiteHeldObjectBackendReader:
    """Read exactly one private backend field per call without retaining truth.

    A weak reference avoids extending the backend lifecycle. Read sequence
    numbers are local to this reader instance and strictly increase for every
    attempted read, including failed reads. They are not backend state
    versions.
    """

    __slots__ = (
        "_backend_reference",
        "_clock",
        "_read_sequence",
        "_session_id",
    )

    def __init__(
        self,
        *,
        backend: object,
        session_id: str,
        clock: HeldObjectClock | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if backend is None:
            raise ValueError("backend must not be null")
        try:
            backend_reference = weakref.ref(backend)
        except TypeError as exc:
            raise TypeError("backend must support weak references") from exc
        self._backend_reference = backend_reference
        self._session_id = session_id.strip()
        self._clock = clock if clock is not None else _SystemClock()
        self._read_sequence = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_read_sequence(self) -> int:
        return self._read_sequence

    def _result(
        self,
        *,
        status: str,
        held_object: object,
        read_sequence: int,
        captured_at: str | None,
        monotonic_time: float | None,
        error_type: str | None = None,
        exception_type: str | None = None,
        error_message: object | None = None,
    ) -> RobosuiteHeldObjectReadResult:
        sanitized_error = (
            None
            if error_message is None
            else sanitize_reader_error_message(error_message)
        )
        return RobosuiteHeldObjectReadResult(
            status=status,
            held_object=held_object,
            session_id=self._session_id,
            session_source=SESSION_SOURCE,
            read_sequence=read_sequence,
            backend_state_sequence=None,
            backend_state_sequence_status=BACKEND_STATE_SEQUENCE_STATUS,
            captured_at=captured_at,
            monotonic_time=monotonic_time,
            source=SOURCE_NAME,
            source_kind="backend_adapter",
            error_type=error_type,
            exception_type=exception_type,
            error_message_sanitized=sanitized_error,
            evidence=(
                "one read of the isolated backend held-object field",
                "read_sequence is reader observation order only",
                "backend_state_sequence is unavailable and unknown",
                "raw observation only; not task truth",
                "adapter performs no backend write or cleanup",
            ),
        )

    def read_held_object_name(self) -> RobosuiteHeldObjectReadResult:
        """Return one metadata envelope; never normalize or mutate the value."""

        self._read_sequence += 1
        read_sequence = self._read_sequence
        try:
            captured_at, monotonic_time = _timestamp(self._clock)
        except Exception as exc:
            return self._result(
                status="error",
                held_object=None,
                read_sequence=read_sequence,
                captured_at=None,
                monotonic_time=None,
                error_type=(
                    RobosuiteHeldObjectReaderReasonCode
                    .BACKEND_READER_CLOCK_FAILED
                ),
                exception_type=type(exc).__name__,
                error_message=exc,
            )

        backend = self._backend_reference()
        if backend is None:
            return self._result(
                status="error",
                held_object=None,
                read_sequence=read_sequence,
                captured_at=captured_at,
                monotonic_time=monotonic_time,
                error_type=(
                    RobosuiteHeldObjectReaderReasonCode
                    .BACKEND_REFERENCE_EXPIRED
                ),
                error_message="backend reference is no longer alive",
            )

        missing = object()
        try:
            raw_value = getattr(backend, PRIVATE_BACKEND_FIELD, missing)
        except Exception as exc:
            return self._result(
                status="error",
                held_object=None,
                read_sequence=read_sequence,
                captured_at=captured_at,
                monotonic_time=monotonic_time,
                error_type=(
                    RobosuiteHeldObjectReaderReasonCode
                    .BACKEND_FIELD_READ_FAILED
                ),
                exception_type=type(exc).__name__,
                error_message=exc,
            )
        if raw_value is missing:
            return self._result(
                status="error",
                held_object=None,
                read_sequence=read_sequence,
                captured_at=captured_at,
                monotonic_time=monotonic_time,
                error_type=(
                    RobosuiteHeldObjectReaderReasonCode
                    .BACKEND_FIELD_MISSING
                ),
                error_message="backend held-object field is missing",
            )

        return self._result(
            status="known",
            held_object=raw_value,
            read_sequence=read_sequence,
            captured_at=captured_at,
            monotonic_time=monotonic_time,
        )


def reader_contract_summary() -> dict[str, object]:
    """Return deterministic, non-runtime contract facts for offline audits."""

    return {
        "reader_version": READER_VERSION,
        "status_ceiling": OFFLINE_READINESS_STATUS,
        "backend_field": PRIVATE_BACKEND_FIELD,
        "source_kind": "backend_adapter",
        "read_only": True,
        "writes_backend": False,
        "clears_stale_state": False,
        "caches_held_object_value": False,
        "retains_backend_strong_reference": False,
        "raw_field_is_task_truth": False,
        "session_source": SESSION_SOURCE,
        "read_sequence_semantics": "reader_observation_order",
        "read_sequence_is_backend_state_sequence": False,
        "backend_state_sequence": None,
        "backend_state_sequence_status": BACKEND_STATE_SEQUENCE_STATUS,
        "generic_sequence_field_present": False,
        "runtime_provider_connected": False,
        "reset_nonempty_must_block_empty_checkpoint": True,
        "post_pick_expected_instance_is_supporting_evidence": True,
        "transport_requires_continuous_expected_instance": True,
        "post_place_null_is_necessary_not_sufficient": True,
        "requires_skill_result_for_place": True,
        "requires_post_place_verification": True,
        "provider_owns_runtime_session_sequence_freshness": True,
        "provider_owns_eight_checkpoints": True,
        "execute_true_allowed": False,
        "runtime_verified": False,
        "real_dispatch_ready": False,
        "physical_execution_ready": False,
    }


__all__ = [
    "OFFLINE_READINESS_STATUS",
    "PRIVATE_BACKEND_FIELD",
    "READER_VERSION",
    "RobosuiteHeldObjectBackendReader",
    "RobosuiteHeldObjectReaderReasonCode",
    "reader_contract_summary",
]
