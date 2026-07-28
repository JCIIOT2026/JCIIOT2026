"""Isolated, read-only L1 held-object snapshot and checkpoint contracts.

This module is intentionally independent from the robot backend and execution
stack. A future adapter or process bridge may implement the reader protocol;
only injected in-memory readers are used by the current offline validation.
"""

from __future__ import annotations

import copy
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


PROVIDER_VERSION = "1.0"
EXPECTED_L1_SKILLS = ("move", "pick_up", "move", "place_down")
VALID_SOURCE_KINDS = frozenset({"mock", "backend_adapter", "process_bridge"})
VALID_READ_STATUSES = frozenset({"known", "unknown", "error"})


class HeldObjectProviderReasonCode:
    """Stable reason codes for fail-closed held-object validation."""

    HELD_OBJECT_READER_MISSING = "HELD_OBJECT_READER_MISSING"
    HELD_OBJECT_READ_FAILED = "HELD_OBJECT_READ_FAILED"
    HELD_OBJECT_VALUE_INVALID = "HELD_OBJECT_VALUE_INVALID"
    HELD_OBJECT_STATE_UNKNOWN = "HELD_OBJECT_STATE_UNKNOWN"
    HELD_OBJECT_STATE_ERROR = "HELD_OBJECT_STATE_ERROR"
    HELD_OBJECT_SNAPSHOT_STALE = "HELD_OBJECT_SNAPSHOT_STALE"
    HELD_OBJECT_SESSION_MISMATCH = "HELD_OBJECT_SESSION_MISMATCH"
    HELD_OBJECT_SEQUENCE_INVALID = "HELD_OBJECT_SEQUENCE_INVALID"
    HELD_OBJECT_CLOCK_INVALID = "HELD_OBJECT_CLOCK_INVALID"
    HELD_OBJECT_EXPECTED_EMPTY = "HELD_OBJECT_EXPECTED_EMPTY"
    HELD_OBJECT_EXPECTED_PRESENT = "HELD_OBJECT_EXPECTED_PRESENT"
    HELD_OBJECT_INSTANCE_MISMATCH = "HELD_OBJECT_INSTANCE_MISMATCH"
    HELD_OBJECT_CHECKPOINT_FAILED = "HELD_OBJECT_CHECKPOINT_FAILED"
    HELD_OBJECT_CHECKPOINTS_VALID = "HELD_OBJECT_CHECKPOINTS_VALID"


class HeldObjectClassificationState:
    """Normalized states; unknown and empty are deliberately distinct."""

    EMPTY = "EMPTY"
    HOLDING_EXPECTED_OBJECT = "HOLDING_EXPECTED_OBJECT"
    HOLDING_UNEXPECTED_OBJECT = "HOLDING_UNEXPECTED_OBJECT"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    STALE = "STALE"
    SESSION_MISMATCH = "SESSION_MISMATCH"


@runtime_checkable
class HeldObjectBackendReader(Protocol):
    """Minimal injected reader protocol; no backend object is exposed."""

    def read_held_object_name(self) -> object:
        ...


@runtime_checkable
class HeldObjectClock(Protocol):
    """Injectable clock used by deterministic freshness tests."""

    def monotonic(self) -> float:
        ...

    def now(self) -> datetime:
        ...


@dataclass(frozen=True)
class HeldObjectReadResult:
    """Optional metadata envelope returned by an isolated reader."""

    status: str
    held_object: object
    session_id: str | None = None
    sequence: int | None = None
    captured_at: str | None = None
    monotonic_time: float | None = None
    source: str = "injected_reader"
    source_kind: str = "backend_adapter"
    error_type: str | None = None
    error_message: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeldObjectSnapshot:
    """Immutable, backend-reference-free result of one reader invocation."""

    provider_version: str
    status: str
    held_object: str | None
    captured_at: str
    monotonic_time: float | None
    session_id: str | None
    sequence: int | None
    source: str
    source_kind: str
    read_succeeded: bool
    error_type: str | None
    error_message_sanitized: str | None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


@dataclass(frozen=True)
class HeldObjectClassification:
    valid: bool
    state: str
    expected_instance: str | None
    held_object: str | None
    reason_code: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeldObjectRuntimeGate:
    allowed: bool
    reason_code: str
    expected_state: str
    expected_instance: str | None
    actual_state: str
    actual_held_object: str | None
    session_id: str | None
    sequence: int | None
    failure_action: str
    remaining_dispatch_calls: tuple[dict[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["remaining_dispatch_calls"] = [
            copy.deepcopy(item) for item in self.remaining_dispatch_calls
        ]
        return data


@dataclass(frozen=True)
class HeldObjectCheckpoint:
    checkpoint_id: str
    dispatch_index: int
    phase: str
    expected_state: str
    expected_instance: str | None
    required: bool
    max_age_seconds: float | None
    failure_action: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reason_codes"] = list(self.reason_codes)
        return data


@dataclass(frozen=True)
class HeldObjectCheckpointContract:
    valid: bool
    allowed: bool
    reason_code: str
    selected_instance: str | None
    checkpoints: tuple[HeldObjectCheckpoint, ...] = ()
    remaining_dispatch_calls: tuple[dict[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "selected_instance": self.selected_instance,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "remaining_dispatch_calls": [
                copy.deepcopy(item) for item in self.remaining_dispatch_calls
            ],
            "reasons": list(self.reasons),
        }


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class OfflineMockHeldObjectReader:
    """Pure in-memory reader used only by offline tests."""

    def __init__(self, results: Sequence[object]) -> None:
        self._results = tuple(copy.deepcopy(list(results)))
        self._index = 0
        self.read_count = 0

    def read_held_object_name(self) -> object:
        self.read_count += 1
        if self._index >= len(self._results):
            raise RuntimeError("mock reader has no remaining result")
        result = copy.deepcopy(self._results[self._index])
        self._index += 1
        if isinstance(result, BaseException):
            raise result
        return result


_WINDOWS_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/])(?:[^ \t\r\n:'\"]+[\\/]?)+"
)
_UNIX_HOME_PATH = re.compile(r"/(?:home|users)/[^ \t\r\n:'\"]+")
_SECRET_TOKEN = re.compile(
    r"(?i)(?:"
    r"authorization\s*[:=]\s*bearer\s+\S+"
    r"|bearer\s+\S+"
    r"|api[_-]?key\s*[:=]\s*\S+"
    r")"
)


def _sanitize_error_message(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = _SECRET_TOKEN.sub("<redacted-secret>", text)
    text = _WINDOWS_PATH.sub("<redacted-path>", text)
    text = _UNIX_HOME_PATH.sub("<redacted-path>", text)
    return " ".join(text.split())[:256]


def _safe_clock(clock: HeldObjectClock | None) -> HeldObjectClock:
    return clock if clock is not None else _SystemClock()


def _captured_at(clock: HeldObjectClock) -> str:
    value = clock.now()
    if not isinstance(value, datetime):
        raise TypeError("clock.now() must return datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _error_snapshot(
    *,
    clock: HeldObjectClock,
    error_type: str,
    message: object,
    source: str,
    source_kind: str,
    session_id: str | None = None,
    sequence: int | None = None,
) -> HeldObjectSnapshot:
    try:
        captured_at = _captured_at(clock)
        monotonic_time = float(clock.monotonic())
    except Exception:
        captured_at = datetime.now(timezone.utc).isoformat()
        monotonic_time = None
    return HeldObjectSnapshot(
        provider_version=PROVIDER_VERSION,
        status="error",
        held_object=None,
        captured_at=captured_at,
        monotonic_time=monotonic_time,
        session_id=session_id,
        sequence=sequence,
        source=source,
        source_kind=source_kind,
        read_succeeded=False,
        error_type=error_type,
        error_message_sanitized=_sanitize_error_message(message),
        evidence=(),
    )


def _normalize_result(
    *,
    raw: object,
    clock: HeldObjectClock,
    reader: HeldObjectBackendReader,
) -> HeldObjectSnapshot:
    if isinstance(raw, HeldObjectReadResult):
        status = raw.status
        held_object = raw.held_object
        session_id = raw.session_id
        sequence = raw.sequence
        source = raw.source
        source_kind = raw.source_kind
        captured_at = raw.captured_at
        monotonic_time = raw.monotonic_time
        error_type = raw.error_type
        error_message = raw.error_message
        evidence = tuple(_sanitize_error_message(item) for item in raw.evidence)
    else:
        status = "known"
        held_object = raw
        session_id = None
        sequence = None
        source = (
            "offline_mock"
            if isinstance(reader, OfflineMockHeldObjectReader)
            else "injected_reader"
        )
        source_kind = (
            "mock"
            if isinstance(reader, OfflineMockHeldObjectReader)
            else "backend_adapter"
        )
        captured_at = None
        monotonic_time = None
        error_type = None
        error_message = None
        evidence = ()

    if status not in VALID_READ_STATUSES:
        return _error_snapshot(
            clock=clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            message=f"invalid reader status: {status!r}",
            source=source,
            source_kind=source_kind,
            session_id=session_id,
            sequence=sequence,
        )
    if source_kind not in VALID_SOURCE_KINDS:
        return _error_snapshot(
            clock=clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            message=f"invalid source_kind: {source_kind!r}",
            source=source,
            source_kind="backend_adapter",
            session_id=session_id,
            sequence=sequence,
        )
    if (
        sequence is not None
        and (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        )
    ):
        return _error_snapshot(
            clock=clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_SEQUENCE_INVALID,
            message="sequence must be a non-negative integer or null",
            source=source,
            source_kind=source_kind,
            session_id=session_id,
        )
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id.strip()
    ):
        return _error_snapshot(
            clock=clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            message="session_id must be a non-empty string or null",
            source=source,
            source_kind=source_kind,
            sequence=sequence,
        )

    if status == "known":
        if isinstance(held_object, str):
            if not held_object.strip() or held_object.strip().lower() in {
                "none",
                "null",
            }:
                return _error_snapshot(
                    clock=clock,
                    error_type=(
                        HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID
                    ),
                    message="known held-object value is an invalid string",
                    source=source,
                    source_kind=source_kind,
                    session_id=session_id,
                    sequence=sequence,
                )
            normalized_object: str | None = held_object
        elif held_object is None:
            normalized_object = None
        else:
            return _error_snapshot(
                clock=clock,
                error_type=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
                message="held-object value must be a string or null",
                source=source,
                source_kind=source_kind,
                session_id=session_id,
                sequence=sequence,
            )
    else:
        if held_object is not None:
            return _error_snapshot(
                clock=clock,
                error_type=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
                message=f"{status} reader result must not carry an object",
                source=source,
                source_kind=source_kind,
                session_id=session_id,
                sequence=sequence,
            )
        normalized_object = None

    try:
        captured = captured_at or _captured_at(clock)
        monotonic = (
            float(monotonic_time)
            if monotonic_time is not None
            else float(clock.monotonic())
        )
        if not isinstance(captured, str) or not captured.strip():
            raise ValueError("captured_at must be a non-empty ISO timestamp")
        datetime.fromisoformat(captured.replace("Z", "+00:00"))
        if not math.isfinite(monotonic):
            raise ValueError("monotonic_time must be finite")
    except Exception as exc:
        return _error_snapshot(
            clock=clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_CLOCK_INVALID,
            message=exc,
            source=source,
            source_kind=source_kind,
            session_id=session_id,
            sequence=sequence,
        )

    return HeldObjectSnapshot(
        provider_version=PROVIDER_VERSION,
        status=status,
        held_object=normalized_object,
        captured_at=captured,
        monotonic_time=monotonic,
        session_id=session_id,
        sequence=sequence,
        source=_sanitize_error_message(source),
        source_kind=source_kind,
        read_succeeded=status != "error",
        error_type=error_type if status == "error" else None,
        error_message_sanitized=(
            _sanitize_error_message(error_message)
            if status == "error" and error_message is not None
            else None
        ),
        evidence=evidence,
    )


def _apply_session_and_freshness(
    *,
    snapshot: HeldObjectSnapshot,
    expected_session_id: str | None,
    max_age_seconds: float | None,
    clock: HeldObjectClock,
) -> HeldObjectSnapshot:
    if snapshot.status == "error":
        return snapshot
    if expected_session_id is not None and (
        snapshot.session_id != expected_session_id
    ):
        return replace(snapshot, status="session_mismatch")
    if max_age_seconds is None:
        return snapshot
    if (
        isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, (int, float))
        or max_age_seconds < 0
        or not math.isfinite(float(max_age_seconds))
    ):
        return replace(
            snapshot,
            status="error",
            read_succeeded=False,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_CLOCK_INVALID,
            error_message_sanitized="max_age_seconds must be non-negative",
        )
    if snapshot.monotonic_time is None:
        return replace(snapshot, status="stale")
    try:
        now_monotonic = float(clock.monotonic())
        if not math.isfinite(now_monotonic):
            raise ValueError("clock.monotonic() must be finite")
        age = now_monotonic - snapshot.monotonic_time
    except Exception as exc:
        return replace(
            snapshot,
            status="error",
            read_succeeded=False,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_CLOCK_INVALID,
            error_message_sanitized=_sanitize_error_message(exc),
        )
    if age < 0:
        return replace(
            snapshot,
            status="error",
            read_succeeded=False,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_CLOCK_INVALID,
            error_message_sanitized="monotonic clock moved backwards",
        )
    if age > float(max_age_seconds):
        return replace(snapshot, status="stale")
    return snapshot


def read_l1_held_object_snapshot(
    *,
    reader: HeldObjectBackendReader | None,
    expected_session_id: str | None = None,
    max_age_seconds: float | None = None,
    clock: HeldObjectClock | None = None,
) -> HeldObjectSnapshot:
    """Read one independent snapshot without caching the reader result."""

    selected_clock = _safe_clock(clock)
    if reader is None:
        return _error_snapshot(
            clock=selected_clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_READER_MISSING,
            message="held-object reader was not provided",
            source="not_connected",
            source_kind="backend_adapter",
        )
    if not isinstance(reader, HeldObjectBackendReader):
        return _error_snapshot(
            clock=selected_clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_READER_MISSING,
            message="reader does not implement read_held_object_name",
            source="invalid_reader",
            source_kind="backend_adapter",
        )
    try:
        raw = reader.read_held_object_name()
    except Exception as exc:
        return _error_snapshot(
            clock=selected_clock,
            error_type=HeldObjectProviderReasonCode.HELD_OBJECT_READ_FAILED,
            message=exc,
            source=(
                "offline_mock"
                if isinstance(reader, OfflineMockHeldObjectReader)
                else "injected_reader"
            ),
            source_kind=(
                "mock"
                if isinstance(reader, OfflineMockHeldObjectReader)
                else "backend_adapter"
            ),
        )
    snapshot = _normalize_result(
        raw=raw,
        clock=selected_clock,
        reader=reader,
    )
    return _apply_session_and_freshness(
        snapshot=snapshot,
        expected_session_id=expected_session_id,
        max_age_seconds=max_age_seconds,
        clock=selected_clock,
    )


def classify_l1_held_object_state(
    *,
    snapshot: HeldObjectSnapshot,
    expected_instance: str,
) -> HeldObjectClassification:
    """Classify a snapshot without treating any unknown state as empty."""

    if not isinstance(snapshot, HeldObjectSnapshot):
        return HeldObjectClassification(
            valid=False,
            state=HeldObjectClassificationState.ERROR,
            expected_instance=None,
            held_object=None,
            reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            reasons=("snapshot must be HeldObjectSnapshot",),
        )
    if not isinstance(expected_instance, str) or not expected_instance.strip():
        return HeldObjectClassification(
            valid=False,
            state=HeldObjectClassificationState.ERROR,
            expected_instance=None,
            held_object=snapshot.held_object,
            reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            reasons=("expected_instance must be a non-empty string",),
        )
    status_states = {
        "unknown": (
            HeldObjectClassificationState.UNKNOWN,
            HeldObjectProviderReasonCode.HELD_OBJECT_STATE_UNKNOWN,
        ),
        "error": (
            HeldObjectClassificationState.ERROR,
            snapshot.error_type
            or HeldObjectProviderReasonCode.HELD_OBJECT_STATE_ERROR,
        ),
        "stale": (
            HeldObjectClassificationState.STALE,
            HeldObjectProviderReasonCode.HELD_OBJECT_SNAPSHOT_STALE,
        ),
        "session_mismatch": (
            HeldObjectClassificationState.SESSION_MISMATCH,
            HeldObjectProviderReasonCode.HELD_OBJECT_SESSION_MISMATCH,
        ),
    }
    if snapshot.status in status_states:
        state, reason = status_states[snapshot.status]
        if snapshot.status == "error" and reason not in {
            HeldObjectProviderReasonCode.HELD_OBJECT_READER_MISSING,
            HeldObjectProviderReasonCode.HELD_OBJECT_READ_FAILED,
            HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            HeldObjectProviderReasonCode.HELD_OBJECT_SEQUENCE_INVALID,
            HeldObjectProviderReasonCode.HELD_OBJECT_CLOCK_INVALID,
            HeldObjectProviderReasonCode.HELD_OBJECT_STATE_ERROR,
        }:
            reason = HeldObjectProviderReasonCode.HELD_OBJECT_STATE_ERROR
        return HeldObjectClassification(
            valid=False,
            state=state,
            expected_instance=expected_instance,
            held_object=snapshot.held_object,
            reason_code=reason,
            reasons=(f"snapshot status is {snapshot.status}",),
        )
    if snapshot.status != "known":
        return HeldObjectClassification(
            valid=False,
            state=HeldObjectClassificationState.ERROR,
            expected_instance=expected_instance,
            held_object=snapshot.held_object,
            reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_VALUE_INVALID,
            reasons=(f"unsupported snapshot status: {snapshot.status}",),
        )
    if snapshot.held_object is None:
        return HeldObjectClassification(
            valid=True,
            state=HeldObjectClassificationState.EMPTY,
            expected_instance=expected_instance,
            held_object=None,
            reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_CHECKPOINTS_VALID,
            reasons=("known null is the only EMPTY representation",),
        )
    if snapshot.held_object == expected_instance:
        return HeldObjectClassification(
            valid=True,
            state=HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT,
            expected_instance=expected_instance,
            held_object=snapshot.held_object,
            reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_CHECKPOINTS_VALID,
            reasons=("held object matches expected_instance",),
        )
    return HeldObjectClassification(
        valid=True,
        state=HeldObjectClassificationState.HOLDING_UNEXPECTED_OBJECT,
        expected_instance=expected_instance,
        held_object=snapshot.held_object,
        reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_INSTANCE_MISMATCH,
        reasons=("held object differs from expected_instance",),
    )


def _sequence_failure(
    *,
    snapshot: HeldObjectSnapshot,
    expected_state: str,
    expected_instance: str,
    reason_code: str,
    reason: str,
) -> HeldObjectRuntimeGate:
    return HeldObjectRuntimeGate(
        allowed=False,
        reason_code=reason_code,
        expected_state=expected_state,
        expected_instance=expected_instance,
        actual_state=HeldObjectClassificationState.ERROR,
        actual_held_object=snapshot.held_object,
        session_id=snapshot.session_id,
        sequence=snapshot.sequence,
        failure_action="stop",
        remaining_dispatch_calls=(),
        reasons=(reason,),
    )


def validate_l1_held_object_runtime_gate(
    *,
    snapshot: HeldObjectSnapshot,
    expected_instance: str,
    expected_state: str,
    expected_session_id: str | None = None,
    max_age_seconds: float | None = None,
    previous_snapshot: HeldObjectSnapshot | None = None,
    clock: HeldObjectClock | None = None,
) -> HeldObjectRuntimeGate:
    """Evaluate one checkpoint and clear all remaining calls on failure."""

    selected_clock = _safe_clock(clock)
    checked = _apply_session_and_freshness(
        snapshot=snapshot,
        expected_session_id=expected_session_id,
        max_age_seconds=max_age_seconds,
        clock=selected_clock,
    )
    if previous_snapshot is not None:
        if checked.session_id != previous_snapshot.session_id:
            return _sequence_failure(
                snapshot=checked,
                expected_state=expected_state,
                expected_instance=expected_instance,
                reason_code=(
                    HeldObjectProviderReasonCode.HELD_OBJECT_SESSION_MISMATCH
                ),
                reason="current and previous snapshots use different sessions",
            )
        if checked.sequence is None or previous_snapshot.sequence is None:
            return _sequence_failure(
                snapshot=checked,
                expected_state=expected_state,
                expected_instance=expected_instance,
                reason_code=(
                    HeldObjectProviderReasonCode.HELD_OBJECT_SEQUENCE_INVALID
                ),
                reason="sequence is required when previous_snapshot is supplied",
            )
        if checked.sequence <= previous_snapshot.sequence:
            return _sequence_failure(
                snapshot=checked,
                expected_state=expected_state,
                expected_instance=expected_instance,
                reason_code=(
                    HeldObjectProviderReasonCode.HELD_OBJECT_SEQUENCE_INVALID
                ),
                reason="sequence must increase strictly",
            )

    classification = classify_l1_held_object_state(
        snapshot=checked,
        expected_instance=expected_instance,
    )
    if not classification.valid:
        return HeldObjectRuntimeGate(
            allowed=False,
            reason_code=classification.reason_code,
            expected_state=expected_state,
            expected_instance=expected_instance,
            actual_state=classification.state,
            actual_held_object=classification.held_object,
            session_id=checked.session_id,
            sequence=checked.sequence,
            failure_action="stop",
            remaining_dispatch_calls=(),
            reasons=classification.reasons,
        )
    if expected_state not in {
        HeldObjectClassificationState.EMPTY,
        HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT,
    }:
        return HeldObjectRuntimeGate(
            allowed=False,
            reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_CHECKPOINT_FAILED,
            expected_state=expected_state,
            expected_instance=expected_instance,
            actual_state=classification.state,
            actual_held_object=classification.held_object,
            session_id=checked.session_id,
            sequence=checked.sequence,
            failure_action="stop",
            remaining_dispatch_calls=(),
            reasons=("unsupported expected_state",),
        )
    if expected_state == HeldObjectClassificationState.EMPTY:
        if classification.state != HeldObjectClassificationState.EMPTY:
            reason = (
                HeldObjectProviderReasonCode.HELD_OBJECT_INSTANCE_MISMATCH
                if classification.state
                == HeldObjectClassificationState.HOLDING_UNEXPECTED_OBJECT
                else HeldObjectProviderReasonCode.HELD_OBJECT_EXPECTED_EMPTY
            )
            return HeldObjectRuntimeGate(
                allowed=False,
                reason_code=reason,
                expected_state=expected_state,
                expected_instance=expected_instance,
                actual_state=classification.state,
                actual_held_object=classification.held_object,
                session_id=checked.session_id,
                sequence=checked.sequence,
                failure_action="stop",
                remaining_dispatch_calls=(),
                reasons=("checkpoint requires an empty gripper",),
            )
    elif (
        classification.state
        != HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT
    ):
        reason = (
            HeldObjectProviderReasonCode.HELD_OBJECT_EXPECTED_PRESENT
            if classification.state == HeldObjectClassificationState.EMPTY
            else HeldObjectProviderReasonCode.HELD_OBJECT_INSTANCE_MISMATCH
        )
        return HeldObjectRuntimeGate(
            allowed=False,
            reason_code=reason,
            expected_state=expected_state,
            expected_instance=expected_instance,
            actual_state=classification.state,
            actual_held_object=classification.held_object,
            session_id=checked.session_id,
            sequence=checked.sequence,
            failure_action="stop",
            remaining_dispatch_calls=(),
            reasons=("checkpoint requires the frozen held-object instance",),
        )
    return HeldObjectRuntimeGate(
        allowed=True,
        reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_CHECKPOINTS_VALID,
        expected_state=expected_state,
        expected_instance=expected_instance,
        actual_state=classification.state,
        actual_held_object=classification.held_object,
        session_id=checked.session_id,
        sequence=checked.sequence,
        failure_action="none",
        remaining_dispatch_calls=(),
        reasons=("runtime held-object checkpoint passed",),
    )


def _call_mapping(call: object) -> Mapping[str, Any] | None:
    if isinstance(call, Mapping):
        return call
    to_dict = getattr(call, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, Mapping) else None
    return None


def _blocked_checkpoint_contract(
    reason: str,
    *,
    selected_instance: str | None,
) -> HeldObjectCheckpointContract:
    return HeldObjectCheckpointContract(
        valid=False,
        allowed=False,
        reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_CHECKPOINT_FAILED,
        selected_instance=selected_instance,
        checkpoints=(),
        remaining_dispatch_calls=(),
        reasons=(reason,),
    )


def build_l1_held_object_checkpoints(
    *,
    dispatch_calls: Sequence[object],
    selected_instance: str,
) -> HeldObjectCheckpointContract:
    """Build the eight required pre/post gates for the reviewed four calls."""

    if not isinstance(selected_instance, str) or not selected_instance.strip():
        return _blocked_checkpoint_contract(
            "selected_instance must be a non-empty string",
            selected_instance=None,
        )
    if len(dispatch_calls) != 4:
        return _blocked_checkpoint_contract(
            "exactly four dispatch calls are required",
            selected_instance=selected_instance,
        )
    mappings = tuple(_call_mapping(item) for item in dispatch_calls)
    if any(item is None for item in mappings):
        return _blocked_checkpoint_contract(
            "every dispatch call must be serializable to a mapping",
            selected_instance=selected_instance,
        )
    skills = tuple(item.get("skill") for item in mappings if item is not None)
    indexes = tuple(
        item.get("dispatch_index") for item in mappings if item is not None
    )
    if skills != EXPECTED_L1_SKILLS or indexes != (1, 2, 3, 4):
        return _blocked_checkpoint_contract(
            f"expected skills/indexes {EXPECTED_L1_SKILLS}/(1,2,3,4)",
            selected_instance=selected_instance,
        )

    definitions = (
        (1, "pre", HeldObjectClassificationState.EMPTY, None),
        (1, "post", HeldObjectClassificationState.EMPTY, None),
        (2, "pre", HeldObjectClassificationState.EMPTY, None),
        (
            2,
            "post",
            HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT,
            selected_instance,
        ),
        (
            3,
            "pre",
            HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT,
            selected_instance,
        ),
        (
            3,
            "post",
            HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT,
            selected_instance,
        ),
        (
            4,
            "pre",
            HeldObjectClassificationState.HOLDING_EXPECTED_OBJECT,
            selected_instance,
        ),
        (4, "post", HeldObjectClassificationState.EMPTY, None),
    )
    checkpoints = tuple(
        HeldObjectCheckpoint(
            checkpoint_id=f"L1_D{dispatch_index}_{phase.upper()}",
            dispatch_index=dispatch_index,
            phase=phase,
            expected_state=state,
            expected_instance=instance,
            required=True,
            max_age_seconds=None,
            failure_action="stop",
            reason_codes=(
                HeldObjectProviderReasonCode.HELD_OBJECT_STATE_UNKNOWN,
                HeldObjectProviderReasonCode.HELD_OBJECT_STATE_ERROR,
                HeldObjectProviderReasonCode.HELD_OBJECT_SNAPSHOT_STALE,
                HeldObjectProviderReasonCode.HELD_OBJECT_SESSION_MISMATCH,
                HeldObjectProviderReasonCode.HELD_OBJECT_SEQUENCE_INVALID,
                HeldObjectProviderReasonCode.HELD_OBJECT_EXPECTED_EMPTY,
                HeldObjectProviderReasonCode.HELD_OBJECT_EXPECTED_PRESENT,
                HeldObjectProviderReasonCode.HELD_OBJECT_INSTANCE_MISMATCH,
            ),
        )
        for dispatch_index, phase, state, instance in definitions
    )
    return HeldObjectCheckpointContract(
        valid=True,
        allowed=True,
        reason_code=HeldObjectProviderReasonCode.HELD_OBJECT_CHECKPOINTS_VALID,
        selected_instance=selected_instance,
        checkpoints=checkpoints,
        remaining_dispatch_calls=(),
        reasons=("eight fail-closed held-object checkpoints were compiled",),
    )


__all__ = [
    "EXPECTED_L1_SKILLS",
    "HeldObjectBackendReader",
    "HeldObjectCheckpoint",
    "HeldObjectCheckpointContract",
    "HeldObjectClassification",
    "HeldObjectClassificationState",
    "HeldObjectClock",
    "HeldObjectProviderReasonCode",
    "HeldObjectReadResult",
    "HeldObjectRuntimeGate",
    "HeldObjectSnapshot",
    "OfflineMockHeldObjectReader",
    "PROVIDER_VERSION",
    "build_l1_held_object_checkpoints",
    "classify_l1_held_object_state",
    "read_l1_held_object_snapshot",
    "validate_l1_held_object_runtime_gate",
]
