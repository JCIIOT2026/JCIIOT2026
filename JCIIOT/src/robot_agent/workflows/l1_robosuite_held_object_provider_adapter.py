"""Explicit Reader-v2 to L1 held-object Provider protocol adapter.

The adapter owns protocol conversion only.  Its Reader and clock are injected,
and one ``read_snapshot`` call performs exactly one Provider call and at most
one Reader call.  Reader-local sequence metadata is retained as trace metadata
instead of being relabelled as a backend state sequence.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .l1_held_object_provider import (
    HeldObjectClock,
    HeldObjectReadResult,
    HeldObjectSnapshot,
    read_l1_held_object_snapshot,
)
from .l1_robosuite_held_object_reader import (
    RobosuiteHeldObjectReadResult,
    sanitize_reader_error_message,
)


ADAPTER_VERSION = "1.0"
VALID_READER_STATUSES = frozenset(
    {"known", "unknown", "unavailable", "error"}
)
_ERROR_TYPE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}\Z")
_HELD_OBJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SENSITIVE_LABEL = re.compile(
    r"(?i)\b(?:authorization|bearer|token|api[_-]?key|secret|traceback)\b"
)


class RobosuiteHeldObjectProviderAdapterReasonCode:
    """Stable failures emitted by the protocol-conversion boundary."""

    ADAPTER_READER_INVALID = "ADAPTER_READER_INVALID"
    ADAPTER_READER_RAISED = "ADAPTER_READER_RAISED"
    ADAPTER_RESULT_INVALID = "ADAPTER_RESULT_INVALID"


class RobosuiteHeldObjectReader(Protocol):
    """Injected Reader-v2 surface used by this adapter."""

    def read_held_object_name(self) -> object:
        ...


@dataclass(frozen=True)
class RobosuiteHeldObjectReadTrace:
    """Sanitized Reader facts retained outside the Provider snapshot."""

    reader_called: bool
    read_status: str
    read_sequence: int | None
    backend_state_sequence: int | None
    backend_state_sequence_status: str
    error_type: str | None
    exception_type: str | None
    message: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobosuiteHeldObjectProviderRead:
    """One Provider snapshot plus safe Reader-local trace metadata."""

    adapter_version: str
    snapshot: HeldObjectSnapshot
    reader_called: bool
    read_status: str
    held_object_semantic: str
    read_sequence: int | None
    backend_state_sequence: int | None
    backend_state_sequence_status: str
    error_type: str | None
    exception_type: str | None
    message: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["snapshot"] = self.snapshot.to_dict()
        return data


def _invalid_trace(
    error_type: str,
    message: str,
    *,
    reader_called: bool,
) -> RobosuiteHeldObjectReadTrace:
    return RobosuiteHeldObjectReadTrace(
        reader_called=reader_called,
        read_status="error",
        read_sequence=None,
        backend_state_sequence=None,
        backend_state_sequence_status="unknown",
        error_type=error_type,
        exception_type=None,
        message=message,
    )


def _safe_exception_type(value: object) -> str | None:
    if type(value) is not str or not _EXCEPTION_TYPE.fullmatch(value):
        return None
    return value


def _safe_error_type(value: object) -> str | None:
    if type(value) is not str or not _ERROR_TYPE.fullmatch(value):
        return None
    return value


def _safe_message(value: object | None) -> str | None:
    if value is None:
        return None
    sanitized = sanitize_reader_error_message(value, max_length=256)
    return _SENSITIVE_LABEL.sub("<redacted>", sanitized)[:256]


def _validate_reader_result(
    raw: object,
) -> tuple[HeldObjectReadResult, RobosuiteHeldObjectReadTrace]:
    invalid_type = (
        RobosuiteHeldObjectProviderAdapterReasonCode.ADAPTER_RESULT_INVALID
    )
    if type(raw) is not RobosuiteHeldObjectReadResult:
        trace = _invalid_trace(
            invalid_type,
            "Reader returned an invalid result envelope",
            reader_called=True,
        )
        return _provider_error_result(trace), trace

    if type(raw.status) is not str or raw.status not in VALID_READER_STATUSES:
        trace = _invalid_trace(
            invalid_type,
            "Reader result status is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if (
        type(raw.read_sequence) is not int
        or raw.read_sequence < 1
    ):
        trace = _invalid_trace(
            invalid_type,
            "Reader result sequence is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if (
        raw.backend_state_sequence is not None
        or type(raw.backend_state_sequence_status) is not str
        or raw.backend_state_sequence_status != "unknown"
    ):
        trace = _invalid_trace(
            invalid_type,
            "Reader backend state sequence metadata is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if type(raw.session_id) is not str or not raw.session_id.strip():
        trace = _invalid_trace(
            invalid_type,
            "Reader result session is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if type(raw.session_source) is not str or raw.session_source != "external":
        trace = _invalid_trace(
            invalid_type,
            "Reader result session source is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if (
        type(raw.source) is not str
        or not raw.source.strip()
        or type(raw.source_kind) is not str
        or raw.source_kind != "backend_adapter"
    ):
        trace = _invalid_trace(
            invalid_type,
            "Reader result source metadata is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if raw.monotonic_time is not None and (
        type(raw.monotonic_time) not in {int, float}
        or not math.isfinite(float(raw.monotonic_time))
    ):
        trace = _invalid_trace(
            invalid_type,
            "Reader result clock metadata is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if raw.captured_at is not None and (
        type(raw.captured_at) is not str or not raw.captured_at.strip()
    ):
        trace = _invalid_trace(
            invalid_type,
            "Reader result timestamp is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if type(raw.evidence) is not tuple or any(
        type(item) is not str for item in raw.evidence
    ):
        trace = _invalid_trace(
            invalid_type,
            "Reader result evidence is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace

    if raw.status == "known":
        if raw.held_object is not None and (
            type(raw.held_object) is not str
            or _HELD_OBJECT_NAME.fullmatch(raw.held_object) is None
        ):
            trace = _invalid_trace(
                invalid_type,
                "Reader held-object value is invalid",
                reader_called=True,
            )
            return _provider_error_result(trace), trace
    elif raw.held_object is not None:
        trace = _invalid_trace(
            invalid_type,
            "Non-known Reader result carried a held object",
            reader_called=True,
        )
        return _provider_error_result(trace), trace

    error_type = _safe_error_type(raw.error_type)
    exception_type = _safe_exception_type(raw.exception_type)
    if raw.error_type is not None and error_type is None:
        trace = _invalid_trace(
            invalid_type,
            "Reader result error type is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if raw.exception_type is not None and exception_type is None:
        trace = _invalid_trace(
            invalid_type,
            "Reader result exception type is invalid",
            reader_called=True,
        )
        return _provider_error_result(trace), trace
    if raw.status in {"unavailable", "error"} and error_type is None:
        trace = _invalid_trace(
            invalid_type,
            "Failed Reader result omitted its error type",
            reader_called=True,
        )
        return _provider_error_result(trace), trace

    message = _safe_message(raw.error_message_sanitized)
    provider_status = (
        "error" if raw.status in {"unavailable", "error"} else raw.status
    )
    provider_result = HeldObjectReadResult(
        status=provider_status,
        held_object=raw.held_object,
        session_id=raw.session_id,
        sequence=None,
        captured_at=raw.captured_at,
        monotonic_time=raw.monotonic_time,
        source=_safe_message(raw.source) or "reader",
        source_kind="backend_adapter",
        error_type=error_type,
        error_message=message,
        evidence=tuple((_safe_message(item) or "")[:128] for item in raw.evidence),
    )
    trace = RobosuiteHeldObjectReadTrace(
        reader_called=True,
        read_status=raw.status,
        read_sequence=raw.read_sequence,
        backend_state_sequence=None,
        backend_state_sequence_status="unknown",
        error_type=error_type,
        exception_type=exception_type,
        message=message,
    )
    return provider_result, trace


def _provider_error_result(
    trace: RobosuiteHeldObjectReadTrace,
) -> HeldObjectReadResult:
    return HeldObjectReadResult(
        status="error",
        held_object=None,
        source="robosuite_reader_provider_adapter",
        source_kind="backend_adapter",
        error_type=trace.error_type,
        error_message=trace.message,
    )


class _ProviderReaderBridge:
    """One-use bridge understood by ``read_l1_held_object_snapshot``."""

    def __init__(self, reader: object) -> None:
        self._reader = reader
        self.trace: RobosuiteHeldObjectReadTrace | None = None

    def read_held_object_name(self) -> HeldObjectReadResult:
        try:
            method = getattr(self._reader, "read_held_object_name", None)
        except BaseException:
            method = None
        if not callable(method):
            self.trace = _invalid_trace(
                RobosuiteHeldObjectProviderAdapterReasonCode
                .ADAPTER_READER_INVALID,
                "Injected Reader does not implement the read protocol",
                reader_called=False,
            )
            return _provider_error_result(self.trace)
        try:
            raw = method()
        except BaseException as exc:
            exception_type = _safe_exception_type(type(exc).__name__)
            self.trace = RobosuiteHeldObjectReadTrace(
                reader_called=True,
                read_status="error",
                read_sequence=None,
                backend_state_sequence=None,
                backend_state_sequence_status="unknown",
                error_type=(
                    RobosuiteHeldObjectProviderAdapterReasonCode
                    .ADAPTER_READER_RAISED
                ),
                exception_type=exception_type,
                message=_safe_message(exc),
            )
            return _provider_error_result(self.trace)
        provider_result, self.trace = _validate_reader_result(raw)
        return provider_result


def _semantic(status: str, held_object: str | None) -> str:
    if status == "known":
        return "not_held" if held_object is None else "held"
    if status in {"unknown", "unavailable"}:
        return status
    return "error"


class RobosuiteHeldObjectProviderAdapter:
    """Injected read-only Provider that adapts exactly one Reader-v2 call."""

    def __init__(
        self,
        *,
        reader: RobosuiteHeldObjectReader,
        clock: HeldObjectClock | None = None,
    ) -> None:
        self._reader = reader
        self._clock = clock

    def read_snapshot(self) -> RobosuiteHeldObjectProviderRead:
        bridge = _ProviderReaderBridge(self._reader)
        snapshot = read_l1_held_object_snapshot(
            reader=bridge,
            clock=self._clock,
        )
        trace = bridge.trace
        if trace is None:
            trace = _invalid_trace(
                RobosuiteHeldObjectProviderAdapterReasonCode
                .ADAPTER_RESULT_INVALID,
                "Provider completed without Reader trace metadata",
                reader_called=False,
            )

        read_status = trace.read_status
        error_type = trace.error_type
        message = trace.message
        if snapshot.status == "error" and read_status not in {
            "unavailable",
            "error",
        }:
            read_status = "error"
            error_type = snapshot.error_type
            message = snapshot.error_message_sanitized
        elif read_status in {"unavailable", "error"}:
            error_type = snapshot.error_type or error_type
            message = snapshot.error_message_sanitized or message

        return RobosuiteHeldObjectProviderRead(
            adapter_version=ADAPTER_VERSION,
            snapshot=snapshot,
            reader_called=trace.reader_called,
            read_status=read_status,
            held_object_semantic=_semantic(
                read_status,
                snapshot.held_object,
            ),
            read_sequence=trace.read_sequence,
            backend_state_sequence=trace.backend_state_sequence,
            backend_state_sequence_status=(
                trace.backend_state_sequence_status
            ),
            error_type=error_type,
            exception_type=trace.exception_type,
            message=_safe_message(message),
        )


def read_l1_robosuite_held_object_snapshot(
    *,
    reader: RobosuiteHeldObjectReader,
    clock: HeldObjectClock | None = None,
) -> RobosuiteHeldObjectProviderRead:
    """Adapt one injected Reader call through the existing Provider."""

    return RobosuiteHeldObjectProviderAdapter(
        reader=reader,
        clock=clock,
    ).read_snapshot()


__all__ = [
    "ADAPTER_VERSION",
    "RobosuiteHeldObjectProviderAdapter",
    "RobosuiteHeldObjectProviderAdapterReasonCode",
    "RobosuiteHeldObjectProviderRead",
    "RobosuiteHeldObjectReadTrace",
    "RobosuiteHeldObjectReader",
    "read_l1_robosuite_held_object_snapshot",
]
