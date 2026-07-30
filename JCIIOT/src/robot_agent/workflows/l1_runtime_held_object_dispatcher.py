"""Minimal read-only runtime dispatcher for L1 held-object queries.

The dispatcher validates one explicit query, invokes one injected read-only
Provider at most once, and returns an immutable structured response.  It has
no action compilation or execution capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol

from .l1_robosuite_held_object_provider_adapter import (
    ADAPTER_VERSION,
    RobosuiteHeldObjectProviderRead,
)
from .l1_held_object_provider import HeldObjectSnapshot


DISPATCHER_VERSION = "1.0"
HELD_OBJECT_QUERY_KIND = "held_object"
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_HELD_OBJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_ERROR_TYPE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}\Z")


class RuntimeHeldObjectDispatchStatus:
    SUCCESS = "success"
    READ_INDETERMINATE = "read_indeterminate"
    READ_FAILED = "read_failed"
    REJECTED = "rejected"


class RuntimeHeldObjectDispatchReasonCode:
    HELD_OBJECT_READ_SUCCEEDED = "HELD_OBJECT_READ_SUCCEEDED"
    HELD_OBJECT_STATE_UNKNOWN = "HELD_OBJECT_STATE_UNKNOWN"
    HELD_OBJECT_READ_UNAVAILABLE = "HELD_OBJECT_READ_UNAVAILABLE"
    HELD_OBJECT_READ_FAILED = "HELD_OBJECT_READ_FAILED"
    EXECUTE_NOT_ALLOWED = "EXECUTE_NOT_ALLOWED"
    REQUEST_INVALID = "REQUEST_INVALID"
    QUERY_KIND_NOT_ALLOWED = "QUERY_KIND_NOT_ALLOWED"
    PROVIDER_INVALID = "PROVIDER_INVALID"
    PROVIDER_RESULT_INVALID = "PROVIDER_RESULT_INVALID"
    PROVIDER_FAILED = "PROVIDER_FAILED"


@dataclass(frozen=True)
class L1RuntimeHeldObjectRequest:
    """Explicit request schema; execution intent has no default."""

    execute: bool
    query_kind: str
    request_id: str | None = None


@dataclass(frozen=True)
class L1RuntimeHeldObjectResponse:
    """Immutable, action-free response for one dispatcher invocation."""

    status: str
    reason_code: str
    execute: bool | None
    query_kind: str | None
    dispatcher_called: bool
    provider_called: bool
    reader_called: bool
    read_status: str
    held_object: str | None
    held_object_semantic: str
    read_sequence: int | None
    backend_state_sequence: int | None
    backend_state_sequence_status: str
    action_generated: bool
    action_dispatched: bool
    error_type: str | None
    exception_type: str | None
    message: str
    request_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeHeldObjectProvider(Protocol):
    """Injected Provider surface required by the dispatcher."""

    def read_snapshot(self) -> RobosuiteHeldObjectProviderRead:
        ...


@dataclass(frozen=True)
class _ValidatedRequest:
    execute: bool | None
    query_kind: str | None
    request_id: str | None
    error_type: str | None
    message: str | None


def _request_values(request: object) -> Mapping[str, object] | None:
    if type(request) is L1RuntimeHeldObjectRequest:
        return {
            "execute": request.execute,
            "query_kind": request.query_kind,
            "request_id": request.request_id,
        }
    if not isinstance(request, Mapping):
        return None
    try:
        values = dict(request)
    except BaseException:
        return None
    if any(type(key) is not str for key in values):
        return None
    if set(values) - {"execute", "query_kind", "request_id"}:
        return None
    return values


def _validate_request(request: object) -> _ValidatedRequest:
    values = _request_values(request)
    if values is None:
        return _ValidatedRequest(
            execute=None,
            query_kind=None,
            request_id=None,
            error_type=RuntimeHeldObjectDispatchReasonCode.REQUEST_INVALID,
            message="request must use the held-object query schema",
        )

    request_id = values.get("request_id")
    valid_request_id = (
        request_id is None
        or (
            type(request_id) is str
            and _REQUEST_ID.fullmatch(request_id) is not None
        )
    )
    if not valid_request_id:
        return _ValidatedRequest(
            execute=(
                values.get("execute")
                if type(values.get("execute")) is bool
                else None
            ),
            query_kind=(
                HELD_OBJECT_QUERY_KIND
                if values.get("query_kind") == HELD_OBJECT_QUERY_KIND
                else None
            ),
            request_id=None,
            error_type=RuntimeHeldObjectDispatchReasonCode.REQUEST_INVALID,
            message="request_id must be a stable identifier or null",
        )

    execute = values.get("execute")
    if "execute" not in values or type(execute) is not bool:
        return _ValidatedRequest(
            execute=None,
            query_kind=(
                HELD_OBJECT_QUERY_KIND
                if values.get("query_kind") == HELD_OBJECT_QUERY_KIND
                else None
            ),
            request_id=request_id,
            error_type=RuntimeHeldObjectDispatchReasonCode.REQUEST_INVALID,
            message="execute must be explicitly supplied as a boolean",
        )
    query_kind = values.get("query_kind")
    if (
        "query_kind" not in values
        or type(query_kind) is not str
        or query_kind != HELD_OBJECT_QUERY_KIND
    ):
        return _ValidatedRequest(
            execute=execute,
            query_kind=None,
            request_id=request_id,
            error_type=(
                RuntimeHeldObjectDispatchReasonCode.QUERY_KIND_NOT_ALLOWED
            ),
            message="only held_object read queries are accepted",
        )
    if execute:
        return _ValidatedRequest(
            execute=True,
            query_kind=HELD_OBJECT_QUERY_KIND,
            request_id=request_id,
            error_type=(
                RuntimeHeldObjectDispatchReasonCode.EXECUTE_NOT_ALLOWED
            ),
            message="execute=true is forbidden by this read-only dispatcher",
        )
    return _ValidatedRequest(
        execute=False,
        query_kind=HELD_OBJECT_QUERY_KIND,
        request_id=request_id,
        error_type=None,
        message=None,
    )


def _response(
    *,
    status: str,
    reason_code: str,
    request: _ValidatedRequest,
    provider_called: bool,
    reader_called: bool = False,
    read_status: str = "not_called",
    held_object: str | None = None,
    held_object_semantic: str = "not_read",
    read_sequence: int | None = None,
    backend_state_sequence: int | None = None,
    backend_state_sequence_status: str = "not_called",
    error_type: str | None = None,
    exception_type: str | None = None,
    message: str,
) -> L1RuntimeHeldObjectResponse:
    return L1RuntimeHeldObjectResponse(
        status=status,
        reason_code=reason_code,
        execute=request.execute,
        query_kind=request.query_kind,
        dispatcher_called=True,
        provider_called=provider_called,
        reader_called=reader_called,
        read_status=read_status,
        held_object=held_object,
        held_object_semantic=held_object_semantic,
        read_sequence=read_sequence,
        backend_state_sequence=backend_state_sequence,
        backend_state_sequence_status=backend_state_sequence_status,
        action_generated=False,
        action_dispatched=False,
        error_type=error_type,
        exception_type=exception_type,
        message=message[:256],
        request_id=request.request_id,
    )


def _valid_provider_result(value: object) -> bool:
    if type(value) is not RobosuiteHeldObjectProviderRead:
        return False
    if value.adapter_version != ADAPTER_VERSION:
        return False
    if type(value.snapshot) is not HeldObjectSnapshot:
        return False
    if type(value.reader_called) is not bool:
        return False
    if type(value.read_status) is not str or value.read_status not in {
        "known",
        "unknown",
        "unavailable",
        "error",
    }:
        return False
    if value.snapshot.status not in {"known", "unknown", "error"}:
        return False
    expected_snapshot_status = (
        "error"
        if value.read_status in {"unavailable", "error"}
        else value.read_status
    )
    if value.snapshot.status != expected_snapshot_status:
        return False
    held_object = value.snapshot.held_object
    if held_object is not None and (
        type(held_object) is not str
        or _HELD_OBJECT_NAME.fullmatch(held_object) is None
    ):
        return False
    expected_semantic = (
        "not_held"
        if value.read_status == "known" and held_object is None
        else "held"
        if value.read_status == "known"
        else value.read_status
        if value.read_status in {"unknown", "unavailable"}
        else "error"
    )
    if (
        type(value.held_object_semantic) is not str
        or value.held_object_semantic != expected_semantic
    ):
        return False
    if value.read_sequence is not None and (
        type(value.read_sequence) is not int or value.read_sequence < 1
    ):
        return False
    if value.backend_state_sequence is not None:
        return False
    if value.backend_state_sequence_status != "unknown":
        return False
    if value.error_type is not None and (
        type(value.error_type) is not str
        or _ERROR_TYPE.fullmatch(value.error_type) is None
    ):
        return False
    if value.read_status in {"unavailable", "error"} and (
        value.error_type is None
    ):
        return False
    if value.exception_type is not None and (
        type(value.exception_type) is not str
        or _EXCEPTION_TYPE.fullmatch(value.exception_type) is None
    ):
        return False
    return value.message is None or (
        type(value.message) is str and len(value.message) <= 256
    )


class L1RuntimeHeldObjectDispatcher:
    """Read-only dispatcher with one injected Provider and one read path."""

    def __init__(self, *, provider: RuntimeHeldObjectProvider) -> None:
        self._provider = provider

    def dispatch(self, request: object) -> L1RuntimeHeldObjectResponse:
        checked = _validate_request(request)
        if checked.error_type is not None:
            return _response(
                status=RuntimeHeldObjectDispatchStatus.REJECTED,
                reason_code=checked.error_type,
                request=checked,
                provider_called=False,
                error_type=checked.error_type,
                message=checked.message or "request rejected",
            )

        try:
            method = getattr(self._provider, "read_snapshot", None)
        except BaseException:
            method = None
        if not callable(method):
            return _response(
                status=RuntimeHeldObjectDispatchStatus.READ_FAILED,
                reason_code=(
                    RuntimeHeldObjectDispatchReasonCode.PROVIDER_INVALID
                ),
                request=checked,
                provider_called=False,
                error_type=(
                    RuntimeHeldObjectDispatchReasonCode.PROVIDER_INVALID
                ),
                message="injected Provider does not implement the read protocol",
            )
        try:
            provider_result = method()
        except BaseException:
            return _response(
                status=RuntimeHeldObjectDispatchStatus.READ_FAILED,
                reason_code=(
                    RuntimeHeldObjectDispatchReasonCode.PROVIDER_FAILED
                ),
                request=checked,
                provider_called=True,
                error_type=(
                    RuntimeHeldObjectDispatchReasonCode.PROVIDER_FAILED
                ),
                message="injected Provider failed during the read",
            )
        try:
            provider_result_valid = _valid_provider_result(provider_result)
        except BaseException:
            provider_result_valid = False
        if not provider_result_valid:
            return _response(
                status=RuntimeHeldObjectDispatchStatus.READ_FAILED,
                reason_code=(
                    RuntimeHeldObjectDispatchReasonCode
                    .PROVIDER_RESULT_INVALID
                ),
                request=checked,
                provider_called=True,
                error_type=(
                    RuntimeHeldObjectDispatchReasonCode
                    .PROVIDER_RESULT_INVALID
                ),
                message="injected Provider returned an invalid result",
            )

        if provider_result.read_status == "known":
            status = RuntimeHeldObjectDispatchStatus.SUCCESS
            reason_code = (
                RuntimeHeldObjectDispatchReasonCode
                .HELD_OBJECT_READ_SUCCEEDED
            )
            message = (
                "held-object read succeeded; no object is held"
                if provider_result.snapshot.held_object is None
                else "held-object read succeeded"
            )
        elif provider_result.read_status == "unknown":
            status = RuntimeHeldObjectDispatchStatus.READ_INDETERMINATE
            reason_code = (
                RuntimeHeldObjectDispatchReasonCode
                .HELD_OBJECT_STATE_UNKNOWN
            )
            message = "held-object state is unknown"
        elif provider_result.read_status == "unavailable":
            status = RuntimeHeldObjectDispatchStatus.READ_FAILED
            reason_code = (
                RuntimeHeldObjectDispatchReasonCode
                .HELD_OBJECT_READ_UNAVAILABLE
            )
            message = "held-object Reader is unavailable"
        else:
            status = RuntimeHeldObjectDispatchStatus.READ_FAILED
            reason_code = (
                RuntimeHeldObjectDispatchReasonCode.HELD_OBJECT_READ_FAILED
            )
            message = "held-object read failed"

        return _response(
            status=status,
            reason_code=reason_code,
            request=checked,
            provider_called=True,
            reader_called=provider_result.reader_called,
            read_status=provider_result.read_status,
            held_object=provider_result.snapshot.held_object,
            held_object_semantic=provider_result.held_object_semantic,
            read_sequence=provider_result.read_sequence,
            backend_state_sequence=(
                provider_result.backend_state_sequence
            ),
            backend_state_sequence_status=(
                provider_result.backend_state_sequence_status
            ),
            error_type=provider_result.error_type,
            exception_type=provider_result.exception_type,
            message=message,
        )


def run_l1_runtime_held_object_dry_run(
    request: object,
    *,
    provider: RuntimeHeldObjectProvider,
) -> L1RuntimeHeldObjectResponse:
    """Dispatch one read-only held-object query through an injected Provider."""

    return L1RuntimeHeldObjectDispatcher(provider=provider).dispatch(request)


__all__ = [
    "DISPATCHER_VERSION",
    "HELD_OBJECT_QUERY_KIND",
    "L1RuntimeHeldObjectDispatcher",
    "L1RuntimeHeldObjectRequest",
    "L1RuntimeHeldObjectResponse",
    "RuntimeHeldObjectDispatchReasonCode",
    "RuntimeHeldObjectDispatchStatus",
    "RuntimeHeldObjectProvider",
    "run_l1_runtime_held_object_dry_run",
]
