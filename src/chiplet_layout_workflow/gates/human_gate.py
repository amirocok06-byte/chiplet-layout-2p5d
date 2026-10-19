from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal


GateDecision = Literal["approve", "revise", "reject"]


class InvalidGateDecision(ValueError):
    """Raised when a human gate produces a value outside its small protocol."""


class GateResult(str):
    """String-compatible gate result with explicit audit state.

    Being string-compatible keeps the gate convenient for simple callers while
    the attributes make disabled gates impossible to mistake for approval.
    """

    decision: GateDecision | None
    enabled: bool
    status: str
    operator_note: str

    def __new__(
        cls,
        value: str,
        *,
        decision: GateDecision | None,
        enabled: bool,
        status: str,
        operator_note: str = "",
    ) -> "GateResult":
        result = str.__new__(cls, value)
        result.decision = decision
        result.enabled = enabled
        result.status = status
        result.operator_note = operator_note
        return result


GateCallback = Callable[[str, str], str | GateResult | tuple[str, str]]


class HumanGate:
    """Collect an explicit approve/revise/reject decision.

    A callback is the test seam. Without one, the gate reads from stdin. A
    disabled gate returns an explicit ``disabled`` result and never approves.
    """

    def __init__(
        self,
        callback: GateCallback | None = None,
        *,
        enabled: bool = True,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], Any] = print,
    ) -> None:
        self.enabled = enabled
        self.callback = callback
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.last_result: GateResult | None = None
        self.history: list[GateResult] = []

    def request(self, stage: str, summary: str) -> GateResult:
        if not self.enabled:
            result = GateResult(
                "disabled",
                decision=None,
                enabled=False,
                status="disabled",
            )
            self._record(result)
            return result

        if self.callback is None:
            self.output_fn(f"Human review required for stage '{stage}': {summary}")
            raw: str | GateResult | tuple[str, str] = self.input_fn(
                "Decision [approve/revise/reject]: "
            )
            while True:
                try:
                    result = self._make_result(raw)
                    break
                except InvalidGateDecision:
                    self.output_fn("Enter exactly approve, revise, or reject.")
                    raw = self.input_fn("Decision [approve/revise/reject]: ")
        else:
            result = self._make_result(self.callback(stage, summary))

        self._record(result)
        return result

    def manifest_metadata(self) -> dict[str, bool]:
        """Return the gate configuration fields that belong in a manifest."""
        return {"human_gates_enabled": self.enabled}

    def _make_result(self, raw: str | GateResult | tuple[str, str]) -> GateResult:
        note = ""
        if isinstance(raw, tuple):
            if len(raw) != 2:
                raise InvalidGateDecision(
                    "human gate decision must be approve, revise, or reject"
                )
            raw, note = raw
        if isinstance(raw, GateResult):
            if raw.decision is None:
                raise InvalidGateDecision(
                    "human gate decision must be approve, revise, or reject"
                )
            note = raw.operator_note
            raw = raw.decision
        if not isinstance(raw, str):
            raise InvalidGateDecision(
                "human gate decision must be approve, revise, or reject"
            )
        decision = raw.strip().lower()
        if decision not in {"approve", "revise", "reject"}:
            raise InvalidGateDecision(
                "human gate decision must be approve, revise, or reject"
            )
        typed_decision = decision  # narrowed by the membership check above
        return GateResult(
            typed_decision,
            decision=typed_decision,
            enabled=True,
            status="approved" if typed_decision == "approve" else typed_decision,
            operator_note=str(note),
        )

    def _record(self, result: GateResult) -> None:
        self.last_result = result
        self.history.append(result)
