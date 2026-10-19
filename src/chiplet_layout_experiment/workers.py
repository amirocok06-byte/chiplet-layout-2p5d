from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Protocol

from .canonical import Layout, Placement
from .evaluator import PhysicsEvaluation, PhysicsEvaluator


@dataclass(frozen=True, slots=True)
class WorkerResult:
    worker_id: str
    status: str
    seed: int
    budget: int
    candidate: Layout | None
    evaluation: PhysicsEvaluation | None
    runtime_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None


class LayoutWorker(Protocol):
    worker_id: str

    def run(
        self,
        layout: Layout,
        *,
        evaluator: PhysicsEvaluator,
        seed: int,
        budget: int,
        config: Mapping[str, Any] | None = None,
    ) -> WorkerResult: ...


def _validate_budget(budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")


def _result(
    worker_id: str,
    layout: Layout,
    *,
    evaluator: PhysicsEvaluator,
    seed: int,
    budget: int,
    started: float,
    metadata: Mapping[str, Any],
) -> WorkerResult:
    evaluation = evaluator.evaluate(layout)
    return WorkerResult(
        worker_id=worker_id,
        status="available",
        seed=seed,
        budget=budget,
        candidate=layout,
        evaluation=evaluation,
        runtime_seconds=max(0.0, time.perf_counter() - started),
        metadata=dict(metadata),
    )


def _sequential_layout(layout: Layout) -> Layout:
    placements: dict[str, Placement] = {}
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for chiplet in layout.chiplets:
        if cursor_x > 0 and cursor_x + chiplet.width > layout.board.width:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0
        if cursor_y + chiplet.height > layout.board.height:
            raise ValueError(f"chiplet {chiplet.name!r} does not fit board")
        placements[chiplet.name] = Placement(cursor_x, cursor_y)
        cursor_x += chiplet.width
        row_height = max(row_height, chiplet.height)
    return Layout(
        board=layout.board,
        chiplets=layout.chiplets,
        placements=placements,
        nets=layout.nets,
        metadata={**dict(layout.metadata or {}), "worker_layout": "sequential"},
    )


class AnalyticalWorker:
    worker_id = "analytical"

    def run(
        self,
        layout: Layout,
        *,
        evaluator: PhysicsEvaluator,
        seed: int,
        budget: int,
        config: Mapping[str, Any] | None = None,
    ) -> WorkerResult:
        _validate_budget(budget)
        started = time.perf_counter()
        candidate = _sequential_layout(layout)
        return _result(
            self.worker_id,
            candidate,
            evaluator=evaluator,
            seed=seed,
            budget=budget,
            started=started,
            metadata={"implementation": "transparent-sequential-baseline", "config": dict(config or {})},
        )


class SAWorker:
    worker_id = "sa"

    def run(
        self,
        layout: Layout,
        *,
        evaluator: PhysicsEvaluator,
        seed: int,
        budget: int,
        config: Mapping[str, Any] | None = None,
    ) -> WorkerResult:
        _validate_budget(budget)
        started = time.perf_counter()
        candidate = _sequential_layout(layout)
        return _result(
            self.worker_id,
            candidate,
            evaluator=evaluator,
            seed=seed,
            budget=budget,
            started=started,
            metadata={"implementation": "bounded-deterministic-sa-smoke", "config": dict(config or {})},
        )


class RLCompatibleWorker:
    worker_id = "rl-compatible"

    def run(
        self,
        layout: Layout,
        *,
        evaluator: PhysicsEvaluator,
        seed: int,
        budget: int,
        config: Mapping[str, Any] | None = None,
    ) -> WorkerResult:
        _validate_budget(budget)
        started = time.perf_counter()
        return _result(
            self.worker_id,
            layout,
            evaluator=evaluator,
            seed=seed,
            budget=budget,
            started=started,
            metadata={
                "implementation": "local-rule-baseline",
                "external_reproduction": False,
                "config": dict(config or {}),
            },
        )
