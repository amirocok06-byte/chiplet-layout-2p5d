"""Immutable, case-grouped E2 data reconstructed from Stage 0 artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .state_projection import project_state
from .trajectory_problem import TracedProblem


@dataclass(frozen=True)
class Stage1State:
    """One contextual-bandit observation with full-information arm labels."""

    state_id: str
    case_group: str
    geometry_version: str
    phase: str
    maps: np.ndarray
    scalars: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    budget: int


@dataclass(frozen=True)
class Stage1Dataset:
    """Aligned observations; arm_ids define every reward/cost column."""

    arm_ids: tuple[str, ...]
    states: tuple[Stage1State, ...]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _problem_from_saved_case(path: Path, positions: np.ndarray) -> TracedProblem:
    source = _read_json(path / "problem.json")
    problem = TracedProblem(
        names=tuple(source["names"]),
        board_mm=np.asarray(source["board_mm"], dtype=float),
        sizes=np.asarray(source["sizes"], dtype=float),
        powers=np.asarray(source["powers"], dtype=float),
        pin_indices=np.asarray(source["pin_indices"], dtype=int),
        pin_offsets=np.asarray(source["pin_offsets"], dtype=float),
        initial_positions=positions,
    )
    problem.hpwl_scale = float(source["hpwl_scale"])
    problem.thermal_scale = float(source["thermal_scale"])
    return problem


def _state_from_rows(case_dir: Path, state_id: str, rows: list[dict], grid_size: int):
    metadata = rows[0]["state_metadata"]
    if any(row["state_metadata"] != metadata for row in rows):
        raise ValueError(f"{state_id}: inconsistent state metadata")
    state_path = case_dir / metadata["state_file"]
    state_data = _read_json(state_path)
    positions = np.asarray(state_data["positions"], dtype=float)
    problem = _problem_from_saved_case(case_dir, positions)
    first_metrics = rows[0]["input_metrics"]
    scalars = np.asarray((
        float(first_metrics["hpwl_mm"]) / problem.hpwl_scale,
        float(first_metrics["thermal_proxy_peak"]) / problem.thermal_scale,
        float(first_metrics["overlap_fraction"]),
        float(bool(first_metrics["valid"])),
    ), dtype=np.float32)
    arms: dict[str, list[dict]] = {}
    for row in rows:
        if row["evaluator_calls"] != row["budget"] or row["candidate_budget"] != row["budget"] - 1:
            raise ValueError(f"{state_id}: invalid macro-action evaluation budget")
        arms.setdefault(row["arm_id"], []).append(row)
    expected_seeds = None
    rewards, costs = {}, {}
    for arm_id, entries in arms.items():
        seeds = tuple(sorted(row["seed"] for row in entries))
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{state_id}: duplicate arm/seed label")
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError(f"{state_id}: incomplete arm/seed grid")
        rewards[arm_id] = float(np.mean([row["bandit_reward"] for row in entries]))
        costs[arm_id] = float(np.mean([row["total_action_seconds"] for row in entries]))
    arm_ids = tuple(sorted(arms))
    return arm_ids, Stage1State(
        state_id=state_id,
        case_group=str(metadata["case_split_group"]),
        geometry_version=str(metadata.get("geometry_version", "original")),
        phase=str(rows[0]["phase"]),
        maps=project_state(problem, positions, grid_size=grid_size).astype(np.float32),
        scalars=scalars,
        rewards=np.asarray([rewards[arm] for arm in arm_ids], dtype=np.float32),
        costs=np.asarray([costs[arm] for arm in arm_ids], dtype=np.float32),
        budget=int(rows[0]["budget"]),
    )


def load_stage1_dataset(root: Path | str, *, phase: str = "legal_optimization",
                        grid_size: int = 64) -> Stage1Dataset:
    """Load one phase from Stage 0 without mutating labels or coordinates.

    Rows are aggregated across repeated seeds for each ``(state, arm)``.  The
    returned observations remain grouped by their benchmark Case, so callers
    can perform case-level rather than state-level train/test splits.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"labels root does not exist: {root}")
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        labels_path = case_dir / "labels.jsonl"
        if not labels_path.exists():
            continue
        for row in _read_jsonl(labels_path):
            if row.get("phase") != phase:
                continue
            grouped.setdefault((case_dir.name, row["state_id"]), []).append(row)
    if not grouped:
        raise ValueError(f"no {phase!r} labels found under {root}")
    expected_arms = None
    states = []
    for (case_name, state_id), rows in sorted(grouped.items()):
        arm_ids, state = _state_from_rows(root / case_name, state_id, rows, grid_size)
        if expected_arms is None:
            expected_arms = arm_ids
        elif arm_ids != expected_arms:
            raise ValueError(f"{state_id}: inconsistent arm catalog")
        states.append(state)
    return Stage1Dataset(arm_ids=expected_arms, states=tuple(states))


def split_by_case(dataset: Stage1Dataset, *, train_cases: list[str],
                  validation_cases: list[str], test_cases: list[str]
                  ) -> tuple[Stage1Dataset, Stage1Dataset, Stage1Dataset]:
    """Make a complete, mutually exclusive Case-level E2 split.

    A state is never the unit of random splitting: checkpoints from one Case
    share geometry and trajectory ancestry.  Requiring an explicit partition
    prevents this source of train/test leakage, including Case4 variants whose
    states still belong to the single ``Case4`` group.
    """
    partitions = {
        "train": tuple(train_cases),
        "validation": tuple(validation_cases),
        "test": tuple(test_cases),
    }
    all_requested = tuple(case for cases in partitions.values() for case in cases)
    if len(set(all_requested)) != len(all_requested):
        raise ValueError("case split partitions overlap")
    present = {state.case_group for state in dataset.states}
    requested = set(all_requested)
    if requested != present:
        missing = sorted(present - requested)
        unknown = sorted(requested - present)
        raise ValueError(
            "case split must cover every dataset case group "
            f"(missing={missing}, unknown={unknown})"
        )
    def subset(case_groups: tuple[str, ...]) -> Stage1Dataset:
        selected = tuple(state for state in dataset.states
                         if state.case_group in set(case_groups))
        return Stage1Dataset(arm_ids=dataset.arm_ids, states=selected)
    return tuple(subset(partitions[name]) for name in ("train", "validation", "test"))
