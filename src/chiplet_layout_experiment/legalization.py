"""Bounded deterministic rectangle legalization, with no objective calls."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


_BOUND_TOL = 1e-10
_OVERLAP_TOL = 1e-10
_FIT_TOL = 1e-14  # roundoff only; final verdict always uses evaluator tolerances


@dataclass
class LegalizationResult:
    positions: np.ndarray
    valid: bool
    status: str
    method: str
    wall_seconds: float
    geometry_checks: int
    displacement: float
    displacement_mm: float
    attempts: tuple[dict, ...]
    overlap_fraction: float
    boundary_valid: bool


def _geometry(sizes, positions, pair_i, pair_j, area):
    """Same intersection and boundary arithmetic as TracedProblem.evaluate."""
    lo = np.maximum(positions[pair_i], positions[pair_j])
    hi = np.minimum(positions[pair_i] + sizes[pair_i],
                    positions[pair_j] + sizes[pair_j])
    overlap = float(np.maximum(hi - lo, 0).prod(axis=1).sum() / area)
    boundary = bool(np.all(positions >= -_BOUND_TOL)
                    and np.all(positions + sizes <= 1 + _BOUND_TOL))
    return boundary and overlap < _OVERLAP_TOL, overlap, boundary


def _split_free(free, lo, hi):
    """MaxRects splitting: keep all empty subrectangles, prune containment."""
    intersects = (np.minimum(free[:, 2:], hi)
                  > np.maximum(free[:, :2], lo)).all(axis=1)
    parts = [free[~intersects]]
    cut = free[intersects]
    for axis in (0, 1):
        left, right = cut.copy(), cut.copy()
        left[:, axis + 2] = np.minimum(left[:, axis + 2], lo[axis])
        right[:, axis] = np.maximum(right[:, axis], hi[axis])
        parts.extend((left, right))
    remaining = np.concatenate(parts)
    remaining = remaining[(remaining[:, 2:] - remaining[:, :2] > _FIT_TOL).all(axis=1)]
    remaining = np.unique(remaining, axis=0)
    if len(remaining) < 2:
        return remaining
    # Row i is contained in column j. Equal rows were removed above.
    contained = ((remaining[:, None, :2] >= remaining[None, :, :2]).all(axis=2)
                 & (remaining[:, None, 2:] <= remaining[None, :, 2:]).all(axis=2))
    np.fill_diagonal(contained, False)
    return remaining[~contained.any(axis=1)]


def _maxrects(sizes, target, order, nearest):
    free = np.array([[0., 0., 1., 1.]])
    positions = np.empty_like(target)
    for placed, index in enumerate(order):
        size = sizes[index]
        fitting = free[(free[:, 2:] - free[:, :2] >= size - _FIT_TOL).all(axis=1)]
        if not len(fitting):
            return None, placed
        room = fitting[:, 2:] - fitting[:, :2] - size
        if nearest:
            candidates = np.maximum(fitting[:, :2],
                                    np.minimum(target[index], fitting[:, 2:] - size))
            score = ((candidates - target[index]) ** 2).sum(axis=1)
            choice = int(np.lexsort((room.max(axis=1), room.min(axis=1), score))[0])
        else:
            candidates = fitting[:, :2]
            choice = int(np.lexsort((fitting[:, 0], fitting[:, 1],
                                     room.max(axis=1), room.min(axis=1)))[0])
        positions[index] = candidates[choice]
        free = _split_free(free, positions[index], positions[index] + size)
    return positions, len(order)


def _shelf(sizes, target, order, transpose):
    dimensions = sizes[:, ::-1] if transpose else sizes
    positions = np.empty_like(target)
    x, y, height = 0., 0., 0.
    for placed, index in enumerate(order):
        width, item_height = dimensions[index]
        if x + width > 1 + _FIT_TOL:
            x, y, height = 0., y + height, 0.
        if y + item_height > 1 + _FIT_TOL:
            return None, placed
        positions[index] = (y, x) if transpose else (x, y)
        x += width
        height = max(height, item_height)
    return positions, len(order)


def legalize(problem, positions, *, max_attempts=6):
    """Preserve valid input, clip, then try at most six deterministic packings.

    Coordinates are normalized lower-left corners; sizes/orientations stay
    fixed. ``max_attempts`` selects a prefix of the six packing strategies.
    It excludes the two cheap input/clip checks. No randomized search,
    objective evaluation, gradient evaluation or hidden solver is used.
    A failed search is not a proof of infeasibility. Displacement is the
    Frobenius norm of output minus input, in normalized axes (and in mm).
    """
    started = perf_counter()
    if (isinstance(max_attempts, (bool, np.bool_))
            or not isinstance(max_attempts, (int, np.integer)) or max_attempts < 0):
        raise ValueError("max_attempts must be a nonnegative integer")
    sizes = np.asarray(problem.sizes, dtype=float)
    original = np.array(positions, dtype=float, copy=True)
    if original.shape != (problem.n, 2) or not np.isfinite(original).all():
        raise ValueError("positions must be finite with shape (n, 2)")
    if (sizes.shape != original.shape or not np.isfinite(sizes).all()
            or np.any(sizes <= 0) or problem.n == 0):
        raise ValueError("sizes must be finite positive rectangles of shape (n, 2)")
    board = np.asarray(problem.board_mm, dtype=float)
    if board.shape != (2,) or not np.isfinite(board).all() or np.any(board <= 0):
        raise ValueError("board_mm must contain two finite positive lengths")
    area = float(sizes.prod(axis=1).sum())
    pair_i, pair_j = np.triu_indices(problem.n, k=1)
    checks, attempts = 0, []

    def check(candidate):
        nonlocal checks
        checks += 1
        return _geometry(sizes, candidate, pair_i, pair_j, area)

    def finish(candidate, verdict, status, method):
        valid, overlap, boundary = verdict
        delta = candidate - original
        return LegalizationResult(
            positions=candidate.copy(), valid=bool(valid), status=status, method=method,
            wall_seconds=perf_counter() - started, geometry_checks=checks,
            displacement=float(np.linalg.norm(delta)),
            displacement_mm=float(np.linalg.norm(delta * board)),
            attempts=tuple(attempts), overlap_fraction=float(overlap),
            boundary_valid=bool(boundary),
        )

    initial = check(original)
    if initial[0]:
        return finish(original, initial, "already_valid", "identity")
    # Tolerance-aware necessary conditions: even an evaluator-valid layout
    # cannot exceed this expanded board area plus its allowed pair overlap.
    if area > (1 + 2 * _BOUND_TOL) ** 2 / (1 - _OVERLAP_TOL):
        return finish(original, initial, "proven_infeasible_by_area", "proof")
    if np.any(sizes > 1 + 2 * _BOUND_TOL):
        return finish(original, initial, "proven_infeasible_by_dimensions", "proof")
    clipped = np.clip(original, 0., np.maximum(1. - sizes, 0.))
    if not np.array_equal(clipped, original):
        verdict = check(clipped)
        if verdict[0]:
            return finish(clipped, verdict, "legalized", "clip")

    indices = np.arange(problem.n)
    area_order = np.lexsort((indices, -sizes.max(axis=1), -sizes.prod(axis=1)))
    height_order = np.lexsort((indices, -sizes[:, 0], -sizes[:, 1]))
    width_order = np.lexsort((indices, -sizes[:, 1], -sizes[:, 0]))
    strategies = (
        ("maxrects-nearest-area", area_order, "nearest"),
        ("maxrects-short-side-area", area_order, "corner"),
        ("maxrects-short-side-height", height_order, "corner"),
        ("maxrects-short-side-width", width_order, "corner"),
        ("shelf-height", height_order, "shelf"),
        ("shelf-width", width_order, "shelf-transposed"),
    )
    for method, order, strategy in strategies[:max_attempts]:
        if strategy.startswith("shelf"):
            candidate, placed = _shelf(sizes, clipped, order, strategy == "shelf-transposed")
        else:
            candidate, placed = _maxrects(sizes, clipped, order, strategy == "nearest")
        verdict = check(candidate) if candidate is not None else None
        attempts.append({"method": method, "placed": int(placed),
                         "complete": candidate is not None,
                         "valid": bool(verdict and verdict[0])})
        if verdict and verdict[0]:
            return finish(candidate, verdict, "legalized", method)
    return finish(original, initial, "search_exhausted", "bounded_packing")
