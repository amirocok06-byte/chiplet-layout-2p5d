"""Explicit, independently owned geometry variants; source data stays intact."""
from __future__ import annotations

from copy import deepcopy

import numpy as np


CASE4_59MM_VERSION = "case4-maskplace-59mm-alternative-v1"
MASKPLACE_CASE4_SHA256 = "87397dc96a29ed206e446b76ac58d8a7ef95815e78d0a52c97509cbd9549e9cf"
MASKPLACE_CASE4_SOURCE = (
    "F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/"
    "4-开源源码/MaskPlace2.5D/cases/Case4/Case4.intpsize"
)
_ALIASES = {"case4-59mm-alternative": CASE4_59MM_VERSION}


def apply_geometry_version(problem, metadata, version):
    """Return owned (problem, metadata), renormalizing all spatial arrays.

    ``original`` preserves the loaded board (37 mm for Case4). The explicit
    Case4 alternative uses the locally audited MaskPlace 59000 um square.
    This frozen provenance does not require that machine's files at runtime.
    No objective/gradient evaluator is called. Derived TracedProblem fields
    and objective normalization are recomputed when the board changes.
    """
    canonical = _ALIASES.get(version, version)
    if canonical not in ("original", CASE4_59MM_VERSION):
        raise ValueError(f"unknown geometry version: {version!r}")
    if canonical == CASE4_59MM_VERSION and metadata.get("case") != "Case4":
        raise ValueError("the 59 mm alternative is only defined for Case4")

    new, info = deepcopy(problem), deepcopy(metadata)
    origin = deepcopy(info.get("geometry_origin", {
        "board_mm": np.asarray(problem.board_mm).tolist(),
        "board_source": info.get("board_source", "loaded problem geometry"),
    }))
    old_board = np.asarray(problem.board_mm, dtype=float)
    board = np.array([59., 59.] if canonical == CASE4_59MM_VERSION
                     else origin["board_mm"], dtype=float)
    if (old_board.shape != (2,) or board.shape != (2,)
            or not np.isfinite(old_board).all() or not np.isfinite(board).all()
            or np.any(old_board <= 0) or np.any(board <= 0)):
        raise ValueError("board dimensions must be finite positive axis lengths")
    fields = ("sizes", "pin_offsets", "initial_positions")
    if not np.array_equal(old_board, board):
        new.board_mm = board.copy()
        for field in fields:
            setattr(new, field, np.asarray(getattr(problem, field)) * old_board / board)
        # Rebuild normalized geometry, thermal variances and reference scales.
        new.__post_init__()
    for field in fields:
        if not np.allclose(getattr(new, field) * board,
                           getattr(problem, field) * old_board,
                           rtol=1e-13, atol=1e-13):
            raise ValueError(f"physical invariant failed: {field}")

    info.update({
        "geometry_version": canonical,
        "geometry_version_requested": version,
        "geometry_origin": origin,
        "board_um": (board * 1000).tolist(),
        "area_utilization": float(new.sizes.prod(axis=1).sum()),
        "proven_infeasible_by_area": bool(new.sizes.prod(axis=1).sum() > 1.),
        "physical_invariants_verified": True,
    })
    if canonical == CASE4_59MM_VERSION:
        info["board_source"] = "MaskPlace2.5D Case4.intpsize; explicit alternative"
        info["geometry_provenance"] = {
            "classification": "alternative, not official correction",
            "source_path": MASKPLACE_CASE4_SOURCE,
            "source_sha256": MASKPLACE_CASE4_SHA256,
            "source_content": "59000.00000",
            "source_unit": "um",
            "interpretation": "width = height = float(first line)",
            "interpretation_source": "MaskPlace2.5D/process_thermal.py:9-15",
            "locally_verified_on": "2026-10-25",
            "original_case4_board_mm": [37., 37.],
        }
    else:
        info["board_source"] = origin["board_source"]
        info["geometry_provenance"] = {
            "classification": "original loaded geometry",
            "board_source": origin["board_source"],
        }
    return new, info
