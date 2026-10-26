from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from chiplet_layout_experiment.geometry_versions import (
    CASE4_59MM_VERSION, MASKPLACE_CASE4_SHA256, apply_geometry_version,
)
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def case4():
    p = TracedProblem(
        names=("a", "b"), board_mm=np.array([37., 37.]),
        sizes=np.full((2, 2), 30 / 37), powers=np.array([2., 4.]),
        pin_indices=np.array([[0, 1]]),
        pin_offsets=np.array([[[3., 7.], [11., 13.]]]) / 37,
        initial_positions=np.array([[1., 2.], [3., 4.]]) / 37,
    )
    return p, {"case": "Case4", "board_um": [37000., 37000.],
               "board_source": "original ATPlace", "files_sha256": {"x": "abc"}}


def test_alternative_preserves_physical_values_and_does_not_mutate_inputs():
    p, meta = case4()
    snapshot, original_meta = deepcopy(p), deepcopy(meta)
    new, info = apply_geometry_version(p, meta, CASE4_59MM_VERSION)
    np.testing.assert_array_equal(new.board_mm, [59., 59.])
    for field in ("sizes", "pin_offsets", "initial_positions"):
        np.testing.assert_allclose(getattr(new, field) * new.board_mm,
                                   getattr(p, field) * p.board_mm, rtol=1e-14)
    np.testing.assert_array_equal(new.powers, p.powers)
    np.testing.assert_array_equal(new.pin_indices, p.pin_indices)
    assert info["geometry_version"] == "case4-maskplace-59mm-alternative-v1"
    assert info["geometry_provenance"]["source_sha256"] == MASKPLACE_CASE4_SHA256
    assert info["geometry_provenance"]["classification"] == "alternative, not official correction"
    assert info["physical_invariants_verified"]
    assert info["board_um"] == [59000., 59000.]
    assert info["area_utilization"] == pytest.approx(1800 / 3481)
    assert not info["proven_infeasible_by_area"]
    for field in vars(snapshot):
        if isinstance(getattr(snapshot, field), np.ndarray):
            np.testing.assert_array_equal(getattr(p, field), getattr(snapshot, field))
        else:
            assert getattr(p, field) == getattr(snapshot, field)
    assert meta == original_meta
    new.sizes[0, 0] = 0
    info["files_sha256"]["x"] = "changed"
    assert p.sizes[0, 0] > 0
    assert meta == original_meta


def test_original_retains_impossible_37mm_and_owns_its_arrays():
    p, meta = case4()
    new, info = apply_geometry_version(p, meta, "original")
    np.testing.assert_array_equal(new.board_mm, [37., 37.])
    assert info["proven_infeasible_by_area"]
    assert info["geometry_version"] == "original"
    assert new is not p and not np.shares_memory(new.sizes, p.sizes)


def test_alias_idempotence_and_return_to_original():
    p, meta = case4()
    new, info = apply_geometry_version(p, meta, "case4-59mm-alternative")
    repeated, repeated_meta = apply_geometry_version(new, info, CASE4_59MM_VERSION)
    np.testing.assert_array_equal(repeated.sizes, new.sizes)
    assert repeated_meta["geometry_version"] == CASE4_59MM_VERSION
    restored, restored_meta = apply_geometry_version(new, info, "original")
    np.testing.assert_allclose(restored.sizes, p.sizes)
    np.testing.assert_array_equal(restored.board_mm, p.board_mm)
    assert restored_meta["board_source"] == meta["board_source"]


def test_unknown_version_and_wrong_case_rejected():
    p, meta = case4()
    with pytest.raises(ValueError, match="version"):
        apply_geometry_version(p, meta, "corrected")
    with pytest.raises(ValueError, match="Case4"):
        apply_geometry_version(p, {**meta, "case": "Case1"}, CASE4_59MM_VERSION)


def test_locally_verified_source_when_available():
    source = Path("F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/"
                  "4-开源源码/MaskPlace2.5D/cases/Case4/Case4.intpsize")
    if not source.exists():
        pytest.skip("optional local provenance audit")
    assert float(source.read_text().strip()) == 59000.
    assert hashlib.sha256(source.read_bytes()).hexdigest() == MASKPLACE_CASE4_SHA256
