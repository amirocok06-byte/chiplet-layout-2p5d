"""Read data files without importing/executing any third-party placer."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import numpy as np

from .trajectory_problem import TracedProblem


BOARD_UM = {
    "Case1": (42000, 42000), "Case2": (32000, 32000),
    "Case3": (39000, 39000), "Case4": (37000, 37000),
    "Case5": (57000, 59000), "Case6": (49000, 53000),
    "Case7": (30000, 25000), "Case8": (26000, 23000),
    "Case9": (59000, 61000), "Case10": (47000, 47000),
}


def load_case(root: Path, case: str, *, seed: int):
    directory = root / case
    board = np.array(BOARD_UM[case], dtype=float)
    names, sizes = [], []
    for line in (directory / f"{case}.blocks").read_text(encoding="utf-8-sig").splitlines():
        if " hardrectilinear " not in line:
            continue
        vertices = np.array(re.findall(r"\(\s*([-\d.]+),\s*([-\d.]+)\s*\)", line), float)
        names.append(line.split()[0])
        sizes.append(vertices.max(axis=0) - vertices.min(axis=0))
    sizes_um = np.array(sizes)
    names = tuple(names)
    ids = {name: i for i, name in enumerate(names)}
    powers = dict(line.split() for line in
                  (directory / f"{case}.power").read_text(encoding="utf-8-sig").splitlines()
                  if line.strip())
    pin_indices, offsets = [], []
    current_pins, current_offsets, expected = [], [], 0
    for line in (directory / f"{case}.nets").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("NetDegree"):
            expected = int(line.split(":")[1].split()[0])
            if expected != 2:
                raise ValueError(f"{case}: only two-pin nets are supported")
            current_pins, current_offsets = [], []
        elif expected and line.strip():
            fields = line.split()
            idx = ids[fields[0]]
            parts = line.split(":")[1].split() if ":" in line else []
            offset = sizes_um[idx] / 2
            if parts:
                for axis, value in enumerate(parts[:2]):
                    offset[axis] += (float(value[1:]) / 100 * sizes_um[idx, axis]
                                     if value.startswith("%") else float(value))
            current_pins.append(idx)
            current_offsets.append(offset / board)
            if len(current_pins) == expected:
                pin_indices.append(current_pins)
                offsets.append(current_offsets)
                expected = 0
    if expected:
        raise ValueError(f"{case}: incomplete last net")
    sizes_normalized = sizes_um / board
    rng = np.random.default_rng(seed)
    # Public .pl files are coincident zero placeholders; use a reproducible
    # bounded scatter shared by all algorithms, and report overlap explicitly.
    initial = rng.uniform(.05, .95, size=(len(names), 2)) * (1 - sizes_normalized)
    problem = TracedProblem(
        names=names, board_mm=board / 1000, sizes=sizes_normalized,
        powers=np.array([float(powers[n]) for n in names]),
        pin_indices=np.array(pin_indices), pin_offsets=np.array(offsets),
        initial_positions=initial,
    )
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(directory.iterdir()) if p.is_file()}
    metadata = {
        "case": case, "source_directory": str(directory),
        "files_sha256": hashes, "coordinate_unit": "um->normalized-axis",
        "wirelength_unit": "mm", "pin_offset_rule": "centre + percentage*dimension",
        "board_um": board.tolist(), "num_chiplets": len(names),
        "area_utilization": float(problem.area_sum),
        "proven_infeasible_by_area": bool(problem.area_sum > 1.),
        "board_source": "ATPlace_pub/reproduce.py CASE_INTERPOSER_SIZE; retained verbatim",
        "num_nets": len(pin_indices), "seed": seed,
        "initialization": "seeded-bounded-scatter; public .pl zero placeholders",
        "priority_source": "normalized incident pin count; ties preserved; no precedence",
        "thermal_model": "uncalibrated Gaussian power-density spreading proxy",
    }
    return problem, metadata
