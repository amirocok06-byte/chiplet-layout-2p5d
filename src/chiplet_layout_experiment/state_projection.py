"""Versioned four-channel observations; exact geometry remains the judge."""
from __future__ import annotations

import numpy as np

STATE_VERSION = "connectivity-priority-thermal-hpwl-legality-v2"
CHANNELS = ("priority", "thermal_proxy", "hpwl", "legality")


def _coverage(lo, hi, grid_size):
    edges = np.arange(grid_size, dtype=float) / grid_size
    return np.maximum(np.minimum(edges + 1 / grid_size, hi)
                      - np.maximum(edges, lo), 0.) * grid_size


def legality_map(positions, sizes, grid_size=256):
    """Exact pair-intersection area fractions, plus edge-projected overshoot.

    Pair contributions are normalized by max(N-1,1). No overlap is inferred
    from two bounding boxes merely touching the same raster cell. Boundary
    overshoot is normalized per board axis, weighted by tangential coverage,
    summed into a one-pixel boundary band and divided by N. The map is a
    spatial diagnostic: exact valid/overlap/boundary scalars accompany it.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    x, size = np.asarray(positions, float), np.asarray(sizes, float)
    if (x.ndim != 2 or x.shape[1] != 2 or size.shape != x.shape or len(x) < 1
            or not np.isfinite(x).all() or not np.isfinite(size).all() or (size <= 0).any()):
        raise ValueError("positions/sizes must be finite (N,2), with positive sizes")
    result = np.zeros((grid_size, grid_size), dtype=float)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            lo = np.maximum(x[i], x[j])
            hi = np.minimum(x[i] + size[i], x[j] + size[j])
            if (hi > lo).all():
                result += np.outer(_coverage(lo[1], hi[1], grid_size),
                                   _coverage(lo[0], hi[0], grid_size)) / max(len(x)-1, 1)
    for pos, dimensions in zip(x, size):
        for axis in (0, 1):
            other = 1-axis
            coverage = _coverage(pos[other], pos[other]+dimensions[other], grid_size)
            if not coverage.any():
                coverage[int(np.clip((pos[other]+dimensions[other]/2)*grid_size, 0, grid_size-1))] = 1.
            for side, amount in ((0, max(-pos[axis], 0.)),
                                 (-1, max(pos[axis]+dimensions[axis]-1., 0.))):
                if axis == 0:
                    result[:, side] += amount * coverage / len(x)
                else:
                    result[side, :] += amount * coverage / len(x)
    return result.astype(np.float32)


def project_state(problem, positions, grid_size=256):
    """Return [priority, thermal proxy, HPWL contribution, legality].

    Priority is max incident-pin degree/max degree over all chiplets covering
    each pixel; ties persist. Max makes overlap display independent of input
    ordering. Empty background=-1, including disconnected chiplets' q=0.
    Legacy maps/evaluator and recorded rewards are not modified.
    """
    old = problem.maps(positions, grid_size=grid_size)
    priority = np.full((grid_size, grid_size), -1., dtype=np.float32)
    for pos, size, value in zip(np.asarray(positions), problem.sizes, problem.priority):
        occupied = np.outer(_coverage(pos[1], pos[1]+size[1], grid_size),
                            _coverage(pos[0], pos[0]+size[0], grid_size)) > 0
        priority[occupied] = np.maximum(priority[occupied], value)
    return np.stack((priority, old[1], old[2],
                     legality_map(positions, problem.sizes, grid_size)))
