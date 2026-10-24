"""Inspectable common objective for real-case trajectory demonstrations.

Coordinates are fractions of each board axis; wirelength uses millimetres.
The Gaussian spreading model is an uncalibrated spatial thermal proxy, with
power density amplitudes. Its values are not temperatures in K or Celsius.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TracedProblem:
    names: tuple[str, ...]
    board_mm: np.ndarray
    sizes: np.ndarray
    powers: np.ndarray
    pin_indices: np.ndarray
    pin_offsets: np.ndarray
    initial_positions: np.ndarray

    VERSION = "trajectory-spatial-proxy-v1"

    def __post_init__(self):
        self.n = len(self.names)
        for name in ("board_mm", "sizes", "powers", "pin_offsets", "initial_positions"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.pin_indices = np.asarray(self.pin_indices, dtype=int)
        if self.pin_indices.ndim != 2 or self.pin_indices.shape[1] != 2:
            raise ValueError("trajectory experiment currently supports two-pin nets")
        if self.sizes.shape != (self.n, 2) or self.initial_positions.shape != (self.n, 2):
            raise ValueError("sizes and positions must have shape [chiplet,2]")
        self.pair_i, self.pair_j = np.triu_indices(self.n, k=1)
        self.amplitudes = self.powers / np.prod(self.sizes * self.board_mm, axis=1)
        self.variances = (self.sizes / np.sqrt(12.0)) ** 2 + .08 ** 2
        self.area_sum = self.sizes.prod(axis=1).sum()
        self.hpwl_scale = max(self._hpwl(self.initial_positions), 1e-9)
        self.thermal_scale = max(self._thermal(self.initial_positions)[0], 1e-9)
        degree = np.bincount(self.pin_indices.ravel(), minlength=self.n).astype(float)
        self.priority = degree / max(float(degree.max()), 1.0)
        self.evaluator_calls = 0
        self.gradient_calls = 0

    def clip(self, positions):
        return np.clip(positions, 0., 1. - self.sizes)

    def _pin_delta(self, positions):
        points = positions[self.pin_indices] + self.pin_offsets
        return points[:, 0] - points[:, 1]

    def _hpwl(self, positions):
        return float((np.abs(self._pin_delta(positions)) * self.board_mm).sum())

    def _thermal(self, positions):
        centres = positions + self.sizes / 2
        delta = centres[:, None, :] - centres[None, :, :]
        contributions = self.amplitudes[None, :] * np.exp(
            -.5 * (delta ** 2 / self.variances[None, :, :]).sum(axis=2)
        )
        fields = contributions.sum(axis=1)
        index = int(fields.argmax())
        return float(fields[index]), index, delta, contributions

    def _overlap_penalty(self, positions):
        centres = positions + self.sizes / 2
        delta = centres[self.pair_i] - centres[self.pair_j]
        penetration = (self.sizes[self.pair_i] + self.sizes[self.pair_j]) / 2 - abs(delta)
        active = np.maximum(penetration, 0)
        return float(active.prod(axis=1).sum() / self.area_sum), delta, active

    def _exact_overlap(self, positions):
        lo = np.maximum(positions[self.pair_i], positions[self.pair_j])
        hi = np.minimum(positions[self.pair_i] + self.sizes[self.pair_i],
                        positions[self.pair_j] + self.sizes[self.pair_j])
        return float(np.maximum(hi - lo, 0).prod(axis=1).sum() / self.area_sum)

    def evaluate(self, positions):
        self.evaluator_calls += 1
        hpwl = self._hpwl(positions)
        thermal = self._thermal(positions)[0]
        overlap = self._exact_overlap(positions)
        penalty = self._overlap_penalty(positions)[0]
        boundary = bool(np.all(positions >= -1e-10) and
                        np.all(positions + self.sizes <= 1 + 1e-10))
        return {
            "objective": hpwl / self.hpwl_scale + .3 * thermal / self.thermal_scale
                         + 20. * penalty,
            "hpwl_mm": hpwl, "thermal_proxy_peak": thermal,
            "overlap_fraction": overlap, "penetration_penalty": penalty,
            "valid": boundary and overlap < 1e-10,
        }

    def smooth_objective(self, positions, smoothing):
        delta = self._pin_delta(positions)
        hpwl = (np.sqrt(delta ** 2 + smoothing ** 2) * self.board_mm).sum()
        return (float(hpwl) / self.hpwl_scale
                + .3 * self._thermal(positions)[0] / self.thermal_scale
                + 20. * self._overlap_penalty(positions)[0])

    def gradient(self, positions, smoothing):
        self.gradient_calls += 1
        delta = self._pin_delta(positions)
        derivative = delta / np.sqrt(delta ** 2 + smoothing ** 2)
        derivative *= self.board_mm / self.hpwl_scale
        result = np.zeros_like(positions, dtype=float)
        np.add.at(result, self.pin_indices[:, 0], derivative)
        np.add.at(result, self.pin_indices[:, 1], -derivative)
        _, index, delta, contributions = self._thermal(positions)
        thermal_grad = contributions[index, :, None] * delta[index] / self.variances
        thermal_grad *= .3 / self.thermal_scale
        result[index] -= thermal_grad.sum(axis=0)
        result += thermal_grad
        _, delta, active = self._overlap_penalty(positions)
        overlap_grad = -np.sign(delta) * active[:, ::-1]
        overlap_grad *= (active > 0).all(axis=1)[:, None] * 20. / self.area_sum
        np.add.at(result, self.pair_i, overlap_grad)
        np.add.at(result, self.pair_j, -overlap_grad)
        return result

    def maps(self, positions, grid_size=256):
        result = np.zeros((3, grid_size, grid_size), dtype=float)
        result[0].fill(-1.)
        coordinates = (np.arange(grid_size) + .5) / grid_size
        xx, yy = np.meshgrid(coordinates, coordinates)
        for i in range(self.n):
            lo = np.floor(positions[i] * grid_size).astype(int)
            hi = np.ceil((positions[i] + self.sizes[i]) * grid_size).astype(int)
            lo = np.clip(lo, 0, grid_size - 1)
            hi = np.maximum(lo + 1, np.clip(hi, 0, grid_size))
            result[0, lo[1]:hi[1], lo[0]:hi[0]] = self.priority[i]
            centre = positions[i] + self.sizes[i] / 2
            result[1] += self.amplitudes[i] * np.exp(
                -.5 * ((xx - centre[0]) ** 2 / self.variances[i, 0]
                       + (yy - centre[1]) ** 2 / self.variances[i, 1]))
        pins = positions[self.pin_indices] + self.pin_offsets
        lo = np.floor(pins.min(axis=1) * grid_size).astype(int)
        hi = np.ceil(pins.max(axis=1) * grid_size).astype(int)
        lo = np.clip(lo, 0, grid_size - 1)
        hi = np.maximum(lo + 1, np.clip(hi, 0, grid_size))
        hpwl = (abs(pins[:, 0] - pins[:, 1]) * self.board_mm).sum(axis=1)
        density = hpwl / np.prod(hi - lo, axis=1)
        difference = np.zeros((grid_size + 1, grid_size + 1))
        for y, x, sign in ((lo[:, 1], lo[:, 0], 1), (hi[:, 1], lo[:, 0], -1),
                           (lo[:, 1], hi[:, 0], -1), (hi[:, 1], hi[:, 0], 1)):
            np.add.at(difference, (y, x), sign * density)
        result[2] = difference.cumsum(0).cumsum(1)[:-1, :-1]
        result[2] = np.maximum(result[2], 0)
        return result.astype(np.float32)
