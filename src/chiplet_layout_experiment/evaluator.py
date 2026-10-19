from __future__ import annotations

from dataclasses import dataclass
import math

from .canonical import Layout


@dataclass(frozen=True, slots=True)
class GeometryReport:
    boundary_violations: tuple[str, ...]
    overlap_violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.boundary_violations and not self.overlap_violations


@dataclass(frozen=True, slots=True)
class PhysicsEvaluation:
    valid: bool
    hpwl: float
    peak_temperature: float
    mean_temperature: float
    ambient_temperature: float
    geometry: GeometryReport
    evaluator_version: str
    evaluator_calls: int


class GeometryEvaluator:
    VERSION = "geometry-v1"

    def evaluate(self, layout: Layout, *, tolerance: float = 1e-9) -> GeometryReport:
        boundary: list[str] = []
        for chiplet in layout.chiplets:
            placement = layout.placements[chiplet.name]
            right = placement.x + chiplet.width
            top = placement.y + chiplet.height
            if placement.x < -tolerance or placement.y < -tolerance:
                boundary.append(f"{chiplet.name} outside board")
            if right > layout.board.width + tolerance or top > layout.board.height + tolerance:
                boundary.append(f"{chiplet.name} exceeds board")

        overlaps: list[str] = []
        for index, left in enumerate(layout.chiplets):
            left_position = layout.placements[left.name]
            for right in layout.chiplets[index + 1 :]:
                right_position = layout.placements[right.name]
                x_overlap = min(
                    left_position.x + left.width, right_position.x + right.width
                ) - max(left_position.x, right_position.x)
                y_overlap = min(
                    left_position.y + left.height, right_position.y + right.height
                ) - max(left_position.y, right_position.y)
                if x_overlap > tolerance and y_overlap > tolerance:
                    overlaps.append(f"{left.name} overlaps {right.name}")
        return GeometryReport(tuple(boundary), tuple(overlaps))


class HPWLEvaluator:
    VERSION = "hpwl-v1-pin-offset-weighted"

    def evaluate(self, layout: Layout) -> float:
        total = 0.0
        for net in layout.nets:
            coordinates = []
            for pin in net.pins:
                placement = layout.placements[pin.chiplet]
                coordinates.append((placement.x + pin.offset_x, placement.y + pin.offset_y))
            xs = [point[0] for point in coordinates]
            ys = [point[1] for point in coordinates]
            total += net.weight * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
        return total


class FixedThermalEvaluator:
    VERSION = "thermal-v1-power-density-proxy"

    def __init__(self, *, ambient_temperature: float = 25.0, scale: float = 10.0) -> None:
        if not math.isfinite(ambient_temperature) or not math.isfinite(scale):
            raise ValueError("thermal parameters must be finite")
        if scale < 0:
            raise ValueError("thermal scale must be non-negative")
        self.ambient_temperature = ambient_temperature
        self.scale = scale

    def evaluate(self, layout: Layout) -> tuple[float, float]:
        densities = [chiplet.power / (chiplet.width * chiplet.height) for chiplet in layout.chiplets]
        board_density = sum(chiplet.power for chiplet in layout.chiplets) / (
            layout.board.width * layout.board.height
        )
        peak = self.ambient_temperature + self.scale * (max(densities, default=0.0))
        mean = self.ambient_temperature + self.scale * board_density
        return peak, mean


class PhysicsEvaluator:
    VERSION = "physics-v1"

    def __init__(self, *, thermal: FixedThermalEvaluator | None = None) -> None:
        self.geometry = GeometryEvaluator()
        self.hpwl = HPWLEvaluator()
        self.thermal = thermal or FixedThermalEvaluator()

    def evaluate(self, layout: Layout) -> PhysicsEvaluation:
        geometry = self.geometry.evaluate(layout)
        peak, mean = self.thermal.evaluate(layout)
        return PhysicsEvaluation(
            valid=geometry.valid,
            hpwl=self.hpwl.evaluate(layout),
            peak_temperature=peak,
            mean_temperature=mean,
            ambient_temperature=self.thermal.ambient_temperature,
            geometry=geometry,
            evaluator_version=self.VERSION,
            evaluator_calls=3,
        )
