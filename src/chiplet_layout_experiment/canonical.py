from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


def _finite(value: float, field: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class Board:
    width: float
    height: float

    def __post_init__(self) -> None:
        width = _finite(self.width, "board.width")
        height = _finite(self.height, "board.height")
        if width <= 0 or height <= 0:
            raise ValueError("board dimensions must be positive")


@dataclass(frozen=True, slots=True)
class Chiplet:
    name: str
    width: float
    height: float
    power: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("chiplet.name must not be empty")
        width = _finite(self.width, "chiplet.width")
        height = _finite(self.height, "chiplet.height")
        power = _finite(self.power, "chiplet.power")
        if width <= 0 or height <= 0:
            raise ValueError("chiplet dimensions must be positive")
        if power < 0:
            raise ValueError("chiplet.power must be non-negative")


@dataclass(frozen=True, slots=True)
class Placement:
    x: float
    y: float

    def __post_init__(self) -> None:
        _finite(self.x, "placement.x")
        _finite(self.y, "placement.y")


@dataclass(frozen=True, slots=True)
class PinRef:
    chiplet: str
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        if not self.chiplet or not self.chiplet.strip():
            raise ValueError("pin chiplet must not be empty")
        _finite(self.offset_x, "pin.offset_x")
        _finite(self.offset_y, "pin.offset_y")


@dataclass(frozen=True, slots=True)
class Net:
    name: str
    pins: tuple[PinRef, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("net.name must not be empty")
        pins = tuple(self.pins)
        weight = _finite(self.weight, "net.weight")
        if not pins:
            raise ValueError("net must contain at least one pin")
        if weight < 0:
            raise ValueError("net.weight must be non-negative")
        object.__setattr__(self, "pins", pins)


@dataclass(frozen=True, slots=True)
class Layout:
    board: Board
    chiplets: tuple[Chiplet, ...]
    placements: Mapping[str, Placement]
    nets: tuple[Net, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        chiplets = tuple(self.chiplets)
        nets = tuple(self.nets)
        placements = dict(self.placements)
        names = [chiplet.name for chiplet in chiplets]
        if len(set(names)) != len(names):
            raise ValueError("chiplet names must be unique")
        if set(placements) != set(names):
            raise ValueError("placements must match chiplet names exactly")
        if any(not isinstance(value, Placement) for value in placements.values()):
            raise TypeError("placements must contain Placement values")
        if len({net.name for net in nets}) != len(nets):
            raise ValueError("net names must be unique")
        known_names = set(names)
        unknown = sorted(
            {pin.chiplet for net in nets for pin in net.pins} - known_names
        )
        if unknown:
            raise ValueError(f"nets reference unknown chiplets: {unknown}")
        object.__setattr__(self, "chiplets", chiplets)
        object.__setattr__(self, "nets", nets)
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "board": {"width": self.board.width, "height": self.board.height},
            "chiplets": [
                {
                    "name": chiplet.name,
                    "width": chiplet.width,
                    "height": chiplet.height,
                    "power": chiplet.power,
                }
                for chiplet in self.chiplets
            ],
            "placements": {
                name: {"x": self.placements[name].x, "y": self.placements[name].y}
                for name in sorted(self.placements)
            },
            "nets": [
                {
                    "name": net.name,
                    "weight": net.weight,
                    "pins": [
                        {
                            "chiplet": pin.chiplet,
                            "offset_x": pin.offset_x,
                            "offset_y": pin.offset_y,
                        }
                        for pin in net.pins
                    ],
                }
                for net in self.nets
            ],
            "metadata": dict(self.metadata or {}),
        }

    def checksum(self) -> str:
        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
