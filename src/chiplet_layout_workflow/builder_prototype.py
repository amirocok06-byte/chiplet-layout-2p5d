"""Dependency-free deterministic chiplet-placement baseline.

This module deliberately implements only a small, independent 2.5D baseline.
The supported input is a strict subset of the common Bookshelf text format:
``.blocks``, ``.nets``, and ``.pl`` use their UCLA headers; ``.power`` uses a
UCLA header followed by exactly ``<chiplet> <power>`` records.  Unsupported or
ambiguous records fail with a :class:`BookshelfParseError` instead of being
interpreted heuristically.

The TAP, ATPlace, MaskPlace, and RLPlanner entries below are descriptive
metadata only.  No external source tree is imported or executed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class BookshelfParseError(ValueError):
    """Raised when a Bookshelf file is absent, malformed, or unsupported."""

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        line_number: int | None = None,
        code: str = "invalid_bookshelf",
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.code = code
        self.diagnostic = Diagnostic(
            level="error",
            code=code,
            message=message,
            path=str(path) if path is not None else None,
            line_number=line_number,
        )
        super().__init__(self.diagnostic.format())


@dataclass(frozen=True)
class Diagnostic:
    level: str
    code: str
    message: str
    path: str | None = None
    line_number: int | None = None

    def format(self) -> str:
        location = self.path or "Bookshelf input"
        if self.line_number is not None:
            location = f"{location}:line {self.line_number}"
        return f"{location}: {self.message} ({self.code})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class Placement:
    x: float
    y: float
    orientation: str | None = None


@dataclass(frozen=True)
class RectangleChiplet:
    """Axis-aligned rectangular chiplet, with its lower-left placement."""

    name: str
    width: float
    height: float
    x: float = 0.0
    y: float = 0.0
    terminal: bool = False
    power: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("chiplet name must not be empty")
        for label, value in (
            ("width", self.width),
            ("height", self.height),
            ("x", self.x),
            ("y", self.y),
        ):
            if not math.isfinite(value):
                raise ValueError(f"chiplet {label} must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("chiplet width and height must be positive")
        if self.power is not None and (
            not math.isfinite(self.power) or self.power < 0
        ):
            raise ValueError("chiplet power must be finite and non-negative")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height

    def with_position(self, *, x: float, y: float) -> "RectangleChiplet":
        return RectangleChiplet(
            name=self.name,
            width=self.width,
            height=self.height,
            x=x,
            y=y,
            terminal=self.terminal,
            power=self.power,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "x": self.x,
            "y": self.y,
            "right": self.right,
            "top": self.top,
            "terminal": self.terminal,
            "power": self.power,
        }


@dataclass(frozen=True)
class Net:
    name: str
    pins: tuple[str, ...]
    pin_directions: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("net name must not be empty")
        if not self.pins:
            raise ValueError(f"net {self.name!r} must have at least one pin")
        if self.pin_directions and len(self.pin_directions) != len(self.pins):
            raise ValueError("pin_directions must match pins length")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pins": list(self.pins),
            "pin_directions": list(self.pin_directions),
        }


@dataclass(frozen=True)
class BookshelfDesign:
    chiplets: tuple[RectangleChiplet, ...]
    nets: tuple[Net, ...]
    placements: Mapping[str, Placement]
    power: Mapping[str, float] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "chiplets": [chiplet.as_dict() for chiplet in self.chiplets],
            "nets": [net.as_dict() for net in self.nets],
            "placements": {
                name: {
                    "x": placement.x,
                    "y": placement.y,
                    "orientation": placement.orientation,
                }
                for name, placement in self.placements.items()
            },
            "power": dict(self.power),
            "diagnostics": [diagnostic.as_dict() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class PowerThermalInput:
    """Optional power/thermal data boundary; it performs no thermal solve."""

    power_by_chiplet: Mapping[str, float] = field(default_factory=dict)
    ambient_temperature: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, power in self.power_by_chiplet.items():
            if not name or not math.isfinite(power) or power < 0:
                raise ValueError("power_by_chiplet values must be named, finite, and non-negative")
        if self.ambient_temperature is not None and not math.isfinite(
            self.ambient_temperature
        ):
            raise ValueError("ambient_temperature must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "power_by_chiplet": dict(self.power_by_chiplet),
            "ambient_temperature": self.ambient_temperature,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    violations: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "violations": list(self.violations)}


@runtime_checkable
class ShapeInit(Protocol):
    """Future shape-initialization seam; no implementation is supplied here."""

    def initialize(self, design: BookshelfDesign) -> Any:
        ...


@runtime_checkable
class ShapeUpdate(Protocol):
    """Future shape-update seam; no implementation is supplied here."""

    def update(self, state: Any, step: int) -> Any:
        ...


@runtime_checkable
class ShapeObjective(Protocol):
    """Future shape-objective seam; no implementation is supplied here."""

    def evaluate(self, state: Any) -> float:
        ...


EXTERNAL_ADAPTERS: dict[str, dict[str, Any]] = {
    "TAP": {
        "name": "TAP",
        "kind": "external-reference",
        "status": "metadata-only",
        "executable": False,
        "description": "Reference only; no TAP source is imported or executed.",
    },
    "ATPlace": {
        "name": "ATPlace",
        "kind": "external-reference",
        "status": "metadata-only",
        "executable": False,
        "description": "Reference only; no ATPlace source is imported or executed.",
    },
    "MaskPlace": {
        "name": "MaskPlace",
        "kind": "external-reference",
        "status": "metadata-only",
        "executable": False,
        "description": "Reference only; no MaskPlace source is imported or executed.",
    },
    "RLPlanner": {
        "name": "RLPlanner",
        "kind": "external-reference",
        "status": "metadata-only",
        "executable": False,
        "description": "Reference only; no RLPlanner source is imported or executed.",
    },
}
REFERENCE_ADAPTERS = EXTERNAL_ADAPTERS


_HEADER_RE = re.compile(r"^UCLA\s+(?P<kind>[A-Za-z]+)\s+(?P<version>\S+)$")
_DECL_RE = re.compile(r"^(?P<key>NumNodes|NumTerminals|NumNets|NumPins)\s*:\s*(?P<value>\d+)$")
_HARDRECT_DECL_RE = re.compile(
    r"^(?P<key>NumSoftRectangularBlocks|NumHardRectilinearBlocks|NumTerminals)\s*:\s*(?P<value>\d+)$"
)
_NET_RE = re.compile(
    r"^NetDegree\s*:\s*(?P<degree>\d+)(?:\s+(?P<name>\S+))?$"
)
_COORDINATE_RE = re.compile(
    r"\(\s*(?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"(?P<y>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BookshelfParseError(
            f"cannot read input: {exc}", path=path, code="unreadable_input"
        ) from exc
    lines: list[tuple[int, str]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        content = raw.split("#", 1)[0].strip()
        if content:
            lines.append((line_number, content))
    return lines


def _fail(path: Path, line_number: int | None, message: str, code: str) -> None:
    raise BookshelfParseError(message, path=path, line_number=line_number, code=code)


def _require_header(lines: list[tuple[int, str]], path: Path, kind: str) -> list[tuple[int, str]]:
    if not lines:
        _fail(path, None, f"missing UCLA {kind} header", "missing_header")
    line_number, header = lines[0]
    match = _HEADER_RE.fullmatch(header)
    if match is None or match.group("kind").lower() != kind:
        _fail(
            path,
            line_number,
            f"expected UCLA {kind} header, got {header!r}",
            "unsupported_header",
        )
    return lines[1:]


def _declarations(
    lines: list[tuple[int, str]], path: Path, required: set[str]
) -> tuple[dict[str, int], list[tuple[int, str]]]:
    values: dict[str, int] = {}
    remaining: list[tuple[int, str]] = []
    for line_number, content in lines:
        match = _DECL_RE.fullmatch(content)
        if match is None:
            remaining.append((line_number, content))
            continue
        key = match.group("key")
        if key in values:
            _fail(path, line_number, f"duplicate declaration {key}", "duplicate_declaration")
        values[key] = int(match.group("value"))
    missing = sorted(required - values.keys())
    if missing:
        _fail(
            path,
            None,
            "missing required declaration(s): " + ", ".join(missing),
            "missing_declaration",
        )
    return values, remaining


def _parse_hard_rectilinear_blocks(
    lines: list[tuple[int, str]], path: Path
) -> tuple[tuple[RectangleChiplet, ...], dict[str, int]]:
    declarations: dict[str, int] = {}
    records: list[tuple[int, str]] = []
    for line_number, content in lines:
        match = _HARDRECT_DECL_RE.fullmatch(content)
        if match is None:
            records.append((line_number, content))
            continue
        key = match.group("key")
        if key in declarations:
            _fail(path, line_number, f"duplicate declaration {key}", "duplicate_declaration")
        declarations[key] = int(match.group("value"))
    required = {"NumSoftRectangularBlocks", "NumHardRectilinearBlocks", "NumTerminals"}
    missing = sorted(required - declarations.keys())
    if missing:
        _fail(path, None, "missing required declaration(s): " + ", ".join(missing), "missing_declaration")
    if declarations["NumSoftRectangularBlocks"]:
        _fail(
            path,
            None,
            "soft rectangular blocks are not supported by this smoke baseline",
            "unsupported_soft_rectangles",
        )

    chiplets: list[RectangleChiplet] = []
    names: set[str] = set()
    for line_number, content in records:
        fields = content.split(maxsplit=3)
        if len(fields) != 4 or fields[1].lower() != "hardrectilinear":
            _fail(
                path,
                line_number,
                "hard-rectilinear record must be '<name> hardrectilinear <vertex-count> <vertices>'",
                "unsupported_hardrect_record",
            )
        name = fields[0]
        if name in names:
            _fail(path, line_number, f"duplicate block name {name!r}", "duplicate_block")
        try:
            vertex_count = int(fields[2])
        except ValueError:
            _fail(path, line_number, "hard-rectilinear vertex count must be an integer", "invalid_vertex_count")
        points = [
            (float(match.group("x")), float(match.group("y")))
            for match in _COORDINATE_RE.finditer(fields[3])
        ]
        if vertex_count != len(points) or vertex_count != 4:
            _fail(
                path,
                line_number,
                "only four-vertex hard-rectilinear rectangles are supported",
                "unsupported_hardrect_shape",
            )
        xs = {point[0] for point in points}
        ys = {point[1] for point in points}
        expected = {
            (min(xs), min(ys)),
            (min(xs), max(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
        }
        if len(xs) != 2 or len(ys) != 2 or set(points) != expected:
            _fail(
                path,
                line_number,
                "hard-rectilinear shape must be an axis-aligned rectangle",
                "unsupported_hardrect_shape",
            )
        chiplets.append(
            RectangleChiplet(
                name=name,
                width=max(xs) - min(xs),
                height=max(ys) - min(ys),
            )
        )
        names.add(name)
    if len(chiplets) != declarations["NumHardRectilinearBlocks"]:
        _fail(
            path,
            None,
            f"NumHardRectilinearBlocks declares {declarations['NumHardRectilinearBlocks']} records but found {len(chiplets)}",
            "count_mismatch",
        )
    if declarations["NumTerminals"]:
        _fail(path, None, "hard-rectilinear terminals are not supported by this smoke baseline", "unsupported_terminals")
    return tuple(chiplets), declarations


def _parse_blocks(path: Path) -> tuple[tuple[RectangleChiplet, ...], dict[str, int]]:
    lines = _logical_lines(path)
    if lines and lines[0][1].startswith("NumSoftRectangularBlocks"):
        return _parse_hard_rectilinear_blocks(lines, path)
    remaining = _require_header(lines, path, "blocks")
    declarations, records = _declarations(remaining, path, {"NumNodes", "NumTerminals"})
    chiplets: list[RectangleChiplet] = []
    names: set[str] = set()
    for line_number, content in records:
        fields = content.split()
        if len(fields) not in (3, 4):
            _fail(
                path,
                line_number,
                "block record must be '<name> <width> <height>' with an optional terminal marker",
                "unsupported_blocks_record",
            )
        name = fields[0]
        if name in names:
            _fail(path, line_number, f"duplicate block name {name!r}", "duplicate_block")
        try:
            width = float(fields[1])
            height = float(fields[2])
            if not math.isfinite(width) or not math.isfinite(height):
                raise ValueError
        except ValueError:
            _fail(path, line_number, "block width and height must be finite numbers", "invalid_dimensions")
        marker = fields[3].lower() if len(fields) == 4 else ""
        if marker and marker not in {"terminal", "terminal_ni"}:
            _fail(path, line_number, f"unsupported block marker {fields[3]!r}", "unsupported_blocks_record")
        try:
            chiplet = RectangleChiplet(
                name=name,
                width=width,
                height=height,
                terminal=bool(marker),
            )
        except ValueError as exc:
            _fail(path, line_number, str(exc), "invalid_block")
        chiplets.append(chiplet)
        names.add(name)
    if len(chiplets) != declarations["NumNodes"]:
        _fail(
            path,
            None,
            f"NumNodes declares {declarations['NumNodes']} records but found {len(chiplets)}",
            "count_mismatch",
        )
    terminal_count = sum(chiplet.terminal for chiplet in chiplets)
    if terminal_count != declarations["NumTerminals"]:
        _fail(
            path,
            None,
            f"NumTerminals declares {declarations['NumTerminals']} but found {terminal_count} explicit terminal markers",
            "count_mismatch",
        )
    return tuple(chiplets), declarations


def _parse_nets(path: Path) -> tuple[tuple[Net, ...], dict[str, int]]:
    lines = _logical_lines(path)
    remaining = (
        lines
        if lines and lines[0][1].startswith("NumNets")
        else _require_header(lines, path, "nets")
    )
    declarations, records = _declarations(remaining, path, {"NumNets", "NumPins"})
    nets: list[Net] = []
    index = 0
    cursor = 0
    while cursor < len(records):
        line_number, content = records[cursor]
        match = _NET_RE.fullmatch(content)
        if match is None:
            _fail(path, line_number, "expected NetDegree declaration", "unsupported_nets_record")
        degree = int(match.group("degree"))
        if degree <= 0:
            _fail(path, line_number, "NetDegree must be positive", "invalid_net_degree")
        name = match.group("name") or f"net_{index}"
        pins: list[str] = []
        directions: list[str | None] = []
        cursor += 1
        for _ in range(degree):
            if cursor >= len(records):
                _fail(path, line_number, f"net {name!r} is missing pin records", "missing_net_pin")
            pin_line, pin_content = records[cursor]
            fields = pin_content.split()
            if len(fields) == 1:
                direction = None
            elif len(fields) == 2:
                direction = fields[1]
            elif len(fields) == 5 and fields[1] in {"B", "I", "O"} and fields[2] == ":":
                direction = fields[1]
            else:
                _fail(path, pin_line, "pin record must be a standard node/direction record or a hard-rectilinear pin record", "unsupported_pin_record")
            pins.append(fields[0])
            directions.append(direction)
            cursor += 1
        if any(existing.name == name for existing in nets):
            _fail(path, line_number, f"duplicate net name {name!r}", "duplicate_net")
        nets.append(Net(name=name, pins=tuple(pins), pin_directions=tuple(directions)))
        index += 1
    if len(nets) != declarations["NumNets"]:
        _fail(
            path,
            None,
            f"NumNets declares {declarations['NumNets']} records but found {len(nets)}",
            "count_mismatch",
        )
    actual_pins = sum(len(net.pins) for net in nets)
    if actual_pins != declarations["NumPins"]:
        _fail(
            path,
            None,
            f"NumPins declares {declarations['NumPins']} records but found {actual_pins}",
            "count_mismatch",
        )
    return tuple(nets), declarations


def _parse_placements(path: Path) -> dict[str, Placement]:
    lines = _logical_lines(path)
    records = (
        lines
        if lines and not lines[0][1].startswith("UCLA ")
        else _require_header(lines, path, "pl")
    )
    placements: dict[str, Placement] = {}
    for line_number, content in records:
        fields = content.split()
        if len(fields) == 3:
            name, x_text, y_text = fields
            orientation = None
        elif len(fields) == 5 and fields[3] == ":":
            name, x_text, y_text, _, orientation = fields
        else:
            _fail(
                path,
                line_number,
                "placement record must be '<name> <x> <y>' or '<name> <x> <y> : <orientation>'",
                "unsupported_pl_record",
            )
        if name in placements:
            _fail(path, line_number, f"duplicate placement for {name!r}", "duplicate_placement")
        try:
            x = float(x_text)
            y = float(y_text)
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError
        except ValueError:
            _fail(path, line_number, "placement coordinates must be finite numbers", "invalid_placement")
        placements[name] = Placement(x=x, y=y, orientation=orientation)
    return placements


def _parse_power(path: Path, known_names: set[str]) -> dict[str, float]:
    lines = _logical_lines(path)
    records = (
        lines
        if lines and not lines[0][1].startswith("UCLA ")
        else _require_header(lines, path, "power")
    )
    power: dict[str, float] = {}
    for line_number, content in records:
        fields = content.split()
        if len(fields) != 2:
            _fail(
                path,
                line_number,
                "power record must be exactly '<chiplet> <power>'",
                "unsupported_power_record",
            )
        name, value_text = fields
        if name not in known_names:
            _fail(path, line_number, f"power record references unknown chiplet {name!r}", "unknown_chiplet")
        if name in power:
            _fail(path, line_number, f"duplicate power record for {name!r}", "duplicate_power")
        try:
            value = float(value_text)
            if not math.isfinite(value) or value < 0:
                raise ValueError
        except ValueError:
            _fail(path, line_number, "power must be a finite non-negative number", "invalid_power")
        power[name] = value
    return power


def _resolve_input_paths(
    source: str | Path,
    nets_path: str | Path | None,
    pl_path: str | Path | None,
    power_path: str | Path | None,
) -> tuple[Path, Path, Path, Path | None]:
    source_path = Path(source)
    if source_path.is_dir():
        def one(suffix: str, *, optional: bool = False) -> Path | None:
            candidates = sorted(source_path.glob(f"*{suffix}"))
            if not candidates and optional:
                return None
            if len(candidates) != 1:
                expected = "optional single" if optional else "single"
                raise BookshelfParseError(
                    f"expected {expected} {suffix} file, found {len(candidates)}",
                    path=source_path,
                    code="ambiguous_input",
                )
            return candidates[0]

        blocks = one(".blocks")
        nets = Path(nets_path) if nets_path is not None else one(".nets")
        placements = Path(pl_path) if pl_path is not None else one(".pl")
        power = Path(power_path) if power_path is not None else one(".power", optional=True)
        assert nets is not None and placements is not None
        return blocks, nets, placements, power
    if source_path.suffix != ".blocks":
        raise BookshelfParseError(
            "source must be a Bookshelf directory or a .blocks file",
            path=source_path,
            code="invalid_source",
        )
    stem = source_path.with_suffix("")
    return (
        source_path,
        Path(nets_path) if nets_path is not None else stem.with_suffix(".nets"),
        Path(pl_path) if pl_path is not None else stem.with_suffix(".pl"),
        Path(power_path) if power_path is not None else (
            stem.with_suffix(".power") if stem.with_suffix(".power").is_file() else None
        ),
    )


def parse_bookshelf(
    source: str | Path,
    nets_path: str | Path | None = None,
    pl_path: str | Path | None = None,
    power_path: str | Path | None = None,
) -> BookshelfDesign:
    """Parse one Bookshelf case directory or its ``.blocks`` path."""

    blocks_file, nets_file, pl_file, power_file = _resolve_input_paths(
        source, nets_path, pl_path, power_path
    )
    chiplets, _ = _parse_blocks(blocks_file)
    nets, _ = _parse_nets(nets_file)
    placements = _parse_placements(pl_file)
    names = {chiplet.name for chiplet in chiplets}
    missing_placements = sorted(names - placements.keys())
    extra_placements = sorted(placements.keys() - names)
    if missing_placements or extra_placements:
        raise BookshelfParseError(
            "placement names must match block names exactly; "
            f"missing={missing_placements}, extra={extra_placements}",
            path=pl_file,
            code="placement_name_mismatch",
        )
    unknown_pins = sorted({pin for net in nets for pin in net.pins} - names)
    if unknown_pins:
        raise BookshelfParseError(
            f"nets reference unknown chiplet(s): {unknown_pins}",
            path=nets_file,
            code="unknown_chiplet",
        )
    power = _parse_power(power_file, names) if power_file is not None else {}
    return BookshelfDesign(
        chiplets=chiplets,
        nets=nets,
        placements=placements,
        power=power,
    )


def deterministic_sequential_place(
    chiplets: Sequence[RectangleChiplet],
    *,
    package_width: float,
    package_height: float,
    spacing: float = 0.0,
) -> tuple[RectangleChiplet, ...]:
    """Place chiplets in input order, left-to-right and then row-by-row."""

    if not math.isfinite(package_width) or not math.isfinite(package_height):
        raise ValueError("package dimensions must be finite")
    if package_width <= 0 or package_height <= 0:
        raise ValueError("package dimensions must be positive")
    if not math.isfinite(spacing) or spacing < 0:
        raise ValueError("spacing must be finite and non-negative")
    placed: list[RectangleChiplet] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for chiplet in chiplets:
        if chiplet.width > package_width:
            raise ValueError(f"chiplet {chiplet.name!r} is wider than package boundary")
        if cursor_x and cursor_x + chiplet.width > package_width:
            cursor_x = 0.0
            cursor_y += row_height + spacing
            row_height = 0.0
        candidate = chiplet.with_position(x=cursor_x, y=cursor_y)
        if candidate.top > package_height:
            raise ValueError(f"chiplet {chiplet.name!r} does not fit package boundary")
        placed.append(candidate)
        cursor_x += chiplet.width + spacing
        row_height = max(row_height, chiplet.height)
    return tuple(placed)


def _placement_map(
    placement: Mapping[str, Placement] | Sequence[RectangleChiplet],
) -> dict[str, Placement]:
    if isinstance(placement, Mapping):
        return {
            name: value if isinstance(value, Placement) else Placement(value.x, value.y)
            for name, value in placement.items()
        }
    return {chiplet.name: Placement(chiplet.x, chiplet.y) for chiplet in placement}


def compute_hpwl(
    nets: Sequence[Net],
    placement: Mapping[str, Placement] | Sequence[RectangleChiplet],
) -> float:
    """Return total half-perimeter wire length using chiplet lower-left points."""

    positions = _placement_map(placement)
    total = 0.0
    for net in nets:
        missing = sorted(set(net.pins) - positions.keys())
        if missing:
            raise ValueError(f"net {net.name!r} has no placement for {missing}")
        xs = [positions[pin].x for pin in net.pins]
        ys = [positions[pin].y for pin in net.pins]
        total += max(xs) - min(xs) + max(ys) - min(ys)
    return total


def check_package_boundary(
    chiplets: Sequence[RectangleChiplet],
    *,
    package_width: float,
    package_height: float,
    tolerance: float = 1e-9,
) -> ValidationReport:
    violations: list[str] = []
    for chiplet in chiplets:
        if chiplet.x < -tolerance or chiplet.y < -tolerance:
            violations.append(f"{chiplet.name}: lower-left corner is outside package")
        if chiplet.right > package_width + tolerance or chiplet.top > package_height + tolerance:
            violations.append(f"{chiplet.name}: rectangle exceeds package boundary")
    return ValidationReport(ok=not violations, violations=tuple(violations))


def check_non_overlap(
    chiplets: Sequence[RectangleChiplet], *, tolerance: float = 1e-9
) -> ValidationReport:
    violations: list[str] = []
    for index, left in enumerate(chiplets):
        for right in chiplets[index + 1 :]:
            x_overlap = min(left.right, right.right) - max(left.x, right.x) > tolerance
            y_overlap = min(left.top, right.top) - max(left.y, right.y) > tolerance
            if x_overlap and y_overlap:
                violations.append(f"{left.name} overlaps {right.name}")
    return ValidationReport(ok=not violations, violations=tuple(violations))


def run_baseline(
    source: str | Path | BookshelfDesign,
    *,
    package_width: float,
    package_height: float,
    spacing: float = 0.0,
    thermal_input: PowerThermalInput | None = None,
) -> dict[str, Any]:
    """Run the deterministic baseline and return JSON-serializable results."""

    design = source if isinstance(source, BookshelfDesign) else parse_bookshelf(source)
    placed = deterministic_sequential_place(
        design.chiplets,
        package_width=package_width,
        package_height=package_height,
        spacing=spacing,
    )
    boundary = check_package_boundary(
        placed, package_width=package_width, package_height=package_height
    )
    overlap = check_non_overlap(placed)
    thermal = thermal_input or PowerThermalInput(power_by_chiplet=design.power)
    return {
        "placement": {chiplet.name: chiplet.as_dict() for chiplet in placed},
        "metrics": {
            "hpwl": compute_hpwl(design.nets, placed),
            "chiplet_count": len(placed),
            "net_count": len(design.nets),
        },
        "checks": {
            "package_boundary": boundary.ok,
            "non_overlap": overlap.ok,
            "boundary_violations": list(boundary.violations),
            "overlap_violations": list(overlap.violations),
        },
        "power": dict(design.power),
        "thermal_input": thermal.as_dict(),
        "diagnostics": [diagnostic.as_dict() for diagnostic in design.diagnostics],
        "adapters": [dict(metadata) for metadata in EXTERNAL_ADAPTERS.values()],
    }


__all__ = [
    "BookshelfDesign",
    "BookshelfParseError",
    "Diagnostic",
    "EXTERNAL_ADAPTERS",
    "Net",
    "Placement",
    "PowerThermalInput",
    "REFERENCE_ADAPTERS",
    "RectangleChiplet",
    "ShapeInit",
    "ShapeObjective",
    "ShapeUpdate",
    "ValidationReport",
    "check_non_overlap",
    "check_package_boundary",
    "compute_hpwl",
    "deterministic_sequential_place",
    "parse_bookshelf",
    "run_baseline",
]
