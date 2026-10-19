from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Arm:
    arm_id: str
    worker_id: str
    config: Mapping[str, Any]
    budget: int

    def __post_init__(self) -> None:
        if not self.arm_id or not self.arm_id.strip():
            raise ValueError("arm_id must not be empty")
        if not self.worker_id or not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget <= 0:
            raise ValueError("budget must be a positive integer")
        object.__setattr__(self, "config", dict(self.config))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            dict(self.config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def default_arm_catalog() -> tuple[Arm, ...]:
    return (
        Arm("sa-balanced", "sa", {"temperature": "medium", "weight": "balanced"}, 4),
        Arm("sa-wire", "sa", {"temperature": "medium", "weight": "wire"}, 4),
        Arm("analytical-wl", "analytical", {"objective": "wirelength"}, 1),
        Arm("analytical-thermal", "analytical", {"objective": "thermal"}, 1),
        Arm("rl-fixed-small", "rl-compatible", {"policy": "fixed", "radius": "small"}, 1),
        Arm("rl-fixed-large", "rl-compatible", {"policy": "fixed", "radius": "large"}, 1),
    )
