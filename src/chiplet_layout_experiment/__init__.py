"""CPU-only first-stage experiment primitives for 2.5D chiplet placement."""

from .canonical import Board, Chiplet, Layout, Net, PinRef, Placement
from .evaluator import PhysicsEvaluation, PhysicsEvaluator
from .arms import Arm, default_arm_catalog
from .sampling import SampleRecord, sample_arms, write_jsonl, write_manifest
from .workers import AnalyticalWorker, RLCompatibleWorker, SAWorker, WorkerResult

__all__ = [
    "Board",
    "Chiplet",
    "Layout",
    "Net",
    "PinRef",
    "Placement",
    "PhysicsEvaluation",
    "PhysicsEvaluator",
    "Arm",
    "default_arm_catalog",
    "SampleRecord",
    "sample_arms",
    "write_jsonl",
    "write_manifest",
    "AnalyticalWorker",
    "RLCompatibleWorker",
    "SAWorker",
    "WorkerResult",
]
