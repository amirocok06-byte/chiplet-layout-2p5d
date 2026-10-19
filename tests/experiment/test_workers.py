from chiplet_layout_experiment.canonical import Board, Chiplet, Layout, Net, PinRef, Placement
from chiplet_layout_experiment.evaluator import PhysicsEvaluator
from chiplet_layout_experiment.workers import (
    AnalyticalWorker,
    RLCompatibleWorker,
    SAWorker,
    WorkerResult,
)


def _layout():
    return Layout(
        board=Board(20.0, 20.0),
        chiplets=(Chiplet("a", 4.0, 2.0, power=10.0), Chiplet("b", 3.0, 3.0, power=5.0)),
        placements={"a": Placement(10.0, 10.0), "b": Placement(1.0, 1.0)},
        nets=(Net("n0", (PinRef("a"), PinRef("b"))),),
    )


def test_all_first_stage_workers_return_one_common_result_envelope():
    evaluator = PhysicsEvaluator()
    workers = [AnalyticalWorker(), SAWorker(), RLCompatibleWorker()]

    for worker in workers:
        result = worker.run(_layout(), evaluator=evaluator, seed=7, budget=4)
        assert isinstance(result, WorkerResult)
        assert result.status == "available"
        assert result.worker_id == worker.worker_id
        assert result.evaluation.evaluator_version == "physics-v1"
        assert result.candidate.checksum()
        assert result.budget == 4


def test_rl_compatible_worker_is_explicitly_local_and_not_a_reproduction():
    result = RLCompatibleWorker().run(_layout(), evaluator=PhysicsEvaluator(), seed=1, budget=1)

    assert result.metadata["implementation"] == "local-rule-baseline"
    assert result.metadata["external_reproduction"] is False
