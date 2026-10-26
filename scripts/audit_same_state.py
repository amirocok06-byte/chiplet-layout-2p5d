"""Independently replay saved coordinates and check labels/action coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chiplet_layout_experiment.counterfactual import array_hash, problem_hash
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_seeds = set(manifest["seeds"])
    count, states = 0, 0
    for path in sorted(root.glob("Case*/problem.json")):
        frozen = json.loads(path.read_text(encoding="utf-8"))
        p = TracedProblem(names=tuple(frozen["names"]),
            **{name: np.array(frozen[name]) for name in (
                "board_mm", "sizes", "powers", "pin_indices", "pin_offsets")},
            initial_positions=np.zeros((len(frozen["names"]), 2)))
        p.hpwl_scale, p.thermal_scale = frozen["hpwl_scale"], frozen["thermal_scale"]
        assert problem_hash(p) == frozen["problem_checksum"], path
        groups = {}
        for line in (path.parent / "labels.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            start = json.loads((path.parent / r["state_metadata"]["state_file"]).read_text(encoding="utf-8"))
            x = np.array(start["positions"])
            assert array_hash(x) == r["input_checksum"] == start["checksum"]
            expected_id = hashlib.sha256((problem_hash(p) + array_hash(x)).encode()).hexdigest()
            assert expected_id == r["state_id"]
            assert r["evaluator_calls"] == r["budget"] == manifest["budget"]
            assert r["postprocess_evaluator_calls"] == 1
            baseline = p.evaluate(x)
            for label, positions in (("input_metrics", x),
                    ("raw_metrics", r["raw_output_positions"]), ("metrics", r["output_positions"])):
                evaluated = p.evaluate(np.array(positions))
                for key, value in evaluated.items():
                    assert np.isclose(value, r[label][key], rtol=1e-12, atol=1e-12), (path, key)
            assert array_hash(r["output_positions"]) == r["output_checksum"]
            assert (r["phase"] == "legal_optimization") == bool(baseline["valid"])
            for key, delta in r["delta_metrics"].items():
                assert np.isclose(delta, r["metrics"][key]-baseline[key])
                assert np.isclose(delta, r["raw_delta_metrics"][key] + r["legalization_delta_metrics"][key])
            groups.setdefault(r["state_id"], []).append(r)
            count += 1
        for group in groups.values():
            assert len(group) == 6 * len(expected_seeds)
            assert len({(r["arm_id"], r["seed"]) for r in group}) == len(group)
            assert {r["seed"] for r in group} == expected_seeds
        states += len(groups)
    assert count == manifest["record_count"]
    return {"pass": True, "coordinate_replayed_records": count, "states": states,
        "checks": ["geometry/problem identity", "exact initial layout", "full action/seed grid",
                   "raw and returned metric replay", "delta decomposition", "evaluation budget",
                   "legal phase separation", "output checksum"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    (args.root / "replay_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
