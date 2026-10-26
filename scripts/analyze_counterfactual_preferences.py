"""Describe same-state preference identifiability and legalizer-only control."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chiplet_layout_experiment.legalization import legalize
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def analyze(root):
    groups = defaultdict(list)
    for file in root.glob("Case*/labels.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            groups[(file.parent, row["state_id"])].append(row)
    entries = []
    for (directory, state_id), rows in groups.items():
        row = rows[0]
        data = json.loads((directory / "problem.json").read_text(encoding="utf-8"))
        state = json.loads((directory / row["state_metadata"]["state_file"]).read_text(encoding="utf-8"))
        p = TracedProblem(names=tuple(data["names"]), initial_positions=np.array(state["positions"]),
            **{key: np.array(data[key]) for key in (
                "board_mm", "sizes", "powers", "pin_indices", "pin_offsets")})
        p.hpwl_scale, p.thermal_scale = data["hpwl_scale"], data["thermal_scale"]
        # No algorithm action: this is a separately billed control, not an arm.
        control = legalize(p, p.initial_positions)
        control_metrics = p.evaluate(control.positions)
        arms = defaultdict(list)
        raw_rewards = defaultdict(list)
        control_deltas = defaultdict(list)
        for r in rows:
            arm = r["arm_id"]
            arms[arm].append(r["bandit_reward"])
            raw = r["raw_metrics"]
            before = r["input_metrics"]
            if r["phase"] == "feasibility_recovery":
                raw_rewards[arm].append(float(raw["valid"]) + before["overlap_fraction"] - raw["overlap_fraction"])
            else:
                raw_rewards[arm].append(
                    -(raw["hpwl_mm"]-before["hpwl_mm"])/p.hpwl_scale
                    -.3*(raw["thermal_proxy_peak"]-before["thermal_proxy_peak"])/p.thermal_scale
                    if raw["valid"] else -1.)
            if control_metrics["valid"] and r["metrics"]["valid"]:
                control_deltas[arm].append({
                    "delta_hpwl_mm": r["metrics"]["hpwl_mm"]-control_metrics["hpwl_mm"],
                    "delta_thermal_proxy": r["metrics"]["thermal_proxy_peak"]-control_metrics["thermal_proxy_peak"],
                })
        means = {arm: float(np.mean(values)) for arm, values in arms.items()}
        raw_means = {arm: float(np.mean(values)) for arm, values in raw_rewards.items()}
        spread = max(means.values())-min(means.values())
        entries.append({"case": row["state_metadata"]["case"], "state_id": state_id,
            "phase": row["phase"], "postprocessed_mean_reward_by_arm": means,
            "raw_mean_reward_by_arm": raw_means,
            "postprocessed_reward_spread": spread,
            "postprocessed_reward_identifiable_at_1e_10": spread > 1e-10,
            "raw_reward_spread": max(raw_means.values())-min(raw_means.values()),
            "legalizer_only_metrics": control_metrics,
            "legalizer_only_seconds": control.wall_seconds,
            "legalizer_only_geometry_checks": control.geometry_checks,
            "control_evaluator_calls": p.evaluator_calls,
            "valid_output_minus_legalizer_only_by_arm": {
                arm: {metric: float(np.mean([d[metric] for d in values]))
                      for metric in ("delta_hpwl_mm", "delta_thermal_proxy")}
                for arm, values in control_deltas.items()},
        })
    summary = []
    for phase in ("feasibility_recovery", "legal_optimization"):
        selected = [e for e in entries if e["phase"] == phase]
        summary.append({"phase": phase, "states": len(selected),
            "states_with_distinct_mean_action_rewards": sum(e["postprocessed_reward_identifiable_at_1e_10"] for e in selected),
            "states_with_distinct_raw_action_rewards": sum(e["raw_reward_spread"] > 1e-10 for e in selected)})
    output = {"format": "same-state-identifiability-v1", "descriptive_only": True,
        "summary": summary, "extra_control_evaluator_calls": sum(e["control_evaluator_calls"] for e in entries),
        "entries": entries,
        "interpretation": "A tied recovery reward after common legalization cannot supervise algorithm preference. Use raw recovery vectors, runtime objectives or legal-state optimization; keep protocol fixed in each comparison."}
    (root / "preference_diagnostics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "extra_control_evaluator_calls": output["extra_control_evaluator_calls"]}))


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
