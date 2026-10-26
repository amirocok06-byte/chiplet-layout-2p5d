"""Collect counterfactual labels from controlled intermediate layouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .counterfactual import (ALGORITHMS, array_hash, clone_at, finite_catalog,
                             problem_hash, sample_state, summarize_labels)
from .geometry_versions import apply_geometry_version
from .legalization import legalize
from .trajectory_data import load_case
from .state_projection import STATE_VERSION


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def source_hashes():
    directory = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(
        directory.glob("*.py"))}


def collect_case(problem, metadata, output, *, budget, seeds, warmup_steps=64,
                 source_algorithms=("sa", "analytical", "rl")):
    """Pool anchors across all algorithm families, plus legal companions.

    All anchors of one case belong in ONE train/validation/test split. The
    common legalizer is used to construct a second input distribution, never
    silently to replace a raw anchor. Deduplicate by exact position bytes.
    """
    if warmup_steps < 2 or warmup_steps % 2:
        raise ValueError("warmup_steps must be positive and even")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"source": metadata, "problem_checksum": problem_hash(problem)}
    write_json(output / "problem.json", {
        **provenance,
        **{key: np.asarray(getattr(problem, key)).tolist() for key in (
            "board_mm", "sizes", "powers", "pin_indices", "pin_offsets")},
        "names": list(problem.names), "hpwl_scale": problem.hpwl_scale,
        "thermal_scale": problem.thermal_scale,
    })
    states, origins, preparation = {}, {}, []

    def add_state(x, info):
        checksum = array_hash(x)
        states[checksum] = x.copy()
        origins.setdefault(checksum, []).append(info)

    for algorithm in source_algorithms:
        fork = clone_at(problem, problem.initial_positions)

        def capture(step, x, details):
            if step in (0, warmup_steps // 2, warmup_steps):
                add_state(x, {"kind": "raw_anchor", "algorithm": algorithm, "step": step})

        started = time.perf_counter()
        ALGORITHMS[algorithm](fork, steps=warmup_steps, seed=metadata["seed"], on_step=capture)
        preparation.append({"algorithm": algorithm, "calls": fork.evaluator_calls,
            "gradient_calls": fork.gradient_calls, "wall_seconds": time.perf_counter()-started})
    for checksum, x in list(states.items()):
        result = legalize(problem, x)
        info = {"kind": "legal_companion", "raw_checksum": checksum,
                "status": result.status, "method": result.method,
                "wall_seconds": result.wall_seconds, "geometry_checks": result.geometry_checks}
        preparation.append(info)
        if result.valid:
            add_state(np.asarray(result.positions), info)

    rows, state_index = [], []
    with (output / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for number, (checksum, x) in enumerate(states.items()):
            state_metadata = {
                "case": metadata["case"], "geometry_version": metadata["geometry_version"],
                "origins": origins[checksum], "state_file": f"states/{checksum}.json",
                "case_split_group": metadata["case"],
                "source_problem": "problem.json",
                "state_version": STATE_VERSION,
            }
            write_json(output / state_metadata["state_file"], {
                "positions": x.tolist(), "checksum": checksum, "metadata": state_metadata,
                "map_shape": [4, 256, 256],
                "state_version": STATE_VERSION,
                "map_reconstruction": "state_projection.project_state(problem, positions, grid_size=256)",
            })
            labels = sample_state(problem, x, finite_catalog(budget), seeds=seeds,
                                  state_metadata=state_metadata)
            for row in labels:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            rows.extend(labels)
            state_index.append({"checksum": checksum, "state_id": labels[0]["state_id"],
                "phase": labels[0]["phase"], "origins": origins[checksum]})
            print(json.dumps({"case": metadata["case"], "version": metadata["geometry_version"],
                "state": number + 1, "total_states": len(states),
                "phase": labels[0]["phase"]}), flush=True)
    write_json(output / "summary.json", summarize_labels(rows))
    write_json(output / "manifest.json", {
        **provenance, "states": state_index, "preparation_costs": preparation,
        "records": len(rows), "seeds": seeds, "evaluation_budget_per_action": budget,
        "source_algorithms": list(source_algorithms), "warmup_steps": warmup_steps,
        "state_version": STATE_VERSION, "map_shape": [4, 256, 256],
        "algorithm_budget_includes_initial_call": True,
        "postprocess_evaluations_per_action": 1,
        "thermal_label": "uncalibrated spatial proxy; HotSpot calibration separate",
        "equal_budget": "objective evaluations, not equal time or gradient computation",
        "status": "complete",
    })
    return rows


def write_overview(output, rows, manifest):
    summary = summarize_labels(rows)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        groups = summary["groups"]
        if groups:
            writer = csv.DictWriter(handle, fieldnames=list(groups[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(groups)
    state_groups = {}
    for row in rows:
        state_groups.setdefault(row["state_id"], []).append(row)
    paired = all(len({r["input_checksum"] for r in group}) == 1
                 and len({r["budget"] for r in group}) == 1
                 and len({r["problem_checksum"] for r in group}) == 1
                 for group in state_groups.values())
    expected = {(arm.arm_id, arm.config_hash, seed)
                for arm in finite_catalog(manifest["budget"]) for seed in manifest["seeds"]}
    complete = bool(state_groups) and all(
        len(group) == len(expected) and
        {(r["arm_id"], r["config_hash"], r["seed"]) for r in group} == expected
        for group in state_groups.values())
    write_json(output / "budget_audit.json", {
        "pass": paired and complete and all(
            r["evaluator_calls"] == r["budget"] == manifest["budget"] for r in rows),
        "states": len(state_groups), "records": len(rows),
        "algorithm_evaluations": sum(r["evaluator_calls"] for r in rows),
        "postprocess_evaluations": sum(r["postprocess_evaluator_calls"] for r in rows),
        "baseline_evaluations": len(state_groups),
        "expected_labels_per_state": 6 * len(manifest["seeds"]),
        "complete_action_grid": complete,
    })
    legal = [r for r in rows if r["phase"] == "legal_optimization"]
    recovery = [r for r in rows if r["phase"] == "feasibility_recovery"]
    lines = ["# 同状态动作收益：首轮采集结果", "",
        f"共 {len(state_groups)} 个不同布局，{len(rows)} 个 action×seed 标签；合法状态优化 {len(legal)} 条，可行性恢复 {len(recovery)} 条。",
        "", "每个 action 使用相同评价预算；初始评价计入预算，梯度、策略训练、公共合法化和额外验证分别计账。"
        "这是等评价次数实验，不能据此宣称等时间性能最优。RSS 是共享进程瞬时值，不是每臂峰值。",
        "", "每个 Case 包含三个算法产生的 0/中点/终点状态并去重，同时另采合法化成功的 companion。"
        "同一 Case 的全部状态和几何版本应进入同一数据划分，禁止随机逐行切分造成泄漏。",
        "", "`raw_delta_metrics` 是纯算法收益；`legalization_delta_metrics` 是后处理增量；"
        "`delta_metrics` 是完整 action+legalizer 收益。三者不得混用。非法输入的线长/热变化只作诊断，"
        "不会混入合法到合法的优化均值。恢复阶段 reward=合法性恢复指示+重叠下降；"
        "优化阶段 reward=归一化 HPWL 改善+0.3×归一化热代理改善。",
        "", "热标签当前为未校准空间代理；HotSpot 适配和真实校准状态见 [热评估说明](../../hotspot_calibration.md)。",
        "", "|Case/版本|阶段|配置|样本|原始合法率|后处理合法率|合法优化 ΔHPWL/mm|合法优化 Δ热代理|算法时间/s|合法化/s|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def num(v):
        return "—" if v is None else f"{v:.5g}"
    for g in summary["groups"]:
        lines.append("|" + "|".join([f'{g["case"]}/{g["geometry_version"]}', g["phase"],
            g["arm_id"], str(g["records"]), num(g["raw_final_legal_rate"]),
            num(g["final_legal_rate"]), num(g["mean_delta_hpwl_mm_legal_only"]),
            num(g["mean_delta_thermal_proxy_legal_only"]), num(g["mean_optimizer_seconds"]),
            num(g["mean_legalization_seconds"])]) + "|")
    lines += ["", "原始逐动作 JSONL、状态坐标、源输入哈希和各阶段调用数在各 Case 目录；"
        "汇总见 [summary.csv](summary.csv)、[summary.json](summary.json)，预算核验见 [budget_audit.json](budget_audit.json)。",
        "", "当前是有限配置全臂数据采集，尚未训练 CNN/Bandit。三个本地算法不等同于论文官方实现；"
        "配置差异是机制可识别性测试，不预设某算法适合某阶段，也不把公共合法化收益归功于某算法。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=[f"Case{i}" for i in range(1, 11)])
    parser.add_argument("--case4-version", default="case4-59mm-alternative")
    parser.add_argument("--budget", type=int, default=65)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261025, 20261026, 20261027])
    parser.add_argument("--source-seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory to avoid stale completed reports')
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "counterfactual-experiment-v1", "cases": args.cases,
        "seeds": args.seeds, "budget": args.budget, "warmup_steps": args.warmup_steps,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True).strip(),
        "source_sha256": source_hashes(), "source_seed": args.source_seed,
        "thread_limits": {n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    for case in args.cases:
        problem, metadata = load_case(args.data_root, case, seed=args.source_seed)
        if case == "Case4":
            write_json(args.output / "Case4-original" / "diagnostic.json", {
                "source": metadata, "geometry_version": "original",
                "status": "proven_infeasible_by_area", "excluded_from_bandit": True,
                "reason": "2064 mm2 chiplet area exceeds 1369 mm2 board area"})
        version = args.case4_version if case == "Case4" else "original"
        problem, metadata = apply_geometry_version(problem, metadata, version)
        destination = args.output / case
        if metadata["proven_infeasible_by_area"]:
            write_json(destination / "diagnostic.json", {"source": metadata,
                "status": "proven_infeasible_by_area", "excluded_from_bandit": True})
            continue
        rows.extend(collect_case(problem, metadata, destination, budget=args.budget,
                                seeds=args.seeds, warmup_steps=args.warmup_steps))
    manifest["record_count"] = len(rows)
    manifest["status"] = "complete"
    write_json(args.output / "manifest.json", manifest)
    write_overview(args.output, rows, manifest)


if __name__ == "__main__":
    main()
