"""Render five checkpoints per algorithm plus complete parameter/metric histories.

The shared export_figure invokes require_matplotlib_panel_alignment() before
exporting PNG, .pdf and .svg; the optional installed auditors then inspect PDFs.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_export import export_figure
from plot_grid import CHANNELS, compute_map_limits
from plot_snapshot_maps import plot_snapshot_maps
from plot_state_flow import plot_state_flow
from plot_algorithm_effects import plot_algorithm_effects

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
METHODS = ("sa", "analytical", "rl")
LABELS = {"sa": "SA (Metropolis)", "analytical": "Analytical (gradient)",
          "rl": "RL (online REINFORCE)"}
COLORS = {"sa": "#0072b2", "analytical": "#d55e00", "rl": "#009e73"}
MAP_LABELS = ("Connectivity priority", "Spatial thermal proxy (uncalibrated)",
              "HPWL contribution (mm/pixel)")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_metrics(directory, summaries, *, dpi, parameters=True):
    histories = {}
    for algorithm in METHODS:
        data = gzip.decompress((directory / algorithm / "trace.jsonl.gz").read_bytes())
        histories[algorithm] = [json.loads(line) for line in data.splitlines()]
    keys = ("objective", "hpwl_mm", "thermal_proxy_peak", "overlap_fraction")
    names = ("Common objective", "HPWL (mm)", "Thermal proxy at chiplet centres",
             "Pair-overlap area / chiplet area")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for ax, key, name in zip(axes.ravel(), keys, names):
        for algorithm in METHODS:
            rows = histories[algorithm]
            ax.plot([r["step"] for r in rows], [r["metrics"][key] for r in rows],
                    color=COLORS[algorithm], label=LABELS[algorithm], linewidth=1.1)
        for step in summaries["sa"]["checkpoint_steps"]:
            ax.axvline(step, color="#777777", linewidth=.5, alpha=.3)
        ax.set(title=name, xlabel="Candidate iteration", ylabel=name)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{directory.name} | full trajectories, one seed", fontsize=12)
    export_figure(fig, directory / "metric_trends.png", dpi=dpi, axes=axes.ravel())
    plt.close(fig)
    if not parameters:
        return
    fields = {
        "sa": ("temperature", "proposal_radius", "best_objective"),
        "analytical": ("learning_rate", "smoothing", "gradient_scale"),
        "rl": ("entropy", "policy_weight_norm", "displacement"),
    }
    for algorithm in METHODS:
        rows = histories[algorithm]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
        for ax, field in zip(axes, fields[algorithm]):
            ax.plot([r["step"] for r in rows],
                    [r["parameters"].get(field, np.nan) for r in rows],
                    color=COLORS[algorithm], linewidth=1.2)
            ax.set_title(field.replace("_", " "), fontsize=10)
            ax.set_xlabel("Candidate iteration", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.15)
        fig.suptitle(f"{directory.name} | {LABELS[algorithm]} | parameter history", fontsize=11)
        export_figure(fig, directory / algorithm / "parameter_trends.png", dpi=dpi, axes=axes)
        plt.close(fig)


def render_case(root, case, *, dpi, refresh_geometry_reports=False):
    directory = root / case
    summaries = {a: load_json(directory / a / "summary.json") for a in METHODS}
    arrays = {}
    for a in METHODS:
        with np.load(directory / a / "raw" / "state_flow.npz", allow_pickle=False) as data:
            arrays[a] = data["maps"]
    limits = [compute_map_limits(np.stack([arrays[a][:, c] for a in METHODS]), channel)
              for c, channel in enumerate(CHANNELS)]
    (directory / "display_scales.json").write_text(
        json.dumps({"channel_labels": MAP_LABELS, "limits": limits,
                    "normalization": ["16-level sequential priority", "PowerNorm gamma=0.6",
                                      "SymLogNorm linear-region=1% fixed maximum"],
                    "hpwl_display_zero_tolerance": "64*float64_epsilon*shared_abs_max",
                    "shared_across": "all three algorithms and all five frames"},
                   indent=2) + "\n", encoding="utf-8")
    # Dimensions come from the frozen source, not from the displayed images.
    names = summaries["sa"]["source"]["num_chiplets"]
    # Recover exact sizes from common data adapter, keeping original power/geometry.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from chiplet_layout_experiment.trajectory_data import load_case
    source_dir = Path(summaries["sa"]["source"]["source_directory"])
    problem, _ = load_case(source_dir.parent, case, seed=summaries["sa"]["seed"])
    area = float(problem.area_sum)
    warning = " | INFEASIBLE BOARD: chiplet area > board" if area > 1 else ""
    for a in METHODS:
        summary = summaries[a]
        boxes = [np.column_stack((c["positions"], problem.sizes))
                 for c in summary["checkpoints"]]
        for index, checkpoint in enumerate(summary["checkpoints"]):
            step = checkpoint["step"]
            metrics = checkpoint["metrics"]
            title = (f"{case} | {LABELS[a]} | iteration {step}/{summary['steps']} | "
                     f"overlap={metrics['overlap_fraction']:.5f} | "
                     f"valid={metrics['valid']}{warning}")
            if refresh_geometry_reports:
                import pymupdf
                existing = directory / a / f"checkpoint_{step:04d}.pdf"
                with pymupdf.open(existing) as pdf:
                    text = " ".join(pdf[0].get_text().split())
                if " ".join(title.split()) in text:
                    continue
            plot_snapshot_maps(
                directory / a / "raw" / f"checkpoint_{step:04d}.npz",
                directory / a / f"checkpoint_{step:04d}.png",
                dpi=dpi, title=title, channel_labels=MAP_LABELS,
                limits=limits, chiplet_boxes=boxes[index])
        if not refresh_geometry_reports:
            plot_state_flow(
            directory / a / "raw" / "state_flow.npz",
            directory / a / "state_flow.png", max_frames=5, dpi=dpi,
            frame_labels=[f"iter {step}" for step in summary["checkpoint_steps"]],
            title=f"{case} | {LABELS[a]} | five continuous checkpoints{warning}",
                channel_labels=MAP_LABELS, limits=limits, chiplet_boxes=boxes)
        print(f"rendered {case}/{a}: five snapshots + state flow", flush=True)
    effects = directory / "sa" / "raw" / "case_effects.npz"
    np.savez_compressed(effects, before_maps=arrays["sa"][0],
                        after_maps=np.stack([arrays[a][-1] for a in METHODS]))
    if not refresh_geometry_reports:
        plot_algorithm_effects(effects, directory / "algorithm_effects.png",
                               [LABELS[a] for a in METHODS], dpi=dpi)
    render_metrics(directory, summaries, dpi=dpi, parameters=not refresh_geometry_reports)
    result = {"case": case, "chiplets": names, "area_utilization": area,
              "proven_infeasible_by_area": area > 1,
              "final_metrics": {a: summaries[a]["checkpoints"][-1]["metrics"] for a in METHODS}}
    (directory / "case_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_gallery(root, cases, results):
    rows = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
            "<title>2.5D 五检查点轨迹</title><style>",
            "body{font-family:system-ui;max-width:1600px;margin:32px auto;background:#fafafa;color:#202830}",
            "img{width:100%;height:auto}section{padding:20px;background:white;margin:22px 0;border:1px solid #ddd}",
            "details{margin:16px 0}summary{cursor:pointer;font-size:19px}a{color:#166b9b}",
            "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}",
            ".note{background:#fff0d8;padding:16px}</style>",
            "<h1>真实 Case1–10：三类本地算法的五检查点</h1>",
            "<p><a href='RESULTS_COMPARISON.md'>逐 Case 运行结果与原论文对照</a> · "
            "<a href='results_comparison.csv'>完整数值 CSV</a></p>",
            "<p>每个算法连续运行 800 次候选迭代，记录 0/200/400/600/800。"
            "每张快照包含 priority、空间热代理、HPWL 三张 256×256 map；同 Case 色阶固定。</p>",
            "<p>算法为本地 Metropolis SA、投影梯度和在线 REINFORCE。"
            "热代理未经 HotSpot 校准。单 seed 轨迹用于实现检查，不能据此判定算法总体优劣。</p>",
            "<p class='note'>Case4 的公开 wrapper 板为37×37 mm，芯粒总面积超过板面积（150.77%）。"
            "保留原配置的五检查点作为输入不可行性诊断，不纳入合法布局性能比较。"
            "其余运行也逐一报告实际重叠与合法性。</p>"]
    for case, result in zip(cases, results):
        rows.extend([f"<section id='{case}'><h2>{case}</h2>",
                     f"<p>{result['chiplets']} chiplets | area utilization "
                     f"{result['area_utilization']:.2%}</p>",
                     f"<img loading='lazy' src='{case}/metric_trends.png' alt='完整指标走势'>"])
        rows.append("<table><tr><th>算法</th><th>最终HPWL/mm</th><th>重叠率</th><th>合法</th></tr>")
        for a in METHODS:
            m = result["final_metrics"][a]
            rows.append(f"<tr><td>{LABELS[a]}</td><td>{m['hpwl_mm']:.3f}</td>"
                        f"<td>{m['overlap_fraction']:.6f}</td><td>{m['valid']}</td></tr>")
        rows.append("</table>")
        for a in METHODS:
            rows.append(f"<details><summary>{LABELS[a]}：五张快照、状态流与参数</summary>"
                        f"<p><a href='{case}/{a}/summary.json'>summary</a> · "
                        f"<a href='{case}/{a}/trace.jsonl.gz'>逐次参数轨迹</a></p>")
            rows.append(f"<img loading='lazy' src='{case}/{a}/state_flow.png' alt='五检查点总览'>")
            rows.append(f"<img loading='lazy' src='{case}/{a}/parameter_trends.png' alt='完整参数走势'>")
            for step in (0, 200, 400, 600, 800):
                prefix = f"{case}/{a}/checkpoint_{step:04d}"
                rows.append(f"<details><summary>iteration {step}</summary>"
                            f"<p><a href='{prefix}.pdf'>PDF</a> · "
                            f"<a href='{prefix}.json'>坐标与指标</a></p>"
                            f"<img loading='lazy' src='{prefix}.png' alt='iteration {step}'></details>")
            rows.append("</details>")
        rows.append(f"<details><summary>三算法最终−初始 map 差分</summary>"
                    f"<img loading='lazy' src='{case}/algorithm_effects.png'></details></section>")
    rows.append("</html>")
    (root / "index.html").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--refresh-geometry-reports", action="store_true",
                        help="Only after verifying all raw maps unchanged: refresh overlap reports")
    args = parser.parse_args()
    manifest = load_json(args.root / "manifest.json")
    cases = args.cases or manifest["cases"]
    results = ([load_json(args.root / case / "case_result.json") for case in cases]
               if args.index_only else
               [render_case(args.root, case, dpi=args.dpi,
                            refresh_geometry_reports=args.refresh_geometry_reports)
                for case in cases])
    if args.no_index:
        return
    (args.root / "rendered_cases.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_gallery(args.root, cases, results)


if __name__ == "__main__":
    main()
