"""Generate a reproducible Markdown comparison from the thirty saved summaries."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import mean, median

METHODS = ("sa", "analytical", "rl")
NAMES = {"sa": "SA", "analytical": "解析法", "rl": "RL"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def load(root):
    runs = {}
    for i in range(1, 11):
        case = f"Case{i}"
        runs[case] = {}
        for a in METHODS:
            s = json.loads((root / case / a / "summary.json").read_text(encoding="utf-8"))
            assert s["status"] == "completed-configured-budget"
            assert s["steps"] == 800 and s["evaluator_calls_total"] == 801
            assert s["checkpoint_steps"] == [0, 200, 400, 600, 800]
            runs[case][a] = s
        initials = [runs[case][a]["checkpoints"][0]["positions"] for a in METHODS]
        assert initials[0] == initials[1] == initials[2]
    return runs


def final(summary):
    return summary["checkpoints"][-1]["metrics"]


def winner(case, key, index=-1):
    values = {a: s["checkpoints"][index]["metrics"][key] for a, s in case.items()}
    best = min(values.values())
    return [a for a, value in values.items() if abs(value - best) <= 1e-10]


def generate(root):
    data = load(root)
    eligible = [c for c, d in data.items() if not d["sa"]["source"]["proven_infeasible_by_area"]]
    stats = {"source_run_commit": "54e72613b25471c39dc92c0db5e0d2a5fe95b36a",
             "run_count": 30, "eligible_case_count": len(eligible), "excluded": ["Case4"],
             "descriptive_minimum_counts": {}, "algorithms": {}}
    for key in ("objective", "hpwl_mm", "thermal_proxy_peak"):
        stats["descriptive_minimum_counts"][key] = {
            a: sum(a in winner(data[c], key) for c in eligible) for a in METHODS}
    csv_rows = []
    for c, d in data.items():
        for a, s in d.items():
            m = final(s)
            initial = s["checkpoints"][0]["metrics"]
            csv_rows.append({
                "case": c, "algorithm": a, "eligible_area_check": c in eligible,
                **m, "initial_hpwl_mm": initial["hpwl_mm"],
                "initial_thermal_proxy_peak": initial["thermal_proxy_peak"],
                "hpwl_change_percent": 100 * (m["hpwl_mm"] / initial["hpwl_mm"] - 1),
                "thermal_reduction_percent": 100 * (1 - m["thermal_proxy_peak"] /
                                                    initial["thermal_proxy_peak"]),
                "optimizer_wall_seconds": s["optimizer_wall_seconds"],
                "capture_seconds": s["capture_seconds"],
                "peak_process_rss_mb": s["peak_process_rss_mb"],
                "evaluator_calls": s["evaluator_calls_total"],
                "gradient_calls": s["gradient_calls"],
            })
    for a in METHODS:
        selected = [data[c][a] for c in eligible]
        stats["algorithms"][a] = {
            "final_legal_count": sum(final(s)["valid"] for s in selected),
            "mean_optimizer_seconds": mean(s["optimizer_wall_seconds"] for s in selected),
            "median_optimizer_seconds": median(s["optimizer_wall_seconds"] for s in selected),
            "rss_range_mb": [min(s["peak_process_rss_mb"] for s in selected),
                             max(s["peak_process_rss_mb"] for s in selected)],
        }
    sections = [
        "# Stage 0 运行结果与原论文实验对照",
        "日期：2026-10-24。数值由本目录 30 份 summary.json 和五检查点记录自动汇总，"
        "对应运行代码提交 54e7261。生成器位于 scripts/summarize_trajectory_results.py；"
        "原始数字另存 [results_comparison.csv](results_comparison.csv) 和 "
        "[results_statistics.json](results_statistics.json)。",
        "## 1. 结论与本轮证据",
        "本轮观察到三类本地算法在不同指标上的侧重不同：解析法较易降低线长和综合目标，"
        "SA 的单次候选计算较轻、部分 Case 的热代理值更低，RL 在这次单种子运行中取得"
        "更多最终合法布局。这个观察限定于本项目当前实现与参数，不等于论文算法族的普适排名。",
        table(["算法", "最低 HPWL 的 Case 数", "最低热代理的 Case 数",
               "最低综合目标的 Case 数", "最终合法 Case 数"],
              [[NAMES[a], stats["descriptive_minimum_counts"]["hpwl_mm"][a],
                stats["descriptive_minimum_counts"]["thermal_proxy_peak"][a],
                stats["descriptive_minimum_counts"]["objective"][a],
                f"{stats['algorithms'][a]['final_legal_count']}/9"] for a in METHODS]),
        "上表仅统计面积检查未判定不可行的 9 个 Case；仍包含算法最终非法的诊断数值。"
        "最低数值统计不施加合法性过滤，因而不能将它读成最终可用解获胜次数。"
        "合法性优先于线长和目标比较。Case4 的所有结果单独保留用于输入诊断。",
        "## 2. 实验对象、指标和对照条件",
        "输入固定为 external/ATPlace2.5D 的 Case1–10；每个 Case 的三算法共享同一组"
        "seed=20261024 初始坐标，每次运行 800 次候选和 801 次共同目标评价，"
        "检查点为 0/200/400/600/800。解析法额外使用 800 次梯度计算；"
        "RL 是本地 NumPy 在线 REINFORCE，其本次训练成本计入耗时。"
        "本轮没有运行 CNN、Contextual Bandit 或 LLM 控制器。",
        table(["对象", "当前实现与含义"],
              [["SA", "Metropolis 移动与几何降温；不是 TAP 官方源码运行"],
               ["解析法", "平滑目标投影梯度；未加入 ATPlace 完整合法化流程"],
               ["RL", "逐实例在线 REINFORCE 与贪心接受；不是 RLPlanner PPO 复现"],
               ["HPWL↓", "保留真实 pin offset 的两 pin 线长总和，单位 mm"],
               ["thermal_proxy↓", "Gaussian 功率密度空间代理在芯粒中心的最大值；没有 K/°C 单位"],
               ["overlap_fraction↓", "所有矩形两两精确交集面积之和 / 芯粒总面积；多重重叠会重复计对"],
               ["penetration_penalty↓", "可导的轴向穿透惩罚，与真实相交面积分开记录"],
               ["priority/order", "incident-pin count / 最大 count；是共享观测，不是被优化的序列损失"]]),
        "综合目标为 J = HPWL/初始HPWL + 0.3 × thermal_proxy/初始proxy + "
        "20 × penetration_penalty。它支持同一 Case 内比较；初始值和面积归一化不同，"
        "不宜将跨 Case 的绝对 J 当成相同难度。所有最终指标取当前迭代，而非历史最优替换。",
        table(["Case", "芯粒数", "网数", "板尺寸/mm", "面积利用率", "最终合法算法"],
              [[c, d["sa"]["source"]["num_chiplets"], d["sa"]["source"]["num_nets"],
                "×".join(f"{x/1000:g}" for x in d["sa"]["source"]["board_um"]),
                f"{d['sa']['source']['area_utilization']:.2%}",
                "、".join(NAMES[a] for a in METHODS if final(d[a])["valid"]) or "无"]
               for c, d in data.items()]),
        "Case4 的固定公开 wrapper 板为 37×37 mm，芯粒面积 2064 mm²，"
        "面积利用率 150.77%，无重叠解在此定义下不可能存在。它不进入 9-Case 汇总。"
        "其余 Case 通过面积必要条件也不代表已证明可装箱；尤其 Case2 利用率约 98.37%，"
        "且本轮没有旋转/完整合法化。",
        "## 3. 不同 Case 的最终 HPWL",
        "单位 mm，越低越好。✓ 仅表示该运行最终合法；无 ✓ 的数字作为非法布局诊断，"
        "不能击败合法解。各 Case 的网数不同，绝对线长不跨 Case 排名。",
        table(["Case", "SA", "解析法", "RL", "不筛合法性的数值最低者"],
              [[c, *[f"{final(d[a])['hpwl_mm']:,.3f}" +
                     (" ✓" if final(d[a])["valid"] else "") for a in METHODS],
                "、".join(NAMES[a] for a in winner(d, "hpwl_mm")) +
                ("（不可行输入）" if c == "Case4" else "")] for c, d in data.items()]),
        "## 4. 最终热代理与重叠率",
        "热代理无温度单位；每个单元为“最终值（相对初始降低百分比）”。"
        "负降低率表示热代理变高，不可表述为降温。",
        table(["Case", "SA thermal_proxy↓", "解析法 thermal_proxy↓", "RL thermal_proxy↓"],
              [[c, *[f"{final(d[a])['thermal_proxy_peak']:.6f} "
                     f"({100*(1-final(d[a])['thermal_proxy_peak']/d[a]['checkpoints'][0]['metrics']['thermal_proxy_peak']):+.2f}%)"
                     for a in METHODS]] for c, d in data.items()]),
        "下表为精确几何 overlap_fraction 的百分比，不是 penetration 惩罚。",
        table(["Case", "SA 重叠率", "解析法重叠率", "RL 重叠率"],
              [[c, *[f"{100*final(d[a])['overlap_fraction']:.5f}%" for a in METHODS]]
               for c, d in data.items()]),
        "## 5. 综合目标与检查点排序",
        table(["Case", "SA J↓", "解析法 J↓", "RL J↓"],
              [[c, *[f"{final(d[a])['objective']:.6f}" for a in METHODS]]
               for c, d in data.items()]),
        "下表是各算法从共同起点独立运行后，在相同迭代号的 J 最低者。"
        "† 表示最低者当前仍非法；0 步三算法完全相同，省略。这里的“阶段”仅指预算进度。",
        table(["Case", "200 步", "400 步", "600 步", "800 步"],
              [[c, *[" / ".join(NAMES[a] + ("" if d[a]["checkpoints"][j]["metrics"]["valid"] else "†")
                               for a in winner(d, "objective", j))
                      for j in range(1, 5)]] for c, d in data.items()]),
        "Case1 的领先者是解析法→RL→解析法→SA，说明只看最终一次排名会丢失过程信息。"
        "但三个算法在 200 步之后处于不同布局；本表不是“在同一中间状态切换算法”的"
        "反事实比较，不能据此宣布某个阶段必然适合某算法。"
        "Case3/5/6/7/8/9/10 在四检查点均由解析法取得最低 J，"
        "也说明当前单种子数据并未显示所有 Case 都存在明显的阶段切换收益。",
        "## 6. 运行时间与内存",
        "下表为优化循环耗时，单位 s，扣除了快照采集时间；Python 启动、输入解析、"
        "最终绘图不在此字段中。测量来自独立顺序子进程，NumPy 线程数固定为 1。"
        "它是本地实现的计算记录，不是三类论文方法的最终效率排名。",
        table(["Case", "SA/s", "解析法/s", "RL/s"],
              [[c, *[f"{d[a]['optimizer_wall_seconds']:.6f}" for a in METHODS]]
               for c, d in data.items()]),
        table(["算法", "9 Case 平均优化耗时/s", "中位耗时/s", "峰值进程内存范围/MiB",
               "每运行额外操作"],
              [[NAMES[a], f"{stats['algorithms'][a]['mean_optimizer_seconds']:.6f}",
                f"{stats['algorithms'][a]['median_optimizer_seconds']:.6f}",
                "–".join(f"{x:.2f}" for x in stats["algorithms"][a]["rss_range_mb"]),
                "800 梯度" if a == "analytical" else ("800 在线策略更新" if a == "rl" else "800 扰动/接受判断")]
               for a in METHODS]),
        "峰值内存包含解释器、输入数组和快照数组，不是优化器独占增量。"
        "亚秒级单次运行受计时噪声影响，没有重复种子和置信区间；"
        "等候选数也不等于等 wall-clock、等算子成本。后续须同时报告同时间和同评价次数两套预算。",
        "## 7. 具体 Case 的可读结论",
        "- **Case1**：SA 与 RL 最终合法，SA HPWL 为 67,034.053 mm，"
        "比 RL 的 67,980.937 mm 低约 1.39%。解析法热代理略低，但仍有 0.05664% "
        "几何重叠，不能作为同等可用解。",
        "- **Case3**：解析法 HPWL 最低，为 87,314.656 mm，但剩余微小重叠；"
        "RL 是三个最终布局中唯一合法者。它说明线长最小和严格合法不是同一指标。",
        "- **Case5**：解析法与 RL 都合法；解析法 HPWL 约低 23.05%，"
        "RL 的热代理却更低。这里存在真实可用解之间的目标权衡。",
        "- **Case7**：解析法是唯一最终合法者，且 J、HPWL、热代理均最低；"
        "它是本轮解析全局更新较有效的具体实例，不是跨任务定律。",
        "- **Case10**：RL 的热代理从约 6.567030 上升至 6.671632（约 +1.59%），"
        "同时 J 下降，说明固定加权目标可能通过降低穿透惩罚补偿热项变差。"
        "后续需评估热硬约束或安全筛选，而非只优化综合目标。",
        "- **Case2/4/6/8/9/10**：三算法均未在固定预算内取得严格合法解。"
        "这些曲线可用于诊断优化行为，不构成最终可制造布局的性能排名。",
        "共同起点有重叠时，解除重叠通常会拉长网络；例如 Case1 三算法 HPWL 都"
        "比初始值更高，但热代理和穿透惩罚下降。因此不能把初始非法布局的短线长"
        "当成算法必须保持的可行基线。",
        "## 8. 原论文结果与本轮的关系",
        "__PAPER_SECTION__",
        "## 9. 当前能支持什么，以及下一步如何验证算法偏好",
        "本轮已支持：三算法连续状态采集可运行；同一 Case 的指标偏好会冲突；"
        "部分检查点数值排序会变化。它尚未测量 CNN/Bandit 的增益，"
        "也没有因果识别某个超参数的影响。",
        "下一组对照优先固定以下步骤：",
        "1. 修正/另立 Case4 几何版本，加入统一合法化接口与 HotSpot 校准；"
        "将非法状态的可行性恢复和合法状态的线长/热优化分别统计。",
        "2. 从同一组中间布局 s_t 分叉，给 SA、解析法、RL 分别相同评价预算；"
        "记录每个候选配置的 ΔHPWL、Δthermal、Δoverlap、合法率和运行成本。"
        "这才产生 Bandit 可用的同状态动作收益标签。",
        "3. 保持其余条件不变扫描 SA 温度/位移、解析学习率/平滑、"
        "RL 学习率/动作尺度；至少采用多 seed，按 Case 划分训练与测试。"
        "阶段由重叠、热点和空间贡献等状态定义，不硬编码为固定算法流水线。",
        "4. 最后比较固定单算法、随机选择、标量 Bandit、CNN+Bandit、"
        "带 wiremask/阶段上下文的控制器；运行成本与合法率和最终质量共同报告。",
        "全部图像入口：[index.html](index.html)。"
        "原始测量与本报告的分工：summary/trace 是证据，本文是受条件限定的解释；"
        "论文数字属于原作者的实验协议，不能并入本轮 30 次运行的统计。",
    ]
    literature_path = root / "paper_results_sources.md"
    if not literature_path.exists():
        raise FileNotFoundError("paper_results_sources.md must be verified before final report generation")
    literature = literature_path.read_text(encoding="utf-8")
    report = "\n\n".join(sections).replace("__PAPER_SECTION__", literature)
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(csv_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    return report + "\n", out.getvalue(), json.dumps(stats, indent=2, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = dict(zip(("RESULTS_COMPARISON.md", "results_comparison.csv",
                        "results_statistics.json"), generate(args.root)))
    for name, value in outputs.items():
        path = args.root / name
        if args.check:
            assert path.read_text(encoding="utf-8") == value, f"stale report: {path}"
        else:
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(value)
    print("verified" if args.check else "generated", ", ".join(outputs))


if __name__ == "__main__":
    main()
