# 面向2.5D系统集成的芯粒布局方法

2026-10-28 新增[配置偏好与选择器实测回答](docs/figures/stage1_preference_probe/RESULTS.md)：
同61状态、17/65/257评价预算、每配置3seed；另完成65预算的9折Case留出×3训练seed。
偏好随状态/预算变化，但当前CNN/LinUCB总体未超过固定配置。原pilot缩放风险经干预未改善结果，
51/61状态的胜出配置至少一次需要公共合法化；结论适用于完整宏动作接口。

阶段 1（E2a）新增 [轻量 CNN + LinUCB 与本机 8GB 设计](docs/stage1_cnn_linucb_8gb.md)、
[执行协议与宏动作定义](docs/stage1_e2_protocol.md)。首版 CNN 为 21,870 参数、
四通道 `64×64` 输入、六个配置收益输出，训练入口为 `scripts/train_stage1_cnn.ps1`。
正式训练显式要求单卡 CUDA，并按本机实际显存设置 allocator 限额与峰值账本。
阶段 0 的 CPU worker、阶段 1 的 GPU 训练分别记录资源预算。
2026-10-28 本机 [GPU首轮结果](docs/figures/stage1_e2_8gb_pilot/RESULTS.md)：
PyTorch峰值allocated/reserved为19.65/24 MiB；本轮CNN与固定策略打平，
LinUCB较差，[解释与下一批实验](docs/figures/stage1_e2_8gb_pilot/INTERPRETATION.md) 已记录。

2026-10-26 状态表示更新为 fourmap v2：`[priority, thermal_proxy, HPWL, legality]`。
[通道公式、单位与图例](docs/channel_reference.md) · [四通道网页](docs/figures/stage0_real_trajectories/index.html) · [更新的实验方案PDF](docs/experimental_design_contextual_bandit_2p5d.pdf)。
已由原始坐标更新 150 个检查点、190 幅图，并为 129 个同状态布局导出四通道观察；原始动作收益与历史三通道记录保留。

2026-10-25 新增：[同状态分叉、公共合法化与分阶段收益实验](docs/same_state_experiment.md)。
实现分支为 `codex/stage0-visualization`，沿用 core 的 paired 记录并引入 baselines 的统计模块；
从同一中间布局运行六个真实参数配置，生成 CNN/Bandit 可用的向量收益标签。
结果入口为 [stage0_same_state](docs/figures/stage0_same_state/README.md)，
数据使用结论见 [标签可识别性与合法化对照](docs/figures/stage0_same_state/EXPERIMENT_FINDINGS.md)，
几何版本与 HotSpot 状态分别见 [几何说明](docs/geometry_legalization.md)、[热校准说明](docs/hotspot_calibration.md)。

**Chiplet Layout for 2.5D System Integration**

目前，模拟退火（Simulated Annealing, SA）、解析式布局（Analytical Placement）和强化学习兼容布局（RL-compatible Placement）三类方法已经具备足够成熟的研究与工程基础。本项目将这三类方法放入统一的 2.5D 芯粒布局实验协议中，在相同输入、物理评估器、预算、随机种子和结果记录规范下进行公平比较，并为后续 CNN 状态编码与 Contextual Bandit 方法提供可复现的数据接口。

当前版本聚焦第一阶段 CPU-only、离线、可审计的最小闭环：

```text
Bookshelf case
  -> canonical layout IR
  -> Geometry / pin-offset weighted HPWL / fixed thermal proxy
  -> SA / analytical / RL-compatible worker
  -> finite arm catalog
  -> JSONL samples + manifest
```

这里的 RL-compatible worker 是保留的 smoke 接口。连续轨迹另有真正更新策略权重的 NumPy REINFORCE，本项目不宣称外部算法复现。已接通真实 HotSpot 并进行留出校准；当前 block 参考模型与代理的误差尚未达可用标准。真实路由、完整分层热模型校准、CPLEX、CNN 和 Contextual Bandit 训练仍待实现。

## English Overview

At the current stage, simulated annealing (SA), analytical placement, and RL-compatible placement have reached a sufficiently mature research and engineering foundation. This project places the three method families under a unified 2.5D chiplet-layout evaluation protocol, enabling fair comparison with the same inputs, physics evaluator, budgets, random seeds, and result-recording rules. It also provides reproducible data interfaces for subsequent CNN state encoding and contextual-bandit selection.

The current release focuses on a CPU-only, offline, auditable first-stage loop:

```text
Bookshelf case
  -> canonical layout IR
  -> Geometry / pin-offset weighted HPWL / fixed thermal proxy
  -> SA / analytical / RL-compatible worker
  -> finite arm catalog
  -> JSONL samples + manifest
```

The RL-compatible worker remains a smoke interface; continuous trajectories use genuine NumPy REINFORCE updates without claiming external-paper reproduction. Real HotSpot calls and held-out calibration are implemented, but the block reference model does not yet validate the spatial proxy. Routed wirelength, layered-package validation, CPLEX, CNN training and contextual-bandit training remain future work.

## 目录 / Repository Layout

```text
src/chiplet_layout_workflow/       workflow and validation package
src/chiplet_layout_experiment/     Phase 1 canonical IR, evaluators, workers, arms
tests/                              regression and experiment tests
configs/                            default configuration, schemas, prompts
docs/                               experiment designs and MVP-1 documentation
references/                         external projects and source boundaries
data/sample/                        public sample-data boundary
scripts/                            reproducible helper-script boundary
.github/workflows/                  continuous integration
```

本次新增的状态化扩展设计见 [`docs/stateful_macro_action_attention_flow_design.md`](docs/stateful_macro_action_attention_flow_design.md)。该文档在现有有限 arm + Contextual Bandit MVP 基础上，补充宏动作定义、wiremask 约束、CNN/GRU/ConvLSTM/局部注意力的递进路线，以及面向整个优化流程的状态流建模；不改变当前 MVP-1 的代码边界。

正式实验前的短期冻结基线见 [`docs/experimental_design_v1_freeze.md`](docs/experimental_design_v1_freeze.md)。v1.0 固定三类算法 Agent、共享 map 语义、macro-step 和公平预算；未经全臂采样、消融或实现错误证据，不改变这些核心定义。

三类 Agent 的职责在设计中固定为 `SA-Agent`、`Analytical-Agent` 和 `RL-Agent`；`order`、`thermal`、`HPWL` 是共享状态 map，`PhysicsEvaluator` 是统一裁判。HPWL 贡献场、order rank/transition 场的定义和公平预算协议见实验设计 PDF/TeX。

特别地，v1.0 中共享 `order` map 表示算法无关的 priority/precedence map；算法运行时的 placement sequence 只作为轨迹诊断，不作为三类 Agent 的共享输入。

可视化脚本为 `scripts/plot_snapshot_maps.py`、`scripts/plot_state_flow.py` 和 `scripts/plot_algorithm_effects.py`；其输入分别为单快照 `[3,H,W]`、状态流 `[T,3,H,W]` 和算法前后状态。项目同步规则记录在 [`task-journal.md`](task-journal.md)。

Stage 0 使用 `2-Benchmarks/external/ATPlace2.5D/Case1..Case10` 作为共同输入。`2-Benchmarks/cases` 是相同设计的另一份副本，共有文件仅换行不同；额外热学仓库尚未下载实际数据。输入审计与外部源码边界见 [`docs/stage0_data_source_freeze.md`](docs/stage0_data_source_freeze.md)。新增连续 SA、投影梯度与在线 REINFORCE 的五检查点方案见 [`docs/five_checkpoint_trajectory_design.md`](docs/five_checkpoint_trajectory_design.md)。

## 快速开始 / Quick Start

真实 Case1–10 的三类算法连续轨迹见 [五检查点结果总览](docs/figures/stage0_real_trajectories/index.html) 和 [结果说明](docs/figures/stage0_real_trajectories/README.md)：每个算法有五张三联图、状态流和全量参数走势。图中的热场为未校准空间代理；合法性按实际结果报告。

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q src
```

The first-stage implementation does not require PyTorch, CUDA, external solvers, SSH jobs, or third-party source execution.

## 参考项目 / References

The external-project list and reproduction boundaries are documented in [`references/README.md`](references/README.md). External repositories are references only; this repository does not silently copy or execute their source code.

## 状态 / Status

This repository is a public research workspace for the first implementation stage. It records interfaces, deterministic evaluation contracts, worker boundaries, finite arm configurations, and audit-friendly result manifests. Scientific performance claims require later benchmark, fidelity, and ablation studies.

运行结果与原论文的逐项比较见 [Stage 0 运行结果与论文对照](docs/figures/stage0_real_trajectories/RESULTS_COMPARISON.md)，包括 Case1–10 的线长、热代理、重叠、检查点排序、时间与内存，以及 TAP、ATPlace、RLPlanner 原文实验条件与结果。
