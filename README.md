# 面向2.5D系统集成的芯粒布局方法

**Chiplet Layout for 2.5D System Integration**

## 中文介绍

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

这里的 RL-compatible worker 是统一接口下的本地基线和适配边界，不宣称复现外部项目中未公开或未锁定版本的强化学习实现。真实路由、HotSpot、CPLEX、CNN 和 Contextual Bandit 训练属于后续阶段。

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

The RL-compatible worker is a local baseline and protocol boundary; it is not presented as a reproduction of an external reinforcement-learning implementation whose source, checkpoint, or dependency versions are not fully available. Routed wirelength, HotSpot, CPLEX, CNN training, and contextual-bandit training are reserved for later phases.

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

## 快速开始 / Quick Start

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

This repository is a private research workspace for the first implementation stage. It records interfaces, deterministic evaluation contracts, worker boundaries, finite arm configurations, and audit-friendly result manifests. Scientific performance claims require later benchmark, fidelity, and ablation studies.
