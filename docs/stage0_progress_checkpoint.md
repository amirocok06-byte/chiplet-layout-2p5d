# Stage 0 三支线恢复检查点

**记录时间：** 2026-10-21
**暂停原因：** 用户要求重启电脑后恢复三条工作支线；本文件用于恢复上下文，不改变实验方案 v1.0 的冻结定义。

## 当前仓库基线

- 主仓库：`F:/GAME/Chiplet-Layout-2.5D`
- 主分支：`main`
- 主分支当前提交：`3c9e8e7`
- 远端：`amirocok06-byte/chiplet-layout-2p5d`
- 主分支暂停时工作树干净。
- 已建立 `.worktrees/` 并加入 `.gitignore`；三条支线均从 `3c9e8e7` 创建。

## 三条工作支线

| 支线 | 分支 | worktree | 暂停状态 |
|---|---|---|---|
| Stage 0 核心 | `codex/stage0-core` | `F:/GAME/Chiplet-Layout-2.5D/.worktrees/stage0-core` | 有未提交实现与测试改动 |
| 可视化 | `codex/stage0-visualization` | `F:/GAME/Chiplet-Layout-2.5D/.worktrees/stage0-visualization` | 干净，尚未开始改动 |
| 科学对照 | `codex/stage0-baselines` | `F:/GAME/Chiplet-Layout-2.5D/.worktrees/stage0-baselines` | 干净，尚未开始改动 |

不要执行 `reset --hard`、`checkout --`、`clean` 或删除 worktree；core 支线的未提交修改是当前实验进度的一部分。

## 已完成的审计结论

现有 Stage 0 已形成：

```text
canonical Layout
  -> Geometry / HPWL / fixed thermal proxy
  -> SA / analytical / RL-compatible worker
  -> finite arm catalog
  -> JSONL + manifest
```

它可以作为 E0 协议基准，但当前 worker 仍是 smoke/local-rule baseline，不能支撑真实算法优越性或算法偏好结论。原始 `sample_arms` 只记录输出结果，已开始补齐输入指标、输出指标、指标差分和 evaluator 成本。

## core 支线当前未提交改动

位于 `codex/stage0-core`：

1. `default_arm_catalog()` 的六个 arm 统一使用 `budget=1`，避免默认配置把不同预算混入算法比较。
2. `SampleRecord` 增加 `input_metrics` 和 `delta_metrics`；`sample_arms()` 对同一输入先做 baseline evaluation，并记录 `baseline_evaluator_calls` 与 `total_evaluator_calls`。
3. `map_projector._resolve_order()` 不再把 chiplet 输入顺序静默当作 order；必须显式传入 priority/precedence order，或在 `Layout.metadata["chiplet_order"]` 中提供。
4. `tests/experiment/test_arms.py`、`test_sampling.py`、`test_map_projector.py` 已增加相应回归测试。
5. `docs/面向2.5D系统集成的芯粒布局方法-MVP1.md` 已同步说明新增字段和 order 约束。

## 已验证结果

- 三个新 worktree 在改动前均通过：`326 passed, 7 skipped`。
- core 的 TDD 红阶段已观察到预期失败：预算一致性、输入/差分指标和隐式 order 均先失败。
- core 当前局部绿阶段已通过：`tests/experiment/test_arms.py tests/experiment/test_sampling.py tests/experiment/test_map_projector.py`，共 `8 passed`。
- core 修改后的完整测试、提交和推送尚未执行。

## 重启后的恢复顺序

1. 运行 `git worktree list`，确认三个 worktree 仍在。
2. 进入 `stage0-core`，先运行完整 `python -m pytest -q`、`python -m compileall -q src scripts` 和 `git diff --check`。
3. 审核 core diff 后提交并推送 `codex/stage0-core`；不要覆盖 `main`。
4. 在 visualization 支线同步 core 提交，然后实现/运行三张 map、state flow、algorithm effect 的可复现输入和 PNG 输出。
5. 在 baselines 支线同步 core 提交，新增“原论文结果复述 + 本项目统一协议下的对照实验矩阵 + 汇总统计脚本/文档”。明确 smoke worker 只能验证协议，不能冒充论文复现结果。
6. 三条支线分别测试、提交、推送；最后在主仓库记录各分支 commit、数据版本、seed、evaluator 和预算。

## 设计边界

本阶段不提前实现 CNN、Contextual Bandit、连续超参数优化或真实 HotSpot。先完成可审计的 Stage 0 数据协议、可视化快照和固定算法对照，之后再以实验数据决定是否进入 CNN + Bandit。
