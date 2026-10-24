 # Task Journal / 任务日志

## 长期同步规定

1. 每次源码、配置、测试、文档、图表或 PDF 修改，都必须同步更新本地目录 `F:/GAME/Chiplet-Layout-2.5D` 与 GitHub 仓库 `amirocok06-byte/chiplet-layout-2p5d`。
2. 同步顺序为：本地修改 → 运行相关测试/构建 → 检查 diff → Git commit → push 到 GitHub → 核验远端提交。
3. 每次部署完成后，必须再次思考并检查是否需要同步修改相关源码、实验配置、测试、README、论文源文件、PDF、图表或 GitHub 内容；如果需要，必须补充修改、测试、提交和推送。
4. 实验结果、生成图和 PDF 必须带有可追溯的版本、配置、seed、evaluator 和预算信息；禁止只更新生成文件而不更新源文件。

## 本次记录（2026-10-20）

本条之前的最新同步（2026-10-24）：可视化分支已提交并推送 1461f6a，新增 RESULTS_COMPARISON.md、论文出处核验、30 行 CSV/统计 JSON 和可复现汇总脚本；报告比较不同 Case/指标并区分合法性与论文实验条件。用户明确要求仓库公开，现已设置 PUBLIC、isPrivate=false；main README 增加公开报告入口。本次仅提交 README 与本日志，不暂存或覆盖主工作树原有 PDF 删除状态。

- 将三类 Agent 明确为 `SA-Agent`、`Analytical-Agent`、`RL-Agent`；`order/thermal/HPWL` 作为共享 map 通道，`PhysicsEvaluator` 作为唯一裁判。
- 增加 HPWL 贡献场、order rank 场和 order transition 场的形式化定义。
- 增加 `map_projector.py` 与三个状态/状态流/算法效果绘图脚本，并为 map projector 增加测试。
- 将公平预算、一步定义、可视化元数据要求和同步规定写入实验设计文档。

## 本次记录（2026-10-21）

- 将实验方案冻结为 v1.0：三类 Agent 固定为 `SA-Agent`、`Analytical-Agent`、`RL-Agent`；`PhysicsEvaluator` 固定为统一裁判；Contextual Bandit 负责选择 `(agent, config, mask, macro_budget)`。
- 将共享 `order` map 的语义修正为算法无关的 priority/precedence map；运行时 placement sequence 不再作为共享输入，只保留为算法轨迹诊断。
- 固定第一版主实验为有限 arm 配置库 + 2D CNN + Contextual Bandit，GRU/ConvLSTM/局部注意力和连续超参数优化只能作为后续递进消融。
- 新增 `docs/experimental_design_v1_freeze.md`，并同步修订状态设计、README、CONTRIBUTING、LaTeX/PDF。
- 从本次起，除非出现实现 bug、协议不可运行、全臂采样证明定义不可识别，或消融结果提供明确反证，否则不得修改上述核心定义；任何修改仍须按“本地修改 → 测试/构建 → diff → commit → push → 远端核验”同步。

## 本次记录（2026-10-24，Stage 0 三支线恢复）

- 核心协议分支 `codex/stage0-core` 已同步 `b74201f` 与 `c54047c`：有限 arm 统一 `macro_budget=1`；每条记录含输入、输出、差分和 evaluator 成本；order 必须显式提供 priority/precedence，且空白固定为 `-1`，与 rank=0 区分。
- 可视化分支 `codex/stage0-visualization` 已同步 `80f215f` 与 `764f209`：新增可复现 synthetic smoke bundle、三张 map、状态流和 action-conditioned map 差分。PNG/manifest 仅验证数据协议，明确不构成 benchmark、第三方复现或算法偏好结论。
- 对照分支 `codex/stage0-baselines` 已同步 `13d7b4e`：新增按有限 arm 汇总 paired record 的工具，以及“外部文献角色—本项目待检验假设—统一对照协议”的边界文档。
- 本轮部署检查确认：`FixedThermalEvaluator` 对 placement 位置不敏感，当前热图与 `ΔTpeak` 只能用于数据链 smoke；接入真实算法和 CNN/Bandit 前，必须更换为位置敏感的热 proxy，并以 HotSpot 或同等求解器校准。
- 主分支保留了用户已有的 `docs/experimental_design_contextual_bandit_2p5d.pdf` 删除状态；本轮未暂存、提交或覆盖该文件。
- 从本次起，除非出现实现 bug、协议不可运行、全臂采样证明某个定义不可识别，或消融结果提供明确反证，否则不得修改上述核心定义；任何修改仍须按“本地修改 → 测试/构建 → diff → commit → push → 远端核验”同步。

## 暂停检查点（2026-10-21）

- 已建立三个隔离 worktree：`codex/stage0-core`、`codex/stage0-visualization`、`codex/stage0-baselines`，恢复信息详见 `docs/stage0_progress_checkpoint.md`。
- 当前暂停时 core 支线保留未提交的 Stage 0 协议改动；可视化和对照支线保持干净，等待 core 协议稳定后同步。
- 重启后必须先恢复 worktree 状态和 core 未提交 diff，再继续测试、提交、推送；禁止用 reset/clean 丢弃进度。
