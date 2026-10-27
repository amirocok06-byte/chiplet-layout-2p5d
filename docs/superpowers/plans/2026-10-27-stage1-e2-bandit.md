# 阶段 1 / E2：四通道 CNN + Contextual Bandit 实施计划

> 目标：在不改写阶段 0 数据的条件下，使用按 Case 分组的阶段 0 full-information 标签，检验四通道状态选择是否优于固定、随机和全局标量选择。该计划不实施 GRU、ConvLSTM、Transformer 或连续超参数搜索。

## 冻结协议

- 一步是宏动作：`(worker, configuration, wiremask, H_eval=65)`，不是内部迭代。
- 65 包含一个起点目标评价与 64 个候选评价；后处理与共享基线调用单列，不混入 arm 算法预算。
- `T_wall` 和 `M_RAM` 是等上限的中止条件与成本字段，不是每个 worker 的相同微步数。
- 输入是 `[priority, thermal_proxy, hpwl, legality]`；`thermal_proxy` 不作为温度标签。
- E2a 使用 `legal_optimization` 标签；恢复阶段的多最优标签保留给单独的 E2b 恢复任务，避免公共合法化造成的伪唯一 winner。

## 文件边界

| 文件 | 责任 |
| --- | --- |
| `src/chiplet_layout_experiment/stage1_data.py` | 只读加载阶段 0 的 Case、state、labels；重建 fourmap；按 arm 聚合 seed；生成分组数据集。 |
| `src/chiplet_layout_experiment/stage1_models.py` | 小型 2D CNN 多 arm reward 预测器、标量 Ridge 基线、固定/随机策略与冻结 encoder 的 LinUCB。 |
| `src/chiplet_layout_experiment/stage1_run.py` | 显式 CLI：按 Case 划分、训练、离线 sequential replay、JSON/CSV 审计输出。 |
| `tests/experiment/test_stage1_data.py` | 数据不变性、Case 切分、seed 聚合、相同 state 的 arm 对齐。 |
| `tests/experiment/test_stage1_models.py` | 模型输出 shape、固定/随机/标量/CNN/LinUCB 选择和 regret 计算。 |
| `tests/experiment/test_stage1_run.py` | 小型合成 fixture 的端到端审计与禁止 Case 泄漏。 |
| `docs/stage1_e2_protocol.md` | E2a/E2b 任务、宏动作预算、资源账本、允许和禁止的结论。 |
| `scripts/run_stage1_e2.ps1` | 固定参数的可复现实验命令，写入独立输出目录。 |

## Task 1：阶段 0 标签数据集（TDD）

**Files:** create `stage1_data.py`, create `test_stage1_data.py`.

1. 先写失败测试，构造含 2 个 Case、每 state 6 arm、每 arm 3 seed 的临时 JSONL；断言 `load_stage1_dataset()` 只接受完整 arm×seed 网格、对每 arm 聚合 reward/cost、并将同一 Case 的所有 state 保留为一个 group。
2. 实现 `Stage1State`（`state_id`, `case_group`, `phase`, `maps[4,H,W]`, `scalars`, `arm_rewards[6]`, `arm_costs[6]`）和 `load_stage1_dataset(root, phase, grid_size)`。
3. 用 `TracedProblem` + `project_state()` 从保存的 `problem.json` 和 `states/*.json` 重建 map；显式恢复 `hpwl_scale` 与 `thermal_scale`，不读取或写回旧 NPZ。
4. 运行数据测试，随后运行原有 `test_counterfactual.py` 与 `test_state_projection.py`，确保 stage 0 不变。

## Task 2：预注册切分与指标（TDD）

**Files:** extend `stage1_data.py`, extend `test_stage1_data.py`, create `docs/stage1_e2_protocol.md`.

1. 先写失败测试，断言 `split_by_case(states, train_cases, validation_cases, test_cases)` 拒绝重叠 Case、遗漏 Case 和将 Case4 几何版本拆开。
2. 实现 case-level split、集合式最佳 arm、oracle reward、selected-arm regret、top-k 命中、质量/成本向量。
3. 文档写明：E2a 只在相同的正式 stage0 label root 上做离线 full-information replay；所谓 online 是测试状态顺序中仅用所选 arm reward 更新 Bandit，不是重跑优化器。
4. 运行数据和协议测试。

## Task 3：小型模型与无空间对照（TDD）

**Files:** create `stage1_models.py`, create `test_stage1_models.py`, modify `pyproject.toml` only to增加可选 `stage1` 依赖。

1. 先写失败测试：固定策略从训练集平均 reward 选择一个 arm；随机策略可用固定 seed 重放；Ridge 仅接收全局标量；CNN 输入 `[B,4,64,64]` 并输出 `[B,6]`。
2. 按 2026-10-28 用户硬件要求，实现本机单卡 CUDA 小型 CNN：两层 3×3 convolution、最大 32 个通道、4×4 adaptive average pooling 和 6-output head；CPU 仅供显式功能检查。模型输入由同一坐标在64×64重新投影，256×256展示图保留。实际结构、RTX 4060 8GB 限额和资源账本见 `docs/stage1_cnn_linucb_8gb.md`。
3. 训练时只用 train Case；validation 只用于选择固定 epoch/regularization；测试 Case 不参与标准化、early stopping 或 arm 优先级估计。
4. 加入每 arm MSE/MAE、top-1/top-k、oracle regret 的明确计算。
5. 运行模型测试与全量回归。

## Task 4：冻结 CNN 特征上的 LinUCB（TDD）

**Files:** extend `stage1_models.py`, extend `test_stage1_models.py`.

1. 先写失败测试：相同 state 序列、固定 seed 和固定 alpha 给出相同选择；更新只使用 selected arm 的 reward；矩阵维度与 arm 数一致。
2. 实现 `LinUCB`，输入为冻结 CNN embedding 与全局标量的连接；训练 Case 可做完整监督 warm start，test Case 按顺序选择、读取所选 arm 标签并更新。
3. 输出每一宏步的 selected arm、observed reward、oracle reward、regret、对应 action 的 `evaluator_calls=65`、历史记录运行成本；禁止把标签中的所有 arm reward 当作在线已观察结果。
4. 运行确定性测试与全量回归。

## Task 5：可复现实验、结果与同步

**Files:** create `stage1_run.py`, `scripts/run_stage1_e2.ps1`; create `test_stage1_run.py`; update `README.md`, `task-journal.md`, `docs/stage1_e2_protocol.md`.

1. CLI 必须要求 `--labels-root`、`--train-cases`、`--validation-cases`、`--test-cases`、`--seed`、`--output`；拒绝复用已有输出目录。
2. 保存 config、源码 hash、split、state IDs、模型超参数、每步选择、评价指标与运行环境；产出 `summary.json`、`decisions.jsonl`、`audit.json`。
3. 先在合成 fixture smoke run，后在正式标签执行留出 Case；若 torch 不存在，给出明确依赖错误，不悄悄退化为非 CNN 模型。
4. 执行完整测试、审计输出、git diff 检查、commit、push 和远端核验。

## 自检

- 阶段 0 标签只读：覆盖 Task 1、Task 5。
- Case 分组、集合最优、replay 边界：覆盖 Task 2、Task 4。
- 四通道 CNN 与标量对照：覆盖 Task 3。
- 有限 arm Contextual Bandit：覆盖 Task 4。
- 审计、复现、同步：覆盖 Task 5。
