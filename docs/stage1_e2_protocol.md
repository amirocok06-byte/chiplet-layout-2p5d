# 阶段 1（E2a）执行协议：四通道状态条件化的有限臂选择

本机单卡 CNN 结构、训练与显存限制见
[轻量 CNN + LinUCB / 8GB 设计](stage1_cnn_linucb_8gb.md)。正式训练要求 CUDA，
CPU 仅供显式指定的功能测试；训练显存与 worker 宏动作预算分开计量。

## 范围与前置验收

阶段 0 已通过其冻结验收：129 个独立状态、2,322 条 action 标签、
完整的 6 arm × 3 seed 网格、65 次目标评价预算审计、坐标/指标重放以及
四通道图的对齐审计。本阶段**只读**该标签集；不重跑 worker，不改写坐标，
也不把阶段 0 的局部 worker 结果称为外部 TAP-2.5D、ATPlace 或 RLPlanner
的复现实验。

E2a 的问题是：给定一个已经合法的布局状态，四通道空间状态是否能帮助
控制器在有限配置库中选择短期收益更高的 `SA / analytical / RL` 宏动作？
恢复非法布局的 E2b 任务单列，原因是公共合法化会造成多个 arm 的并列最优，
不宜与合法态的线长/热代理优化共用一个唯一 winner 标签。

## “一步”的冻结定义

Bandit 的一步是一次**宏动作**：

`a_t = (worker, configuration, wiremask, H_eval = 65)`。

- 一个 SA 邻域提议、一个解析更新、一个 RL 环境动作都是 worker 内的
  **微步**，不直接跨算法比较。
- 一个宏动作从同一输入布局 `s_t` 出发，计 1 次起点评价与 64 次候选评价；
  worker 返回候选 `s_(t+1)` 和其前后指标。它不是“跑到收敛”。
- 统一后处理/合法化和共享 baseline 另列 evaluator 调用，不能计作某个
  worker 的能力，也不能从 65 次 action 预算中扣除。
- `T_wall` 和 `M_RAM` 是所有 arm 相同的强制停止上限，同时记录为成本；
  它们不是“一步”的定义。以相同内部迭代次数作预算是不公平的，因为三类
  worker 的微步语义和单步成本不同。

因此，离线标签的 `evaluator_calls=65`、`candidate_budget=64` 是 E2 的主要
公平账本；`total_action_seconds` 是模型质量—成本分析的次要结果字段。

## 状态、动作和监督信号

输入状态为 `X_t ∈ R^(4×64×64)`，由阶段 0 保存的 `problem.json`、坐标和
投影函数重新得到，不读取预渲染图片：

| 通道 | 表示 | 作用 |
| --- | --- | --- |
| priority | incident-pin 归一化优先级场 | 指示连线敏感区域 |
| thermal_proxy | 功率扩散代理峰值场，非摄氏温度 | 指示热风险空间分布 |
| HPWL | net bounding-box 的归因场 | 指示线长贡献区域 |
| legality | 精确重叠/越界投影 | 区分可行性风险 |

附加标量为归一化的 HPWL、热代理峰值、overlap 和合法标志。动作仅从阶段 0
冻结的六个有限 arm 中选择；在 E2a 首版中 `wiremask` 作为状态约束和审计
字段，不增加连续参数搜索。

每个 `(state, arm)` 的 reward 和 cost 都是三个 seed 的均值。每一状态的
oracle 是该状态所有 arm 的最大**集合最优** reward；如果有并列，任一并列
arm 都算 top-1 正确，避免伪造唯一标签。

## 切分、对照与报告

切分单位是 benchmark `Case`，不是状态、动作或 seed。同一 Case 的所有
checkpoint（含 Case4 几何版本）必须完全属于 train、validation 或 test
其中之一。E2a 报告至少包括：

1. 固定 arm：只用 train Case 的平均 reward 选出一个 arm；
2. 固定种子的随机 arm；
3. 只用四个全局标量的 Ridge 收益预测；
4. 四通道 2D CNN 的六臂 reward 预测；
5. 冻结 CNN embedding 上的 LinUCB（测试序列只观察已选 arm 的 reward）。

每个方法报告 per-arm MAE/MSE、集合 top-1/top-k、selected reward、oracle
reward、regret、所选 arm 的 `total_action_seconds`，并分别给出 Case 级结果。
所谓 online replay 仅指在测试状态顺序中用**被选择 arm**的已存标签更新
LinUCB；不允许把同一状态所有 arm 的完整标签泄漏给在线更新。

## 成功边界

若 CNN/Bandit 超过固定、随机和无空间标量对照，结论仅为：在这一冻结的
阶段 0 数据、同一宏动作预算和给定有限 arm 上，空间状态对动作选择有增益。
它不等价于证明已学得长期策略、连续超参数最优，或替代真实 HotSpot 温度。
