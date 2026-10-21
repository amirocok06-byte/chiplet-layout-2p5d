# 2.5D 芯粒布局状态化算法组合：实验方案 v1.0（短期冻结版）

**版本：** v1.0
**冻结日期：** 2026-10-21
**文档定位：** 正式实验前的可执行方案基线。短期内不改变 Agent 身份、三类 map 的核心语义、macro-step 定义和公平预算；只有出现明确的实验失败、实现 bug 或数据证据，才允许修改，并必须在任务日志中记录原因。

## 1. 研究主线

本项目不把三种算法机械地串成流水线，也不把 HPWL、thermal、order 三个指标包装成三个 Agent。主线是：

> 在统一 2.5D 布局表示和统一 PhysicsEvaluator 下，利用当前空间状态、wiremask 和有限历史，预测不同算法宏动作的短期收益，并在相同评估预算下选择更适合当前状态的算法与配置。

因此系统只有一个核心学习问题：

```text
布局状态 X_t -> 状态编码器 -> Contextual Bandit/Coordinator
             -> 选择 (Agent, config, mask, macro_budget)
             -> 算法 Agent 产生候选布局
             -> PhysicsEvaluator 统一裁决
             -> 新状态 X_{t+1} 与 reward 反馈
```

CNN 或后续的 ConvLSTM/注意力只负责状态表征和收益/状态转移预测；它们不替代真实物理评估器，也不直接被解释为物理梯度模型。

## 2. 三类 Agent 的准确定位

三类 Agent 是三种算法 worker。Agent 的身份来自其可执行的优化机制和配置空间，而不是来自观察指标或评估器。

| Agent | 算法身份 | 适合处理的状态倾向 | 可调配置示例 |
|---|---|---|---|
| `SA-Agent` | 模拟退火/随机邻域搜索 | 局部残余冲突、非光滑约束、离散扰动、需要跳出局部结构的状态 | 初温、降温率、邻域半径、接受准则、局部预算 |
| `Analytical-Agent` | 解析式连续优化 + 合法化/局部修复 | 目标函数可平滑表达、全局密度或线长仍有明显下降空间的状态 | 步长、平滑强度、密度/热惩罚、迭代数、合法化预算 |
| `RL-Agent` | 学习得到的动作策略/候选序列 | 有重复拓扑分布、需要长程动作序列、已有策略先验的状态 | rollout 长度、候选动作数、探索率、局部更新预算 |

上层 `Contextual Bandit/Coordinator` 选择 `(agent, config, mask, macro_budget)`；`PhysicsEvaluator` 是唯一的最终裁判。`order/thermal/HPWL` 是共享观察通道，不是三个 Agent。第一版应使用有限配置库，先验证“是否存在状态条件化算法偏好”，再研究连续超参数优化。

## 3. 文献证据与边界

已有文献支持“不同算法在不同条件下有不同优势”这一研究动机，但没有提供一个可直接移植到本项目、对所有 2.5D case 都成立的固定阶段顺序。

- [TILOS: Assessment of Reinforcement Learning for Macro Placement](https://arxiv.org/abs/2302.11014) 统一比较了 RL、SA、RePlAce 等宏布局方法；其结果显示 proxy cost、最终线长、时序和稳定性之间并不由同一个算法同时占优，说明算法偏好应条件化而不是固定排序。
- [TAP-2.5D](https://github.com/bu-icsg/TAP-2.5D) 代表热感知的 SA 路线，并将温度和线长/路由约束纳入 2.5D 布局处理。
- [RLPlanner](https://arxiv.org/html/2312.16895v2) 代表 RL + 快速热模型路线，并使用动作约束处理不可行动作；其报告结果不能直接视为本项目统一预算下的结论。
- [ATPlace2.5D](https://github.com/PKU-IDEA/ATPlace_pub) 代表解析式热感知布局路线；它的初始化、解析优化和合法化阶段说明解析法可以承担全局目标驱动，但代码边界和评估协议需要单独核对。
- [Cyclic RL and Simulated Annealing](https://arxiv.org/abs/2011.07577) 提供“RL 生成较好粗初始化、SA 做局部细化”的机制启发，但该工作不是 2.5D 芯粒布局，不能作为本项目固定阶段顺序的证明。

因此本文只把下列内容作为待验证假设：粗粒度高重叠阶段可能更适合解析法或 RL 初始化；局部残余冲突和非光滑合法性修复可能更适合 SA；中间阶段由状态和预算决定。最终必须用全臂采样和消融实验学习 `Preference(agent | state_cluster, phase)`，不能预先写成“前期 analytical、中期 RL、后期 SA”的事实结论。

## 4. 三张 map 的核心定义

最终 CNN 输入不是只有三张图；推荐输入为：

```text
X_t = [M_order, M_thermal, M_hpwl, M_occ, M_legal, M_wiremask, v_t]
```

其中 `v_t` 包含全局 HPWL、峰值温度、合法率、利用率、剩余预算、最近动作和阶段编码。`M_occ` 与 `M_legal` 用于区别空白区域、不可用区域和真实零值。

### 4.1 Thermal map

优先使用网格温度场：

```text
M_thermal(g) = T(g) - T_ambient
```

同时保存 raw 温度、归一化温度和 evaluator fidelity。快速模型只能标记为 `thermal_proxy`，HotSpot 或等价高保真热求解结果用于校准和最终验证。

### 4.2 HPWL contribution map

不能把一个全局 HPWL 标量复制到每个像素。对网络 `n`，令 `B_n` 为其引脚包围盒，`L_n` 为加权 HPWL，`w_n` 为网络权重，则：

```text
M_hpwl(g) = sum_n [w_n * L_n / |B_n| * 1(g in B_n)]
```

这样在离散化误差可控时，所有像素贡献之和等于加权 HPWL。高值区域表示对当前线长贡献较大的空间区域。建议绘图使用 case 内固定的分位数或对数归一化，并另存原始数值。

### 4.3 Order map：必须是算法无关的优先级/先后约束

`order` 不能默认表示左右顺序，也不能直接使用某个 RL worker 的运行时放置顺序，因为 SA 和解析法没有同构的天然序列。

v1.0 的共享 `M_order` 定义为**算法无关的 priority/precedence map**：若 benchmark 给出 precedence DAG `D=(V,E)`，先为芯粒 `c_i` 计算拓扑层级或任务优先级 `r_i`，再把归一化优先级栅格化到芯粒占据区域：

```text
M_order(g) = r_i / R_max,  if g belongs to footprint(c_i)
             background,    otherwise
```

同时保存 `precedence_edges` 或节点 token，因为单一 rank 图不能完整表达 DAG 关系。若 benchmark 没有自然的先后约束，不应人为制造 order；应将该通道改名为 `priority/criticality map`，并从任务元数据或关键路径定义得到。

算法实际运行时产生的序列另存为 `placement_sequence_map`，只作为算法特定的轨迹诊断，不作为三类 Agent 的共享输入。这个区分是 v1.0 中对 order 语义的硬约束。

## 5. 一步、轨迹、状态流与 evaluator

### 5.1 一步的定义

一个 `macro-step` 定义为：同一 Agent 在固定 `macro_budget` 内运行若干内部微步，产生一个候选布局；候选布局经过一次统一 `PhysicsEvaluator`；系统记录前后 map、指标、动作和资源。SA 的一次邻域交换、解析法的一次梯度更新和 RL 的一次 environment action 都是 worker 内部 `micro-step`，不能直接当作可比较的一步。

### 5.2 算法轨迹与状态流的区别

- **算法轨迹：** 一个 worker 内部的全部微步、候选布局、接受/拒绝、梯度更新或 rollout，主要用于调试和分析算法内部行为。
- **状态转移：** 一个可供 Bandit 学习的样本 `(X_t, a_t, X_{t+1}, Δmetrics, cost)`。
- **状态流：** 在固定 macro-step 时间尺度上串联的 `X_t -> X_{t+1} -> ...`，强调空间状态随决策历史的变化，供 GRU/ConvLSTM/时序注意力使用。

第一版只需要完整保存宏观转移；第二版再按固定窗口保存状态流。训练/验证/测试应按 benchmark instance 划分，避免网络记住 case ID。

### 5.3 PhysicsEvaluator 实现边界

所有 Agent 只负责产生候选布局，不得自带另一套最终 reward。统一 evaluator 分为：

1. `GeometryEvaluator`：边界、重叠、可行性、利用率和合法性；
2. `WirelengthEvaluator`：pin offset 加权 HPWL，必要时另报 routed wirelength；
3. `ThermalEvaluator`：快速热代理用于在线筛选，HotSpot/等价求解器用于校准和最终验证；
4. `CostTracker`：真实 evaluator 次数、墙钟时间、CPU/RAM/VRAM、失败率和 fidelity。

最终 reward 由 evaluator 统一计算，例如质量改善、合法性惩罚、峰值温度变化和资源成本的组合。不得把快速代理误称为物理真值，也不得把 CNN 的状态预测当作 evaluator 替代品。

## 6. 固定的 v1.0 实验链路

### E0：统一接口与评估器一致性

建立 canonical layout IR，将 TAP、ATPlace 和 RLPlanner 风格输入转换到统一坐标、单位、pin offset 和约束。确认三类 worker 由同一个 evaluator 评价，并输出相同字段。

### E1：有限 arm 全臂采样

先定义有限配置库：

```text
A = {(k, c) | k in {SA, Analytical, RL}, c in C_k}
```

对相同初始布局、相同 seed 集、相同 evaluator fidelity 和相同预算，依次运行所有 arm。记录：

```text
(case, phase, X_t, arm, config, mask, macro_budget,
 X_{t+1}, delta_metrics, evaluator_calls, wall_time,
 peak_memory, failure_reason)
```

该阶段先回答“不同状态下是否存在稳定的算法偏好”，不做连续超参数优化。

### E2：2D CNN + Contextual Bandit

训练轻量 2D CNN 预测每个 arm 的短期收益或收益分布，再接 LinUCB 或 Thompson Sampling。wiremask 同时作为输入通道和动作约束。主对照为：固定 SA、固定 analytical、固定 RL、随机 arm、仅全局标量 Bandit、2D CNN、2D CNN+wiremask、phase-conditioned Bandit。

### E3：CNN + GRU

只有当 E2 表明静态状态存在系统性不足时，才加入最近 `L` 个 macro-step 的 memory packet，检验历史是否提高收益预测和在线选择。

### E4：小型 ConvLSTM/低分辨率局部注意力

只有在 E3 证明状态变化有额外信息时才训练。控制参数量、输入窗口、训练样本和 evaluator 预算；注意力权重只能作为可视化线索，不能直接宣称物理因果解释。

### E5：连续超参数和 flow critic

最后才让控制器在每个 Agent 的可信范围内调节连续参数。必须把搜索成本计入 reward，并保留有限 arm 版本作为主要归因基线。动作条件 flow critic 预测 `F(X_t,a_t) -> (delta_map, delta_metrics, cost, uncertainty)`，用于评价动作对整个流程的影响，而不是只看下一步的即时变化。

## 7. 公平比较规范

正式比较同时锁定资源向量：

```text
(N_eval, T_wall, M_RAM, M_VRAM)
```

其中 `N_eval` 是真实 evaluator 调用次数，`T_wall` 是墙钟上限，`M_RAM/M_VRAM` 是峰值内存。每个 macro-step 按相同的 evaluator budget 或相同墙钟预算比较，并报告：最终目标、HPWL、峰值温度、合法率、evaluator 次数、墙钟时间、峰值内存、失败率、Bandit regret 和 arm 命中率。若某 Agent 需要预训练，分别报告包含训练成本的端到端结果与已训练 worker 的部署结果。

## 8. 证明方案科学性的关键消融

1. 全局标量上下文 vs 三张空间 map：证明空间信息是否真的预测算法相对收益。
2. 无 wiremask vs wiremask 仅输入 vs wiremask 输入+动作约束：证明 mask 是否真正减少无效动作并改善局部修复。
3. 固定算法 vs 随机组合 vs 状态条件 Bandit：证明收益来自条件化选择，而不是“多跑了几个算法”。
4. 2D CNN vs CNN+GRU vs ConvLSTM/局部注意力：证明历史和状态流是否带来在线决策收益，而不仅是离线预测精度。
5. 有限配置库 vs 连续超参数 Bandit：证明连续调参的额外收益是否抵消了搜索成本。
6. 同一状态簇内比较算法优先级：报告 `Preference(agent | state_cluster, phase)`、oracle regret 和质量—成本 Pareto，而不是只报告全局平均值。

## 9. 允许的结论与禁止的过度表述

允许的中心结论是：在统一 2.5D 表示、evaluator 和资源预算下，状态条件化的宏动作控制器能否比固定算法或无状态组合更好地选择布局优化器。

在没有完成上述对照前，不得声称：

- 三个 Agent 组成就等于多 Agent 协同创新；
- 某个算法普遍适合某个固定阶段；
- CNN/ConvLSTM/Transformer 替代了物理评估器或物理梯度；
- attention 权重就是物理因果解释；
- 外部论文报告的数值就是本项目的实验结果。

## 10. 开源实现与复用边界

- SA/2.5D： [TAP-2.5D](https://github.com/bu-icsg/TAP-2.5D)
- 解析式布局： [ATPlace_pub](https://github.com/PKU-IDEA/ATPlace_pub)
- RL/热模型参考： [RLPlanner](https://github.com/weiweihook/RLPlanner)
- 掩码/热模型参考： [MaskPlace2.5D](https://github.com/xunk33333/MaskPlace2.5D)
- 综合 2.5D 布局参考： [STAMP2.5D](https://github.com/MicroSystemsDesignLab/STAMP2.5D)
- 热评估器： [HotSpot](https://github.com/uvahotspot/HotSpot)

这些项目只作为实现参考或 baseline 来源。运行前必须记录具体 commit/hash、许可证、依赖版本、输入转换和评估边界；第三方 worker 产生候选布局后，必须回到本项目统一 evaluator。

## 11. 短期冻结条件

以下内容从 v1.0 起冻结：

1. 三类 Agent 固定为 `SA-Agent`、`Analytical-Agent`、`RL-Agent`；
2. `PhysicsEvaluator` 不是 Agent，而是统一裁判；
3. `order` 的共享语义固定为算法无关的 priority/precedence map；运行时 placement sequence 只做诊断；
4. 一个 Bandit decision 对应一个 macro-step，不对应某个算法的单个内部循环；
5. 第一版使用有限 arm 配置库，不与连续超参数优化同时做主实验；
6. 2D CNN + Contextual Bandit 是主方法，GRU/ConvLSTM/局部注意力是后续递进消融；
7. 所有结论以统一 evaluator、固定资源预算和跨实例划分为准。

只有以下情况可以修改冻结内容：实现 bug、统一评估协议无法运行、全臂采样证明某个定义不可识别，或消融实验提供了明确反证。修改必须同步更新本文件、LaTeX/PDF、状态设计文档、README、task-journal 和 GitHub。
