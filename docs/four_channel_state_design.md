# 四通道状态设计：fourmap v2

逐通道的输入数据、公式、单位、色图、色条与读图示例说明见 [通道计算与图例手册](channel_reference.md)。

网页快照左到右与状态流上到下固定按四通道排列；每个 Case 提供逐芯粒 incident-pin 数及 q 值表。
129 个已采集状态的 v2 sidecar 由 `scripts/export_four_channel_contexts.py` 导出，新采集的状态元数据默认声明 v2 四通道。
独立轨迹重绘使用 `scripts/render_four_channel_trajectories.py`，不重跑优化；历史数据和 action reward 不回写。

## 决策对象与实现边界

三类 agent 固定为 SA、Analytical、RL 算法 worker。Coordinator 选择算法、配置、mask 与宏动作预算；统一 evaluator 返回指标和精确合法性。通道数量不代表 agent 数量。

`src/chiplet_layout_experiment/state_projection.py` 的独立 `project_state(problem, positions, grid_size=256)` 已实现四通道投影，输出 `[4,H,W]`，固定顺序：

| 索引 | 通道 | 含义与背景 |
| --- | --- | --- |
| 0 | priority | 归一化关联 pin 次数；背景 -1，孤立芯粒 0 |
| 1 | thermal_proxy | 复用历史热代理投影；不具有摄氏度语义 |
| 2 | hpwl | 复用历史 HPWL 空间贡献投影 |
| 3 | legality | 精确 pair-intersection coverage 与边界越界投影之和；无冲突背景 0 |

`STATE_VERSION = connectivity-priority-thermal-hpwl-legality-v2`，简称 fourmap v2。历史三联图及其 evaluator、reward 保留原始版本；新接口不回写这些记录。全局标量与预算在旁路融合；wiremask 是派生动作约束或可选门控，不是基础第五通道。

当前四通道 API、129 个状态、150 个 checkpoint 和 190 幅图的同步属于表示和可视化实现。拟训练的 2D CNN 输入为四通道；CNN 尚未训练，没有四通道带来选择收益的实验结果。2D CNN + history、GRU、ConvLSTM 与局部注意力属于未来实验。

## 1. Priority：次数归一化，保留并列

对芯粒 i，`d_i = count(incident pins on i)`：按关联 pin 出现次数计数，不去重成邻居数，不加 net weight，不使用时序关键度或计划放置顺序。

`q_i = d_i / max(max_j d_j, 1)`。

每个像素 g 按正面积相交确定覆盖集合 `C(g) = {i: area(g ∩ R_i) > 0}`：有覆盖时 `priority(g) = max_{i in C(g)} q_i`，否则为 -1。最大值合成与芯粒遍历顺序无关，不采用最后写入覆盖。相同 d 保留相同 q；全体无连接时芯粒均为 0，背景仍为 -1。旧记录中的 order 名称只用于历史兼容，不解释成 DAG 层级、rank 或 precedence。

## 2. Thermal proxy 与 HPWL

新投影直接复用 `problem.maps(...)[1]` 和 `[2]`，保持当前通道生成和归一化约定。热代理表达当前功耗/快速模型分布，不是温度场真值。HPWL 将网络包围盒线长贡献分配到空间；记录 pin offset、权重、退化包围盒处理和 projector 版本，精确 HPWL 标量仍由 evaluator 提供。

已有 27 个 HotSpot 校准样本的留出 MAE 为 58.91 °C，仿射拟合斜率为负；这不支持把当前代理标成摄氏温度，也不支持温度标定已可靠的结论。HotSpot 结果与代理分别记录；未来校准只在训练/校准分区拟合，并在独立留出部分评价。

## 3. Legality：两两交集与边界越界的空间诊断

设 N ≥ 1，R_i 是芯粒矩形，g 是板内像素单元。单位板坐标下，当前实现为：

`L_overlap(g) = sum_{i<j} area(g ∩ R_i ∩ R_j) / area(g) / max(N-1, 1)`。

先做精确矩形两两交集，再按交集覆盖像素的面积比例累加。保留 pair multiplicity，三重重叠贡献三个 pair；不能改成重叠区域并集。两芯粒只落入同一像素但几何上不相交时，不产生 overlap 项。

对每个芯粒和板边 s：

- `o_i,s` 为越过该边的距离，除以该边法向轴的板长；单位板坐标中板长为 1。
- `c_i,s(g)` 为芯粒切向区间与像素切向区间相交的长度，占像素切向长度的比例。
- `b_s(g)` 为最近板边的一像素边界带指示量。

`L_boundary(g) = sum_i sum_s o_i,s * c_i,s(g) * b_s(g) / N`。

实现分别处理 left/right/bottom/top，各边贡献相加；角点可以接收两侧贡献。切向区间完全在板外而没有任何像素覆盖时，取切向中心裁剪到板边后的最近像素，并赋切向覆盖权重 1，避免完全板外芯粒在图中消失。

`M_legality = L_overlap + L_boundary`，不截断到 [0,1]。合法且无冲突的背景为 0。两项的归一化不同，不可将其图上总和解释成精确重叠面积、越界面积或合法概率。

精确 GeometryEvaluator 是唯一合法性裁判。单独保存 exact valid、exact overlap、exact boundary 与容差及版本。栅格不能证明合法性；数值容差附近 map 非零也不必意味着 exact valid 为 false。不得以 map 是否为零覆盖 evaluator 判决。

## 4. 数据与问题身份

当前阶段有 2322 条标签、129 个状态，61 个 recovery 状态存在并列最优。标签不等于独立状态数，状态也不等于独立 case 数。并列最优保留集合式标签，命中任一最优 arm 均计为正确，并报告相对最优收益的 regret。

Case4 原始 37 板问题不可行；59 板替代版是新的问题定义，须使用独立 case/variant 身份和板尺寸元数据，不计作原始 37 板已恢复合法。数据划分时，同源变体归为一组以避免泄漏。

## 5. 消融与评价协议

三对四通道比较使用同一 canonical 状态重新生成的 v2 前三通道与完整四通道，唯一差异为 legality。历史三联图归档保留，不将旧 priority 的覆盖规则混入第四通道效应。

Priority 消融保持 thermal_proxy、hpwl、legality 不变，比较完整 q、删除 priority、仅占用（芯粒 0、背景 -1）、状态内按固定种子打乱芯粒 q。打乱保留 q 值的多重集与并列，继续使用相同 max 合成。

各组固定状态 ID、case holdout、全臂标签、奖励、配置库、种子集合、evaluator fidelity、在线评价预算与最终验证流程。按 case 及同源变体分组划分，禁止同一 case 的中间状态跨集合。归一化与调参仅使用训练/验证部分；匹配训练与调参预算，单独报告输入变化造成的参数量和投影成本差异。

报告收益预测误差、集合式 top-k 命中、regret、精确恢复合法率及同预算在线质量和资源成本；按留出 case 汇总配对差值和不确定性。热代理与摄氏温度指标分栏，不把代理下降写成温降。三对四通道、degree priority、历史上下文带来的收益均为待检验假设，不预填增益。

## 6. 状态与图表追溯

快照 `[4,H,W]`，状态流 `[T,4,H,W]`；效果图为相同状态、相同色标下的 before/after 及差分。Priority 背景 -1 与 legality 背景 0 分别标注，legality 色标不得隐式裁掉大于 1 的值。绘图色标缩放不回写训练张量。

保存 case/variant、板尺寸、状态 ID、checkpoint、step、agent、配置、seed、state version、通道顺序、归一化、evaluator 版本和精确标量。历史三联图与 v2 四联图使用不同版本身份，保证旧实验可重放。
