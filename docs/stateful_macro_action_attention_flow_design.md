# 状态化宏动作、注意力与流程记忆：2.5D 芯粒布局扩展设计

> 2026-10-26：当前可执行状态为 fourmap v2，形状 `[4,256,256]`，顺序 priority、thermal_proxy、HPWL、legality；priority 是关联 pin 数归一化后的连接度。公式、颜色、色条与读图边界见 [通道手册](channel_reference.md)。本文中的早期三图讨论作为设计沿革，后续 CNN/时序/注意力实验均应按 v2 四通道及相应消融执行。

> 本文是 `experimental_design_contextual_bandit_2p5d.tex/pdf` 的扩展设计，补充近期关于决策步粒度、CNN 与注意力、循环记忆以及流程级影响建模的思考。它不替换现有 MVP-1，也不把当前的 RL-compatible worker 表述为正式 RL 复现。

## 1. 本次扩展要解决的核心问题

已有方案的主线是：

```text
布局状态 -> 空间编码 -> 选择算法/配置/预算 -> worker -> Physics Evaluator -> 新状态
```

本次扩展进一步回答三个容易被混淆的问题：

1. **一步到底是什么？** 不能把 Contextual Bandit 的一步定义为 SA 的一个随机移动、RL 的一个环境动作，也不能定义为某个算法一直运行到收敛。建议把一步定义为一次固定资源预算下的 **宏动作（macro-action）**。
2. **空间异常和历史变化如何进入决策？** 不把注意力图直接等同于解释或因果关系，而是采用“空间编码 + 显式诊断 + 循环记忆”的结构，将当前三张/多张 map 与最近几步的变化联合输入控制器。
3. **如何避免只优化下一步？** 先保留 Contextual Bandit 作为第一版可验证基线；当结果需要体现跨步骤影响时，引入带 n-step 回报或 flow critic 的状态化控制器，而不是简单把多个 agent 串联成固定流水线。

因此，项目的核心不是“让多个 agent 互相调用”，而是学习一个可审计的映射：

```text
状态结构 + 历史状态转移 + 资源预算
        -> 哪个 worker/config/budget 在当前状态下最有价值
        -> 该宏动作会如何改变后续布局状态
```

## 2. 统一决策单位：微步、宏步和 episode

| 层级 | 定义 | 是否由上层控制器逐次选择 | 记录内容 |
|---|---|---:|---|
| 微步 micro-step | SA 的一次邻域移动、解析法的一次更新、RL 的一个环境动作 | 否 | worker 私有轨迹，可选采样 |
| 宏步 macro-step | 一个 worker 在固定 evaluator 预算下连续运行若干微步后返回候选布局 | 是 | 起点状态、动作、终点状态、指标差分、成本 |
| episode | 从初始布局到停止条件的完整优化过程 | 否 | 宏步序列、最终布局、累计收益和总预算 |

宏动作定义为：

\[
a_t=(k_t,c_t,H_t,M_t),
\]

其中 `k_t` 是算法类别（SA、analytical、RL），`c_t` 是离散配置，`H_t` 是 evaluator 或迭代预算，`M_t` 是 wiremask/动作约束。一个宏步结束的条件是达到 `H_t`、worker 主动停止、超时或产生不可修复的非法布局。这样既响应“agent 自主选择算法和参数”的要求，也避免控制器被迫理解三个算法的全部内部循环。

第一版只选择有限的 `(算法, 配置, 预算)` arm；连续超参数优化放到后续版本。这样才能把第一版的科学问题限定为“空间状态能否预测不同算法的相对收益”。

## 3. 推荐模型路线：轻量混合结构，而不是直接使用全局 Transformer

### 3.1 第一版：CNN/ECA + Contextual Bandit

对 `256×256` 输入先用共享 CNN 提取局部空间结构，再用轻量通道注意力（例如 ECA 或 CBAM 风格的通道/空间门控）突出当前重要的 map 通道。控制器对每个 arm 预测：

\[
\hat q_t(a)=f_\theta(z_t,e_a,g_t),
\]

其中 `z_t` 是空间特征，`e_a` 是 arm embedding，`g_t` 是全局指标、剩余预算、阶段编码等标量。第一版的目标是收益预测和 arm 选择，不建模长序列。

### 3.2 第二版：CNN/ECA + GRU 或 ConvLSTM

如果第一版说明同一个静态状态对应的最优 arm 依赖最近的变化方向，则保存固定长度窗口：

\[
X_{t-L+1:t},\quad \Delta X_{t-L+1:t},\quad a_{t-L+1:t-1}.
\]

使用 GRU 处理压缩后的全局特征，或使用小型 ConvLSTM 保留空间变化。推荐先做 GRU，再做 ConvLSTM：GRU 的显存和数据需求更低，能先验证“记忆是否有用”；ConvLSTM 只有在空间位置本身对转移预测有明显贡献时才值得引入。

### 3.3 第三版：低分辨率/局部窗口注意力

不建议在 `256×256` 全分辨率上直接做像素级全局自注意力。可行的升级顺序是：

1. 将 map 下采样为 `16×16` 或 `32×32` patch token；
2. 使用局部窗口或 shifted-window attention；
3. 使用 chiplet token、热点区域 token 和 wiremask 区域 token；
4. 只让注意力模块负责跨 map、跨时间和跨区域的信息整合。

这比直接训练一个大 Transformer 更符合当前项目的成本约束。注意力权重只能作为诊断线索，不能直接宣称“模型发现了因果热点”；应同时输出显式的诊断 mask、预测指标差分，并用 evaluator 反馈验证。

## 4. 状态输入、wiremask 与结构化记忆

### 4.1 输入张量

当前状态可以组织为：

\[
X_t=[M_{occ},M_{power},M_{thermal},M_{wire},M_{legal},M_{order},M_{phase},v_t].
\]

其中 `M_phase` 表示当前优化阶段或动作历史的空间投影，`v_t` 包含 HPWL、峰值温度、重叠面积、利用率、剩余预算和最近动作。

建议把模型输出拆成三个头：

- **诊断头 `D_t`**：预测当前最值得干预的区域、指标和异常类型；
- **转移头 `F_t(a)`**：预测候选宏动作对 map 和全局指标的变化；
- **选择头 `Q_t(a)`**：将预测收益、成本、风险和长期价值合成为 arm 分数。

这三个头分别对应“哪里有问题”“执行后会怎样”“现在是否值得执行”，避免把一个 attention heatmap 同时当成定位、因果和决策依据。

### 4.2 wiremask 的职责

wiremask 不是一个独立阶段，也不是只在流程末尾使用。它同时承担两种职责：

1. **结构化输入**：作为 CNN/注意力模块的通道，让模型看到连接拥挤、长连线敏感或热-线长耦合的区域；
2. **动作约束**：限制 SA 邻域、解析更新或 RL 动作的候选区域、方向、半径和可行域。

在第一版中，mask 采用确定性生成器最稳妥：由 HPWL 贡献、net bounding box、拥挤度、热图和合法性膨胀规则产生。阈值、膨胀半径和通道权重放入有限配置库，作为可消融因素。后续才考虑由诊断头预测软 mask，并通过 evaluator 反馈校准。

### 4.3 结构化 memory packet

不要把每一轮内部状态写成自然语言 prompt 交给 LLM。内循环应使用可序列化的结构化记忆：

```json
{
  "step": 7,
  "phase": "repair_or_refine",
  "focus_region": [[x0, y0, x1, y1]],
  "focus_metric": "hpwl",
  "mask_version": "wire-v2",
  "last_action": {"worker": "sa", "config": "sa-wire", "budget": 4},
  "predicted_delta": {"hpwl": -2.1, "peak_temperature": 0.3},
  "observed_delta": {"hpwl": -0.4, "peak_temperature": 0.1},
  "uncertainty": 0.28,
  "remaining_budget": 96,
  "recent_actions": ["analytical-wl", "sa-wire"]
}
```

GRU/ConvLSTM 读取该信息的张量化版本；若后续需要 LLM，它只用于外层实验配置、异常审计或生成可读解释，不参与每一个 inner-loop 决策。这样可以保留 agent 的可解释接口，又避免自然语言成为不可控的状态传递协议。

## 5. 让控制器关注整个流程，而不只关注下一步

纯 Contextual Bandit 只根据当前上下文估计单步收益，适合作为第一版基线，但它不天然表达“这一步会不会让后续更容易”。因此实验设计分三层推进：

### 层 1：短期收益基线

\[
r_t^{local}=J(X_t)-J(X_{t+1})-\lambda C_t-\rho I_t,
\]

其中 `J` 是归一化目标，`C_t` 是时间/evaluator 调用成本，`I_t` 是非法或修复惩罚。此版本用于验证算法选择是否比固定算法和随机选择更有效。

### 层 2：n-step/terminal value

对同一宏步记录后续 `K` 个状态的累计收益：

\[
R_t^{(K)}=\sum_{i=0}^{K-1}\gamma^i r_{t+i}+\gamma^K V(X_{t+K}).
\]

这仍可用轻量网络实现，不必立即引入完整 RL。它能检验某个动作是否牺牲了少量即时 HPWL，却改善了后续热或合法性。

### 层 3：动作条件的 flow critic

对每个 `(状态, arm)` 学习：

\[
F_\theta(X_t,a_t)\rightarrow(\widehat{\Delta M},\widehat{\Delta v},\widehat{cost},\widehat{uncertainty}).
\]

决策器依据预测的局部改善、后续价值、成本和不确定性选 arm。最终论文应报告“预测准确率”和“在线布局结果”两类证据，不能只展示 attention 图。

## 6. 轨迹、状态流与训练样本的边界

三者不能混称：

| 对象 | 含义 | 适合的模型 |
|---|---|---|
| 算法轨迹 | 某个 worker 内发生的所有微步、候选布局、接受/拒绝或梯度更新 | worker 调试、算法分析 |
| 宏观转移 | `(X_t, a_t, X_{t+1}, Δmetrics, cost)` | 2D CNN + Bandit |
| 状态流 | 固定宏步时间尺度上的 `X_t -> X_{t+1} -> ...`，强调状态如何变化 | GRU、ConvLSTM、时序注意力 |

因此第一版只需要完整保存宏观转移；第二版再按固定窗口保存状态流。若把不同 case 的中间状态随机打散到训练集，模型可能记住实例布局而非学习算法偏好，应按 benchmark instance 划分训练/验证/测试。

## 7. 与现有代码的对接计划

现有 MVP-1 已提供稳定的协议边界，扩展不应破坏它：

| 现有位置 | 本次扩展的对应职责 |
|---|---|
| `src/chiplet_layout_experiment/arms.py` | 将 `(worker, config, budget)` 明确为 `MacroAction`/finite arm；后续可加入 mask 配置和阶段编码 |
| `workers.py` | 保持 `run` 作为完整宏动作接口；后续增加 `snapshot/restore` 或 `step`，保存 worker 私有状态 |
| `evaluator.py` | 继续作为唯一指标裁决者；扩展返回 raw metrics、约束、成本、fidelity 和可选的空间 map |
| `sampling.py` | 从单臂样本扩展为 `(X_t,a_t,X_{t+1},ΔX,Δmetrics,cost)` 转移记录 |
| `canonical.py` | 继续作为所有 worker 和 encoder 共享的唯一布局 IR |
| `docs/experimental_design_contextual_bandit_2p5d.tex` | 保留第一版主实验和公平预算定义；本文提供状态化/时序扩展 |

建议的后续文件边界是：

```text
context.py       map、全局标量和阶段信息的编码
masking.py       wiremask 生成、版本和动作约束
memory.py        memory packet、GRU/ConvLSTM 输入窗口
bandit.py        arm 选择、探索策略和成本归一化 reward
flow_critic.py   动作条件状态转移/长期价值预测
```

当前只应先实现 `context.py`、`masking.py` 的可审计数据格式和离线转移采集，不要同时实现大模型、全局 Transformer、连续超参数优化和高保真热仿真。

## 8. 由易到难的实验顺序

### E0：协议和评估器一致性

同一布局输入 SA、analytical、RL-compatible worker，使用相同 evaluator、seed 和预算。确认 HPWL、热代理、合法性和 checksum 口径一致。

### E1：有限 arm 的全臂采样

对每个 benchmark 采集所有 arm 的宏观转移，建立算法偏好表。只报告“在某状态簇/资源预算下某 arm 的相对收益”，不把它写成普遍算法优越性。

### E2：2D CNN + Contextual Bandit

对比：固定 SA、固定 analytical、固定 RL、随机 arm、仅全局标量、2D CNN + Bandit。主要指标为最终目标、evaluator calls、wall-clock、regret、arm 命中率和非法率。

### E3：CNN + GRU

加入最近 `L` 个宏步的 memory packet，检验历史是否提升收益预测和在线选择。若 GRU 无增益，不继续增加 ConvLSTM/Transformer。

### E4：ConvLSTM/低分辨率注意力

只在 E3 证明存在时序或空间转移价值后进行。对比静态 CNN、GRU、ConvLSTM、低分辨率局部注意力，并严格控制参数量、训练样本和 evaluator 预算。

### E5：连续超参数和 flow critic

最后才让控制器在每个算法的可信范围内调整温度、邻域半径、步长或 RL 配置。此时必须将参数搜索成本计入 reward，并保留有限 arm 版本作为归因基线。

## 9. 论文主线、证据和边界

### 建议的中心主张

在统一 2.5D 布局表示和物理评估预算下，**状态条件化的宏动作控制器**能够利用空间 map、wiremask 和有限历史，选择比固定算法或无状态随机组合更合适的布局优化器，并在相同 evaluator 成本下取得更好的质量—成本折中。

### 必须由实验回答的问题

1. 空间 map 是否比全局标量更能预测各算法的相对收益？
2. wiremask 是仅有输入价值，还是同时能减少无效动作和 evaluator 成本？
3. 历史状态流是否带来超过静态状态的在线决策收益，而不仅仅是离线预测精度？
4. 控制器是否学习到跨 benchmark 的布局规律，而不是记住 case ID？

### 不能直接声称的内容

- 注意力权重本身不等于物理因果解释；
- 使用三个 worker 不等于多 agent 协同创新；
- 预测状态转移不等于替代真实物理评估器；
- RL-compatible worker 的 smoke 实现不等于复现外部 RL 论文；
- 3D CNN/Transformer 只有在消融显示其带来决策收益时，才应作为方法贡献。

## 10. 参考依据与使用边界

以下工作用于支持表示、时序建模和算法选择的设计启发，并不直接证明它们在本项目的 2.5D 芯粒布局中有效：

- [CBAM](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)：通道/空间注意力的轻量设计参考；
- [Swin Transformer](https://arxiv.org/abs/2103.14030)：局部窗口和层次化视觉表示参考；
- [PredRNN](https://arxiv.org/abs/2103.09504)：空间—时间状态流建模参考；
- [Self-Attention ConvLSTM](https://ojs.aaai.org/index.php/AAAI/article/view/6819)：循环记忆与注意力结合的参考；
- [TimeSformer](https://arxiv.org/abs/2102.05095)：时空注意力分解参考；
- [RL-DAS](https://arxiv.org/abs/2403.02131)：布局设计空间中状态、动作和长期优化的背景参考；
- [DSPLACE](https://doi.org/10.1587/transele.2025ECS6022) 与 [PPAPlace](https://arxiv.org/abs/2608.13790)：芯片布局中学习型状态表示和优化闭环的相关背景；
- [ChiPFormer](https://proceedings.mlr.press/v202/lai23c.html)：芯片相关视觉/序列表示的启发性参考。

本项目的直接证据仍必须来自自己的统一 evaluator、固定预算、跨实例划分和消融实验。上述文献只用于说明设计动机、选择可行的网络组件和构造对照，不应替代本项目的实验结果。

## 11. 三类 Agent 的最终职责边界

本项目中的三类 Agent 最终定义为三种算法 Agent，而不是将 HPWL、thermal、order 分别包装成三个 Agent：

- `SA-Agent`：负责在给定 wiremask、当前布局和预算下执行受约束的局部交换、移动、旋转或扰动，并自动选择温度、降温率、邻域半径和接受准则等有限配置。
- `Analytical-Agent`：负责在可行域和 wiremask 约束下执行解析式连续优化、合法化和局部修复，并选择步长、惩罚权重、平滑强度和迭代预算等配置。
- `RL-Agent`：负责使用统一布局接口生成候选动作序列或局部策略，并选择 rollout 长度、探索率、候选动作数和局部更新预算等配置。

`order`、`thermal`、`HPWL` 是三个共享观察通道；`PhysicsEvaluator` 是唯一的物理裁判；上层 Contextual Bandit/Coordinator 才是选择 `(agent, config, mask, macro_budget)` 的决策器。这样三类 Agent 有真实的算法身份，而不是把三个目标函数或三个 evaluator 误称为 Agent。

## 12. HPWL 与 order 的形式化空间表示

热图可直接以温度场的颜色表示；HPWL 和 order 也必须定义为具有明确语义的栅格场，而不是把一个全局标量复制到每个像素。

对网格像素 `p`，设网络 `n` 的引脚包围盒为 `B_n`，加权 HPWL 为 `L_n`，网络权重为 `w_n`，定义：

    M_hpwl(p) = sum_n [ w_n * L_n / |B_n| * 1(p in B_n) ]

其中 `|B_n|` 是包围盒覆盖的像素数。于是 `sum_p M_hpwl(p)` 等于加权 HPWL（离散化误差仅来自边界量化），高值区域表示对当前线长贡献较大的空间区域。

对于 order，单一的 rank 图和关系图必须分开保存：

- `M_order_rank`：在芯粒占据区域内填充 `rank(c_i)/(N-1)`，颜色由早到晚或由低到高表示拓扑顺序；空白区域为 0，并配合有效区域 mask。
- `M_order_trans`：对相邻顺序节点的中心连接线或 precedence edge 绘制带宽度的走廊，值随边的 precedence level、跨越距离或该次操作的修改强度归一化。
- 若 order 来自 DAG，应额外保存 `precedence_edges` 或节点 token；栅格图只作为 CNN 输入，关系边作为轻量 graph/token 分支输入。

当前 MVP 的 thermal 通道是功率密度代理，必须在元数据中标注 `thermal_proxy`；最终论文应使用 HotSpot 或等价热求解器生成温度场进行校准和验证。HPWL/order map 不是物理梯度，也不宣称替代 evaluator。

## 13. 公平预算与一步的定义

一个 `step` 定义为：一个 Agent 在固定的 `macro_budget` 内产生一个候选布局，候选布局经过一次统一 `PhysicsEvaluator`，并记录前后状态、动作、资源和 reward。不能把 SA 的单次邻域交换、解析法的一次梯度迭代和 RL 的一次 rollout 直接称为同一个 step。

公平比较至少锁定资源向量 `(N_eval, T_wall, M_RAM, M_VRAM)`，其中 `N_eval` 是统一的真实 evaluator 调用次数，`T_wall` 是墙钟时间上限，`M_RAM/M_VRAM` 是峰值存储上限。当前 `arms.py` 中不同 worker 的默认内部预算只适合 smoke test；正式对比应改为相同 `N_eval` 或同一时间预算，并同时报告质量、运行时间、峰值内存、evaluator 次数和失败率。

## 14. 三个可审计的可视化程序

仓库新增三个脚本，分别对应单快照、状态流和算法动作效果：

 - `scripts/plot_snapshot_maps.py`：输入 `maps [3,H,W]`，输出固定顺序的 `order / thermal / hpwl` 三联图。
 - `scripts/plot_state_flow.py`：输入 `maps [T,3,H,W]`，按通道共享色标绘制时间列，用于观察一次宏动作前后的空间状态变化。
 - `scripts/plot_algorithm_effects.py`：输入 `before_maps [3,H,W]` 和 `after_maps [K,3,H,W]`，绘制每个 Agent 的 `after-before`，直接比较不同算法对三类通道的影响。

`src/chiplet_layout_experiment/map_projector.py` 提供可测试的栅格化接口。所有图都应保存 `case_id、step、agent、config_id、seed、evaluator_version、normalization` 元数据，避免把不同归一化方式下的颜色误读为性能差异。

## 15. 项目同步规则

从本次设计开始，项目采用以下强制同步规则：每次源码、实验配置、测试、文档或 PDF 修改，都必须在本地 `F:/GAME/Chiplet-Layout-2.5D` 完成对应更新、测试和构建后，提交到 Git，并同步推送到 GitHub 仓库 `amirocok06-byte/chiplet-layout-2p5d`。部署完成后必须再次检查是否产生了需要同步的相关源码、配置、测试、文档、图表或 PDF；若有，必须补充提交并再次推送。该规则同时记录在仓库根目录 `task-journal.md`、`.planning/.../progress.md` 和 `CONTRIBUTING.md` 中。

## 16. v1.0 短期冻结与语义修正

从 2026-10-21 起，本项目将 `docs/experimental_design_v1_freeze.md` 作为短期实验基线。以下定义覆盖本文件中较早的模糊表述：

1. 三个 Agent 固定是 `SA-Agent`、`Analytical-Agent` 和 `RL-Agent` 三种算法 worker；`order`、`thermal`、`HPWL` 是共享 observation map，`PhysicsEvaluator` 是唯一裁判，Contextual Bandit 才是上层选择器。
2. 共享的 `order` map 不表示左右方向，也不直接表示某个 worker 的运行时放置序列。若 benchmark 提供 precedence DAG，则使用算法无关的 priority/precedence rank 栅格，并额外保存 precedence edges 或 token；运行时序列另存为 `placement_sequence_map`，仅用于算法轨迹诊断。
3. 一个 Bandit decision 对应一个固定预算的 `macro-step`：worker 产生候选布局，统一 evaluator 评价并记录前后状态。SA 邻域交换、解析更新和 RL environment action 都只是 worker 内部 `micro-step`。
4. v1 主方法固定为有限 arm 配置库 + 2D CNN + Contextual Bandit；连续参数、GRU、ConvLSTM 和局部注意力必须以后续消融形式加入，不得在第一版主实验中同时改变多个因素。
5. 文献只支持“算法偏好可能随状态和预算变化”的研究假设，不支持预先宣称固定的阶段顺序。阶段变量应作为上下文，让全臂采样学习 `Preference(agent | state_cluster, phase)`。

短期冻结期间，除非出现实现 bug、协议不可运行、全臂采样证明定义不可识别，或消融结果提供明确反证，否则不修改 Agent 身份、map 语义、macro-step 定义和公平预算。详细的冻结方案、公式、实验矩阵和开源边界见 [`experimental_design_v1_freeze.md`](experimental_design_v1_freeze.md)。
