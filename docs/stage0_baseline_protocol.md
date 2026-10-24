# Stage 0：有限 arm 对照与证据边界

**状态：** 可执行协议；尚无真实算法比较结论。  
**研究问题：** 在同一 2.5D 状态、同一裁判和同一资源边界下，SA、解析式和 RL worker 的短期相对收益是否随空间状态而改变？

本文件把项目的贡献限定为一件可检验的事：把“多种算法可能适合不同状态”的动机，转化为可配对的 transition dataset 和后续的状态条件选择实验；不是预设一个“前期解析、中期 RL、后期 SA”的流程。

## 1. 证据层级：外部结果、研究动机与本项目结论必须分开

| 来源 | 可确认的角色 | 对本项目的用途 | 不允许的外推 |
|---|---|---|---|
| [TILOS: Assessment of RL for Macro Placement](https://arxiv.org/abs/2302.11014) | 在其自身协议内比较 RL、SA 和 RePlAce | 支持“比较须在同一协议、多个质量维度与稳定性下进行”的动机 | 不能当作 2.5D 芯粒 case 的固定阶段顺序 |
| [TAP-2.5D](https://github.com/bu-icsg/TAP-2.5D) | 热感知 2.5D SA 路线 | 候选 SA adapter 的实现来源 | 其论文数值不能移植为本项目结果 |
| [ATPlace2.5D](https://github.com/PKU-IDEA/ATPlace_pub) | 热感知解析式 2.5D 布局路线 | 候选 analytical adapter 的实现来源 | 不能把初始化/合法化流程直接视为 agent 偏好证据 |
| [RLPlanner](https://arxiv.org/html/2312.16895v2) | RL 和快速热模型路线 | 候选 RL adapter、动作约束与热 proxy 参考 | 不能把其训练先验计为零成本，也不能按其动作粒度比较 macro-step |
| [MaskPlace2.5D](https://github.com/xunk33333/MaskPlace2.5D) | mask 约束和 2.5D 放置学习参考 | wiremask 作为共同状态/动作约束的实现线索 | 不证明三种算法的普遍排序 |

当前仓库内的 `SAWorker`、`AnalyticalWorker` 和 `RLCompatibleWorker` 均是透明 smoke/local-rule workers；它们只验证接口、记录格式和资源计数。没有接入并锁定上述第三方实现的 commit、输入适配和许可证前，任何输出都不能称为论文复现或算法偏好。

## 2. 可检验假设与决策

| 问题 | 零假设 | 备择观察 | 最小证据 |
|---|---|---|---|
| 是否有状态条件偏好？ | 同一 state cluster 中各 arm 的质量—成本分布没有稳定差异 | 某些 cluster 中一个或多个 arm 有可重复 Pareto 优势 | 跨 case 的 paired full-arm transitions；留出 case 验证 |
| 空间 map 是否有用？ | 仅全局标量与 map 条件化选择的 held-out quality—cost 相同 | CNN+map 减少 oracle regret 或提高相同预算质量 | 固定 arm catalog、相同训练/验证划分和相同在线 evaluator 预算 |
| wiremask 是否有用？ | mask 不改变有效动作比例或质量—成本 | mask 降低无效候选且不损害质量 | 无 mask、只输入、输入+动作约束三组消融 |

这些是假设，不是本阶段的结论。即使一个 arm 在全局均值上最好，也不足以说明“不同状态需要不同算法”；需要在共同状态簇和跨 case 测试上观察稳定差异。

## 3. E1 全臂采样的实验单位与公平性

一个样本单位为：

```text
(case_id, initial_layout_checksum, phase, seed, evaluator_fidelity,
 arm_id, config_hash, wiremask_id, macro_budget)
  -> (output_layout_checksum, delta_metrics, resource_cost, status)
```

每个 `(case_id, initial state, seed)` 要枚举有限 arm catalog 中的所有 arm；arm 执行顺序由预生成、平衡的置换决定，避免固定先后污染缓存或资源。每个输出都由同一个 `PhysicsEvaluator` 重新评估。

| 固定项 | E1 的具体要求 |
|---|---|
| 状态 | 同一 canonical layout、相同显式 priority/precedence、相同 wiremask、相同 phase |
| 算法动作尺度 | 一个 arm 为一个 macro-step；内部 SA move、解析更新或 RL environment action 均不跨 worker 比较 |
| 搜索配置 | 只用预先登记的有限 config；不在某一算法上额外在线调参 |
| 质量裁判 | 同一个 geometry、HPWL、thermal evaluator 版本与 scalarization；worker 自带 reward 不作最终判据 |
| 资源 | 同时报 `total_evaluator_calls`、wall time、CPU/RAM（可得时 GPU/VRAM）、失败原因和预训练成本 |
| 随机性 | 每个 arm 跑同一 seed 集；case 按实例划分 train/validation/test，禁止把同一 case 的邻近状态泄漏到测试集 |

默认 catalog 当前令所有 arm 的 `macro_budget=1`；它确保接口层没有把四倍 SA 预算混进“算法差异”。真算法接入时，必须额外报告同 `N_eval`、同 `T_wall` 以及质量—成本 Pareto 三种资源视角，不能只选对某个算法有利的一种。

## 4. 记录、汇总与当前限制

`sample_arms()` 已记录输入指标、输出指标、差分和总 evaluator 调用数；使用：

```powershell
$env:PYTHONPATH = "src"
python -m chiplet_layout_experiment.stage0_summary <samples.jsonl> <summary.json>
```

`summary.json` 只按 arm 汇报可用率、均值 `ΔHPWL`、均值 `ΔTpeak`、运行时间和总 evaluator 调用数。它不输出“winner”，不替代配对显著性、置信区间、跨 case 汇总或质量—成本 Pareto 图。

当前 `FixedThermalEvaluator` 的峰值/均值只依赖芯粒功率密度和板级总功率，**不随 placement 位置变化**。所以 E0/E1 smoke 中 `ΔTpeak` 预期为零；thermal map 此时仅验证空间输入/可视化链路。若要研究热优化偏好，必须先替换为位置敏感的快速热 proxy，并以 HotSpot 或同等求解器校准和最终验证；在完成前不得报告热优化结果。

## 5. 进入 E2 前的验收门槛

只有同时满足下列条件，才允许以 E1 数据训练 2D CNN + Contextual Bandit：

1. 三个实际 adapter 已锁定来源 commit、许可证、输入转换和显式 config catalog；smoke worker 不能替代；
2. 每个记录含 case ID、phase、mask ID、状态/输出 checksum、输入/差分指标、资源和 evaluator fidelity；
3. 真实热目标已换成位置敏感并完成与高保真 evaluator 的校准；
4. full-arm 数据按 benchmark instance 划分，且每个 held-out initial state 有完整 paired arm 结果；
5. 已先报告固定 SA、固定 analytical、固定 RL、随机 arm、标量 bandit 等非学习选择基线。

若任何一项失败，系统仍是可靠的接口 smoke test，但不是可训练的“算法偏好”数据集。
