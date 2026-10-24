### 原论文证据范围与阅读层级

核验日期：2026-10-24。三篇核心论文均已获取全文，以下为**作者报告结果**，不代表本项目算法复现或当前轨迹实验。页码均从 PDF 第 1 页起计。附件笔记仅作为待核实材料；未将其结论当作原论文证据。未重复本项目运行结果表。

TAP-2.5D：核对官方 DATE 全文的方法、实验部分；ATPlace2.5D：核对作者网站全文的方法、实验表格；RLPlanner：核对指定 arXiv:2312.16895v2 的两页全文及 HTML 表格，不混用其他版本数字。[TAP 官方全文](https://www.date-conference.com/proceedings-archive/2021/pdf/1516.pdf)；[ATPlace 作者全文](https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf)；[RLPlanner v2 全文](https://arxiv.org/pdf/2312.16895v2)。

### TAP-2.5D（DATE 2021）：温度改善需要与布线条件一起报告

该方法采用 SA、Occupation Chiplet Matrix、HotSpot 热评估及 MILP 网络布线。实验通常采用 45×45 mm interposer，SA 运行 5 次取最好解；单次预算校准为约 25 小时、4500 步。比较对象 Compact-2.5D 使用 B*-tree 与 fast-SA，也是 TAP 的初始化。线长为互连总长度，不能直接称为本项目的 HPWL。（PDF pp.2–4，§III-A、III-B、III-C、§IV 开头。）[原文](https://www.date-conference.com/proceedings-archive/2021/pdf/1516.pdf#page=2)

- **Multi-GPU，非流水 repeaterless 链路**：Compact 的 95.31°C、88,059 mm → TAP 的 91.25°C、96,906 mm；对应降温 4.06°C，线长增加约 10.05%（根据原始数值计算）。改用 gas-station 链路时为 91.52°C、51,010 mm，涉及链路机制改变，不能视为同条件的单纯布局收益。（PDF p.5，§IV-A，Fig.4。）[原文](https://www.date-conference.com/proceedings-archive/2021/pdf/1516.pdf#page=5)
- **CPU-DRAM**：原始布局 115.94°C、67,686 mm；Compact 113.54°C、100,864 mm；TAP repeaterless 为 94.89°C、216,064 mm，gas-station 为 93.89°C、138,956 mm。另行调节 CPU 功率、按 85°C 约束计算 TDP 时，原始布局的 400 W 提升到 TAP repeaterless 布局的 550 W。**150 W 是 TDP 包络收益，不是固定功率下的节能量**。（PDF p.5，§IV-B，Fig.5 见 p.6。）[原文](https://www.date-conference.com/proceedings-archive/2021/pdf/1516.pdf#page=5)

### ATPlace2.5D（ICCAD 2024）：确认指标是 TWL，并保留百分比口径

实验为作者构建的 10 个 UCIe 接口案例、6–61 个 dies；Linux/Xeon 2.10 GHz、最多 80 核、128 GB RAM，算法上限 12 小时。作者适配 TAP 开源代码并沿用其超参数；未比较 RL，§4.3 说明当时联系作者后仍未获得可执行文件。（PDF pp.6–7，§4.1、Table 3、§4.3。）[原文](https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf#page=6)

**热感知结果**：摘要及 §4.3.1 报告相对 TAP 的温度“5%”、线长“42%”改善及 **23×** 加速；Table 6 的 Avg. 行实际列为 TAP/ATPlace：RT **23×**、TWL **1.42×**、Tmax **1.05×**。因此引用应保留作者措辞及比值口径，不能把 1.42× 擅自写成“相对 TAP 降低 42%”：单个 1.42 比值换算到基线分母为约 29.6%，且平均比值与平均相对降幅并不等价。表中指标明确为 **TWL/m，不是 HPWL**。（PDF p.1 摘要；p.7，Table 6、§4.3.1。）[原文](https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf#page=7)

Table 6 的 61-die Case10：TAP 为 **300 min、141.44 m、103.9°C**，ATPlace 为 **23 min、105.23 m、104.0°C**，说明并非每例温度都降低；Case1 也从 93.0°C 升至 94.1°C。纯线长优化属于另一协议，Table 7 的 TAP/ATPlace 平均比值是 RT 4.6×、TWL 3.43×，不可与热感知结果混用。[Table 6–7，PDF p.7](https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf#page=7)

**热模型单独计时**：每例 5 个随机布局训练、另 10 个测试；Table 5 平均 MAE **1.16°C**、MAPE **3.23%**、相关系数 **0.987**、相对 HotSpot **2575×**。这不是布局端到端加速。§4.3.4 指出模型构建约占所分析运行时间的 60%–80%，Fig.7 包含数据生成、训练、初始化、优化和合法化。（PDF pp.6–8，§4.2、Table 5、§4.3.4。）[原文](https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf#page=7)

### RLPlanner（arXiv:2312.16895v2，2024-01-16）：20.28% 属于 RND 综合目标

**热模型**：2000 个合成系统，以 HotSpot 为参照；Table II 的 MAE 为 **0.2523 K**，MAPE **0.0726%**，单次推理 **0.1012 s vs 12.8976 s**，表列 **127×**。摘要写作“±0.25 K”及“over 120x”；MAE 是平均绝对误差，不能把该写法解读为逐例误差上界或置信区间。（PDF p.2，Table II、§III-A；摘要 p.1。）[原文](https://arxiv.org/pdf/2312.16895v2#page=2)

**布局目标**：§III-B 明确为 **RLPlanner with RND**，在 3 个公开 benchmark 加 5 个合成系统、共 8 例上，相对 TAP+HotSpot 的综合目标平均改善 **20.28%**，相对 TAP+快速热模型为 **9.25%**，在相近或更少运行时间内获得。奖励同时包含线长项及超过温度阈值的惩罚；这些比例不是单独线长或温度降低比例。表脚注将快速模型 TAP 的预算对齐到 RLPlanner **600 epochs 的训练时间**，不能称为训练后推理与 SA 的等时比较。（PDF p.2，§II-C、Tables I/III、§III-B。）[HTML 原文及表格](https://arxiv.org/html/2312.16895v2#S3.SS2)

Table I 的 Multi-GPU 示例：RLPlanner(RND) 为 **104,636 mm、91.85°C、20,380 s**；TAP+HotSpot 为 **124,639 mm、91.68°C、35,211 s**。综合收益不意味着所有分项同时改善；此例 RND 的温度略高。热模型替换与策略优化同时发生，因此 20.28% 不能全部归因于“RL 相比 SA”的纯算法效应。[PDF p.2，Table I](https://arxiv.org/pdf/2312.16895v2#page=2)

### 阶段偏好的证据边界

循环 RL+SA 论文的**原始摘要已核验**：作者报告 RL 提供 SA 的较好初始化，结合训练后粗解生成与后续启发式改进。这里只采用摘要层面的定性结论，不引用未核实的表格数字；任务是 IC placement，不能直接当作热感知 2.5D 的阶段规律。[原始摘要，arXiv:2011.07577](https://arxiv.org/abs/2011.07577)

TILOS 条目本次仅核验**当前原始摘要**：arXiv:2302.11014 已更新至 2026-03-10 的 v3，标题为 *An Updated Assessment of Reinforcement Learning for Macro Placement*，研究 CT、加强 SA 与其他宏布局器。附件中的 ICCAD04“15/17、16/17”未逐表复核，本文件不采用；该研究也不是 2.5D 热布局的统一三方对比。[当前版本与摘要](https://arxiv.org/abs/2302.11014v3)

上述材料支持在各自协议内比较方法、模型和目标权衡，**没有建立普适的固定算法阶段顺序**。ATPlace 的自身迭代曲线（p.8，Fig.8）也不能证明不同算法之间的阶段优劣。对本项目而言，把阶段与状态作为条件、通过统一 evaluator 和预算学习算法偏好，是可检验的研究假设；它与本地实测结果应分别陈述。[ATPlace Fig.8](https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf#page=8)
