# Stage 0 数据集与源码冻结记录

**冻结日期：2026-10-24**
**适用分支：`codex/stage0-visualization`**

本记录回答“首轮运行到底使用哪一份数据、哪一版源码”。它把数据归档、算法源码和本项目的可观测轨迹分开，避免把同一个 Case 的副本当成独立样本，也避免把论文结果图误当成可执行算法。

## 1. 首轮共同输入

首轮共同布局输入固定为：

```text
F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/
  2-Benchmarks/external/ATPlace2.5D/Case1 ... Case10/
```

每个 Case 使用同一目录中的：

```text
<Case>.blocks
<Case>.nets
<Case>.pl
<Case>.power
WL-driven.json
Thermal-aware.json   # 仅在热感知配置实验中使用
```

选择这份归档的理由是：它与本地 `4-开源源码/ATPlace_pub/cases` 逐文件一致，并且包含 ATPlace 官方公开 wrapper 所需的两类配置。Case 的 interposer 尺寸使用 ATPlace 公共 wrapper 中的 case-specific 定义；不能从 `.pl` 中的全零初始坐标推断板尺寸。

`2-Benchmarks/cases/Case1...Case10` 是同一批设计的工作副本，不作为额外独立样本。2026-10-24 完整复核了 50 个共有文件：原始 SHA-256 差异仅来自 LF/CRLF 换行，规范化后 50/50 字节内容一致，结构化解析也一致。external 额外包含 10 份 Thermal-aware.json；与源码 cases 的 60/60 文件完全一致。上轮仅据哈希不同推断数据不同的结论已在此修正。

## 2. `5-额外Benchmark` 的边界

以下数据不进入首轮布局算法主 benchmark：

| 数据 | 固定版本 | 首轮用途 | 不纳入的原因 |
|---|---|---|---|
| `ThermalBench` | `75566fa8a213a5fd61bb89d33a25af258d2a4ae8` | 后续热场/热代理模型验证 | 其 S2–S5 是 64×64 热场学习任务，不是 Case1–10 的布局迭代输入 |
| `thermo-3dic-surrogates` | `0c6d2fb391893b8826fdbde186cbbc95b119f827` | 后续 3D-ICE/热代理模型对照 | 面向 3D 几何与温度代理模型，输入 schema、网格和物理边界不同 |

这两个仓库可供后续热模型校准参考。目前本地仅有源码与说明，ThermalBench 的实际 .mat 数据、thermo-3dic-surrogates 的生成 .npz 数据均未下载，不能称为已经加载的实验数据集。

## 3. 算法源码版本与角色

截图中的五个地址与本地副本对应如下：

| 角色 | 地址 | 本地固定 commit | Stage 0 边界 |
|---|---|---|---|
| SA / 热感知布局 | [TAP-2.5D](https://github.com/bu-icsg/TAP-2.5D) | `3a3f4e5a4d019b2d9cda1f72d8d1755238845dbb` | 作为 SA 的源码和参数语义参考；正式 HotSpot/CPLEX 运行另行锁定环境 |
| 解析式布局 | [ATPlace_pub](https://github.com/Brilight/ATPlace_pub) | `9e8de02d4a5e927c233b406b664d15051e056cb1` | 作为解析式 wrapper、Bookshelf/JSON 配置和输出格式参考；核心 PlaceFlow 加密，不能声称内部迭代参数可审计 |
| RL 布局 | [RLPlanner](https://github.com/weiweihook/RLPlanner) | `a75abf0b167a316dc5e2dd8b8d3d926114039271` | 作为 RL/快速热模型的配置和论文结果参考；当前归档公开的是配置与结果资源，未形成可直接运行的训练/布局源码臂 |
| Mask 热模型参考 | [MaskPlace2.5D](https://github.com/xunk33333/MaskPlace2.5D) | `897593e62da057affeffa611384068697e9cbed4` | 只用于 mask/紧凑热模型参考；README 明确说明核心 placement 代码未公开 |
| 综合布局参考 | [STAMP2.5D](https://github.com/MicroSystemsDesignLab/STAMP2.5D) | `f0057520c10292cdf8973fc282c0fbc4bd55a6e5` | 暂不纳入首轮比较；其流程依赖 ANSYS、SSH/Flask 服务和许可证，不能与 CPU-only Stage 0 同预算运行 |

原 SAWorker/AnalyticalWorker/RLCompatibleWorker 保留为协议 smoke 接口。2026-10-24 新增独立的连续轨迹模块：Metropolis SA、投影梯度解析法、NumPy 在线 REINFORCE；三者具备实际迭代和参数更新，使用真实 Case1–10，仍明确标为本地算法族实现。具体运行和五检查点定义见 five_checkpoint_trajectory_design.md；不把它们标成 TAP、ATPlace 或 RLPlanner 官方复现。

## 4. 迭代轨迹的可观测性冻结

每个 macro-step 都必须保存：

1. 输入/输出布局 checksum、三张 map、标量 HPWL、热指标和合法性；
2. 算法名、源码 commit、配置 ID、seed、wall-clock、峰值内存和 evaluator 调用次数；
3. 算法公开可见的参数变化，例如 SA 温度/接受率、解析法阶段与学习率、RL 的 action/reward/value 统计；
4. 如果某个外部实现因加密或源码缺失无法暴露这些字段，记录 `unobservable`，不能用推测值补齐。

这里的“一步”固定为一次 macro-step，而不是外部算法内部任意一次循环。每个 Agent 在一次 macro-step 内运行固定的候选预算，然后由同一个 `PhysicsEvaluator` 评价；内部 iteration 作为轨迹字段保存，不能改变跨 Agent 的预算定义。

## 5. 当前可视化运行的含义

`docs/figures/stage0_synthetic_smoke/` 仍是协议 smoke 输出，不是上述 Case1–10 的算法结果。它用于验证 `[order, thermal, hpwl]` 三通道、状态流和候选差分的绘图链路。方案 A 的输出规范已经固定为：

- 原始 map 保留 `256×256`，`interpolation="nearest"`；
- 默认输出 `300 dpi`；
- 覆盖可见的 `16×16` 显示宏网格，每个显示格对应 `16×16` 原始像素；
- 坐标轴显示原始像素边界，不把显示网格误称为原始 map 分辨率。

真实 Case1–10 的连续轨迹单独保存在 docs/figures/stage0_real_trajectories，以 Case/algorithm 组织五份快照、全量参数轨迹和追溯记录。thermal 使用明确标注的空间代理，未完成 HotSpot 校准。
