# 同状态实验的实际结论与下一步使用边界

## 已执行

- 10 个 Case（Case4 使用单列的 59 mm 替代版本），129 个不同状态。
- 每状态 6 个数值配置 × 3 个 seed，合计 2,322 条标签。
- 每动作 65 次目标评价，合计 150,930 次；另有 129 次共享 baseline 和 2,322 次后处理评价。
- 68 个恢复状态：1,224 条标签；61 个合法优化状态：1,098 条标签。
- [budget_audit.json](budget_audit.json) 和 [replay_audit.json](replay_audit.json) 均通过。
  后者从保存坐标重新计算所有原始/返回指标，核验 delta 可加性与完整配置×seed 网格。

## 合法化确实改变了比较对象

原始算法最终合法 542/2,322；公共合法化/回退后合法 2,196/2,322。
Case2 的 126 条记录全部未恢复合法，保留为 `search_exhausted`，不能称为不可行证明。
其余 Case 后处理全部合法；这主要证明了公共合法化可恢复可行性，不是三种算法原生合法率都很高。

额外执行了每个状态上的 **legalizer-only 控制**，它不运行任何算法。
这项控制另外占 129 次评价，不计入上述算法预算；记录中同时保存控制耗时与几何检查数。
各动作相对控制的 HPWL/thermal 差异见
[preference_diagnostics.json](preference_diagnostics.json) 中
`valid_output_minus_legalizer_only_by_arm`。

## 哪些标签真的能教 Bandit 选择

|阶段|不同状态|后处理奖励有动作均值差异|原始奖励有动作均值差异|
|---|---:|---:|---:|
|恢复可行性|68|7|68|
|合法状态优化|61|61|61|

判别阈值为六臂三次重复的 mean reward 极差 > 1e-10，仅是数值可识别性检查，不是显著性检验。
后处理恢复奖励在 61 个状态上打平，因为共同合法化使输出都合法且 overlap=0。
在这类状态上强行指定“最好算法”会制造伪标签。

第一轮训练建议固定两个不同任务：

1. 恢复任务使用 `raw_delta_metrics`、原始合法率与运行成本，公共合法化作为独立兜底；
   或保留完整 action+legalizer 目标，但对打平状态保留多最优动作/不生成唯一 winner。
2. 合法优化任务使用合法到合法的收益，同时对照 legalizer-only；不能将原始非法状态因重叠而变短的 HPWL 当作优化成功。

本次未更改已有标签定义：原始向量、后处理向量和 scalar reward 都完整保留。
未来采用其他 reward 时必须另命名目标版本，重新核验评价集合，禁止悄悄覆写当前结果。
CNN/Bandit 训练与独立测试尚未执行；同 Case 所有状态、seed 和几何版本必须在同一 split 内。

## 热物理校准结论

真实 HotSpot block 参考模型已成功运行，正式集合采样 27 个严格合法布局。
训练 21 个布局；按 Case 留出 Case8/Case10 共 6 个布局，MAE=58.9105 °C、RMSE=68.9954 °C，
拟合斜率 −5.02865。见 [完整校准结果](../stage0_hotspot_calibration/README.md)。
因此热标签仍是空间代理，不能标 °C；当前校准不满足物理温度优化的使用条件。
下一项热建模工作应补齐实际层叠材料/封装参数，并检查物理单位、空间扩散尺度和功耗总量对模型的影响。
本轮只接通和核验平面 block 参考协议，没有冒称完成真实 2.5D 封装热模型复现。

## 可复现性与成本

轨迹与标签的运行源码以 manifest 中 SHA-256 为准；source_commit 是启动时工作树的基准提交。
运行期间热适配器在独立任务中完善，该文件未参与布局标签计算；其后续真实热模拟有独立 adapter hash。
`trajectory_*`、`counterfactual*`、geometry/legalization 等实际布局执行文件的 hash 均可与当前源码核对。

时间为本机描述性测量，目标评价数严格相等，但梯度、训练、合法化开销不同。
采集时后台验证任务可能造成调度噪声，当前 RSS 是共享进程瞬时值；
此轮不据此发布运行时间/峰值内存胜出结论。等时间与隔离峰值成本实验应另立协议。

入口：[逐 Case/配置指标表](README.md)、[CSV](summary.csv)、[分层 JSON](summary.json)。

## 改动归属

代码与结果均在仓库 `amirocok06-byte/chiplet-layout-2p5d` 的 `codex/stage0-visualization`。
继承 `codex/stage0-core` 的 paired sampling 接口；从 `codex/stage0-baselines` 迁入
`stage0_summary.py` 及其测试。两个旧分支保留原提交，不重写历史。
