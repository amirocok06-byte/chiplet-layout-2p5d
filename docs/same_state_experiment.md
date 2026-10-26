# 阶段 0：同状态分叉与可行性分层（2026-10-25）

## 实现位置与已有支线复用

实现工作树为 `.worktrees/stage0-visualization`，分支 `codex/stage0-visualization`。
`codex/stage0-core` 的 `sampling.py` paired input/output checksum、baseline 与 delta 设计已在此分支中，继续保留；
本轮从 `codex/stage0-baselines` 引入 `stage0_summary.py` 及其测试，保持旧记录可汇总。
旧 smoke workers 不承担真实算法对照。新 `counterfactual.py` 使用连续 SA、投影解析法和在线 REINFORCE。

## 状态与动作

- 状态 `s_t` 是确定的物理问题、当前所有芯粒坐标、固定目标归一化常数和 evaluator 版本。
  problem checksum 包含板尺寸、芯粒尺寸/功耗/引脚/网络、归一化常数；state ID 再加入精确坐标 checksum。
- 三张 256×256 map 由 `TracedProblem.maps` 确定性生成。priority 为算法无关连通性优先级，thermal 为空间代理，HPWL 为真实 pin bounding-box 贡献场。
- 首轮六个有限动作：SA 小/大位移与温度、解析法小/大学习率、RL 小/大位移与策略学习率。
  配置参数实际传入优化器，不再以不同配置名调用同一段恒等逻辑。
- 每个 action 的“一步”是一个宏动作：默认 64 次候选 + 1 次起始评价，共 65 次目标评价。
  所有接受/拒绝候选均占预算；解析法的梯度另外计数，RL 的训练另外计数。
- 相同 seed 不代表不同算法使用相同随机数序列，只表示重复试验编号；同状态、同 evaluator、同预算才是配对基础。

## 几何、合法化与热校准

Case4 原 37×37 mm 仍保留，不覆盖冻结数据；替代版本单独命名、记录依据和 hash。
物理芯粒与引脚尺寸不变，改变 interposer 会同时改变问题难度，禁止与原板混为算法改进。
详见 [几何版本与公共合法化](geometry_legalization.md)。

同一个 `legalize(problem, positions)` 处理每个算法的输出，记录状态、方法、几何检查次数、位移和耗时。
合法输入无法恢复时回退原合法输入，并显式记录 rollback。
原始算法效果、公共合法化效果和完整动作效果各自保留，不能把公共合法化算作算法自身的优化能力。
公共合法化有相同确定性工作上限，但实际开销会依布局而异，单独报告，不能称为相同总算力。

HotSpot 使用真实 executable、米制 floorplan、W 功耗、物理 package config 与温度输出解析；
合法布局才可用作校准样本。校准必须按 Case/布局分组留出，训练与测试不得共享布局。
代理拟合温度不等于优化算法已使用物理温度，是否实际完成校准及误差以
[HotSpot 校准说明](hotspot_calibration.md) 和校准 JSON 为准。

## 分账与学习标签

`phase=feasibility_recovery`：输入非法，统计最终合法率、恢复率、重叠变化与运行成本。
HPWL/thermal 可保留诊断，但不进入合法状态优化均值。

`phase=legal_optimization`：输入合法，只对输出也合法的记录统计 HPWL/thermal 优化。
同时报告原始合法率、后处理合法率、候选合法率、当前 iterate 合法率和 rollback，防止最终合法率掩盖算法频繁产生非法输出。

逐动作字段：

- `raw_delta_metrics = raw(s_{t+1}) − s_t`。
- `legalization_delta_metrics = returned(s_{t+1}) − raw(s_{t+1})`。
- `delta_metrics = returned(s_{t+1}) − s_t`，三者逐项可加。
- `runtime_seconds/evaluation_seconds`：宏动作算法总耗时及其中目标评价耗时。
- `legalization_seconds/geometry_checks`：公共后处理成本。
- `evaluator_calls`：严格预算；共享 baseline 和 postprocess evaluator calls 分列。
- `rss_before_mb/rss_after_mb`：共享进程瞬时 RSS，不能当每算法峰值；等时间/峰值内存研究需另用隔离子进程协议。

Bandit reward 分阶段固定：恢复阶段为合法性恢复指示 + overlap 下降；
合法优化阶段为 `−ΔHPWL/hpwl_scale − 0.3 Δthermal_proxy/thermal_scale`。
同时保留完整向量，后续更改研究权重时可以重算；不同 phase reward 不直接混合排名。
当前是 full-information 全臂标签，不把它冒称为已完成 Bandit 策略训练。

## 首轮控制实验

每 Case 使用三个来源算法，各取 0/32/64 步中间状态，坐标精确去重；
对成功合法化的 raw anchor 再保留一个 legal companion，分布来源显式记录。
每个不同状态上运行六臂 × 三个 seed，默认每臂 65 次目标评价。
共同原始起点 seed=20261024，动作 seed=20261025/26/27。
预备轨迹、构建合法 companion、共享 baseline、算法调用与后处理分账。

原始 Case4 单独输出不可行诊断；主动作库使用另立的 Case4 几何版本。
全部 Case 的配置、状态、原始输出坐标、后处理输出坐标和 labels JSONL 都保存。
配置、seed、状态覆盖完整性与预算由 audit JSON 核验。

训练/验证必须按 Case 分组，Case4 不同版本属于同一组；同一 case 的近邻时间点不能随机分到训练和测试。
先比较同状态配对收益，再学习配置选择；来源算法分布、原始/合法 companion 分布及 phase 分别做消融。
三个 seed 仅为首轮机制与可复现性验证，配置优越性和跨数据集泛化仍需增加 Case 与 seed。

## 运行

在项目工作树中安装 `.[test,viz]` 后，用 PowerShell：

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m chiplet_layout_experiment.counterfactual_run `
  --data-root 'F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/2-Benchmarks/external/ATPlace2.5D' `
  --output build/same-state-reproduction-20261025
```

输出目录必须不存在；已有的正式结果不会被覆盖。更改输出目录名即可独立复现实验。

结果入口：[本轮同状态实验结果](figures/stage0_same_state/README.md)。

实际执行结论：[合法化控制、标签可识别性与热模型局限](figures/stage0_same_state/EXPERIMENT_FINDINGS.md)。
本轮产生 129 个状态、2,322 条标签；两项预算/回放审核通过。
公共合法化后的恢复奖励有 61 个状态打平，不能强制指定唯一最优算法；原始恢复向量仍保留。
正式 HotSpot 集合 27 个成功模拟，按 Case 留出 MAE=58.9105 °C，因此不将当前代理改标温度。
