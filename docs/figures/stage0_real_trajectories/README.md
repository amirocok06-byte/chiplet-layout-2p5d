# 真实 Case1–10 的三算法五检查点

用浏览器打开本目录的 [index.html](index.html)，按 Case 展开 SA、Analytical、
RL，即可查看五张三联图、3×5 状态流、参数曲线和三算法指标曲线。
每张快照同时显示 connectivity priority、空间热代理、HPWL 贡献场。

## 已固定的实验

- 数据：2-Benchmarks/external/ATPlace2.5D/Case1..Case10；
  6–61 个硬矩形芯粒，保留全部 53,900 条两 pin 网和有功耗的 DUMMY 芯粒。
- 初始布局：seed=20261024 的板内确定性散点，三算法逐 case 共享完全相同初始坐标。
- 每个算法：800 次候选迭代，记录 0、200、400、600、800，共五个检查点。
- 算法：本项目可审计的 Metropolis SA、平滑目标投影梯度、NumPy 在线 REINFORCE。
  RL 的奖励、权重范数、更新量和初末权重 hash 均有记录。
- 原始数组：256×256；显示宏网格：16×16；PNG：300 DPI，并导出 PDF/SVG。
- 每个 Case 的所有算法、所有检查点共用同一通道色阶。HPWL 使用 SymLogNorm，
  热代理使用 gamma=0.6 的 PowerNorm。机器精度级 HPWL 残差只在显示时掩码，
  不改动原始数组。浮点显示零容差为 64 × float64 epsilon × 共享最大绝对值。

每个 Case 计划输出 23 张图：15 张检查点三联图、3 张状态流、
3 张参数走势、1 张指标对比、1 张最终−初始差分。10 个 Case 共 230 张。

## 文件读法

| 文件 | 内容 |
|---|---|
| CaseX/algorithm/checkpoint_NNNN.png/pdf/svg | 第 NNNN 次迭代的三张 map |
| CaseX/algorithm/checkpoint_NNNN.json | 该时刻的真实坐标、指标与参数 |
| CaseX/algorithm/state_flow.png | 三通道 × 五检查点 |
| CaseX/algorithm/parameter_trends.png | 每次迭代的内部参数走势 |
| CaseX/algorithm/trace.jsonl.gz | 含初始状态的 801 条完整迭代记录 |
| CaseX/algorithm/summary.json | 输入和源码 SHA-256、预算、耗时、调用数与合法性 |
| CaseX/algorithm/raw/*.npz | 可再生的数组，不纳入 Git |
| CaseX/metric_trends.png | 三算法完整目标、HPWL、热代理与重叠率走势 |
| CaseX/algorithm_effects.png | 三算法最终−初始差分，每通道共用对称色阶 |
| CaseX/display_scales.json | 显示色阶与归一化规则 |
| *.alignment.json / *.collision.json / *.qa.json | 每个图的对齐和 PDF 校验记录 |

快照保存当前迭代，拒绝的动作不会强行造成可视化变化；采集后继续使用同一
优化器状态，并不在每个检查点重新初始化。

## 解释边界

1. 第一通道是网表连通性 priority，不是 benchmark 中不存在的 precedence
   约束。数值等于 incident-pin count / 最大 count，保留并列值；它随芯粒
   位置迁移，用于诊断，不是新增的待优化 order 损失。
2. 热场是未经 HotSpot 校准的 Gaussian 空间代理，其变化来自布局和功耗；
   不把颜色当作摄氏温度，也不借此声称复现 TAP、ATPlace 或 RLPlanner。
3. 五检查点是有限预算运行结束，不等同于收敛或最优。全量运行报告实际合法性，
   最终仍重叠的运行保留，并显示在总览表中。
4. Case4 的公开 ATPlace wrapper 板尺寸为 37×37 mm，而芯粒总面积 2064 mm²，
   利用率为 150.77%，可由面积直接证明不可行。本轮保留该固定配置作输入诊断，
   不计入合法布局偏好/性能排名。MaskPlace 的 Case4.intpsize 为 59 mm，
   说明不同公开包的封装定义有差异，后续修改应作为显式新版本，而非悄悄替换。
5. 三算法都是 800 次候选、801 次目标评价；解析法另有 800 次梯度计算，
   RL 的在线训练计入运行。单 seed、等迭代轨迹用于观察机制，不是同时间
   预算下的算法优劣证据；性能结论需要独立的多 seed、统一 wall-clock 对照。
6. 优化目标使用 penetration_penalty（轴向穿透惩罚），图表中的
   overlap_fraction 则按矩形的精确交集计算；二者在大小不同的矩形相互包含
   时会不同。完整日志同时保存两项，合法性使用真实几何结果。

## 复现

从可视化 worktree 根目录运行，Python 环境需要安装本项目的 viz 依赖：

    $env:PYTHONPATH = "src"
    python -m chiplet_layout_experiment.trajectory_run --data-root "<冻结输入根目录>" --output docs/figures/stage0_real_trajectories --steps 800 --seed 20261024 --revision "<source revision>"
    python scripts/render_trajectory_runs.py docs/figures/stage0_real_trajectories

设置 CHIPLET_FIGURE_QA_TOOLS 为已安装科研绘图校验工具的 scripts 目录，
会在导出前后分别执行面板对齐、PDF 字体与碰撞检查；无需设置即可复现普通图。
最终审核结果见 bundle_audit.json。若 raw NPZ 缺失，先运行轨迹采集命令再绘图。
