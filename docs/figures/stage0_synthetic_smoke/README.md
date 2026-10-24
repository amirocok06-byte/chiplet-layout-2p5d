# Stage 0 三图与状态流：synthetic smoke 输出

本目录的 PNG 只验证“统一布局 → 三张 map → 状态流/候选差分”的数据链可运行。输入由 `chiplet_layout_experiment.visual_demo.synthetic_smoke_layout()` 确定性构造，**不属于公开 benchmark，不是 TAP-2.5D、ATPlace 或 RLPlanner 的复现，也不能证明三类算法的偏好**。

## 三个图的读法

| 图 | 读取对象 | 可作出的结论 | 不可作出的结论 |
|---|---|---|---|
| `snapshot_maps.png` | 初始布局的 `[order, thermal, hpwl]` 三通道 | 投影的几何方向、通道尺度和数据格式正确 | 某算法优于另一算法 |
| `state_flow.png` | 一个预先声明的 macro-step 路径 `initial → analytical → SA → RL-compatible` | 后续模型将看到的前后空间状态格式 | 真实优化过程会固定采用此顺序 |
| `algorithm_effects.png` | 三个 worker 从**相同初始状态**出发的 map 差分 | 候选动作可以被表达为条件化状态变化 | 差分大小就是算法优越性或物理因果 |

三个 worker 目前是 local smoke implementations；`SA` 与 `analytical` 可出现相同的简化布局，`RL-compatible` 可保持原布局。这种相似/差异是接口验证信号，不能作为论文结果。

## 通道语义

- **order**：算法无关的 priority/precedence rank。芯粒 footprint 内为 `r_i/R_max`；空白固定为 `-1`，因此 rank=0（最低优先级）不会和背景混淆。它不是左右位置，也不是某个 worker 的 placement sequence。
- **thermal**：当前是芯粒功率密度的空间 proxy，而不是 HotSpot 温度场；用于验证 map 数据流。正式热结论需要高保真温度场校准。
- **hpwl**：每个 net 的加权 HPWL 在其 pin bounding box 中均匀分摊；像素和保持为当前加权 HPWL（栅格浮点误差除外）。

`raw/manifest.json` 是图的追溯记录：包含 source revision、map 格式、seed、macro budget、worker implementation、输入 checksum 和 evaluator 版本。

## 复现

在仓库根目录的 PowerShell 中运行：

```powershell
$env:PYTHONPATH = "src"
python -m chiplet_layout_experiment.visual_demo docs/figures/stage0_synthetic_smoke/raw --source-revision "<source-commit>"
python scripts/plot_snapshot_maps.py docs/figures/stage0_synthetic_smoke/raw/snapshot_maps.npz docs/figures/stage0_synthetic_smoke/snapshot_maps.png
python scripts/plot_state_flow.py docs/figures/stage0_synthetic_smoke/raw/state_flow.npz docs/figures/stage0_synthetic_smoke/state_flow.png
python scripts/plot_algorithm_effects.py docs/figures/stage0_synthetic_smoke/raw/algorithm_effects.npz docs/figures/stage0_synthetic_smoke/algorithm_effects.png --labels sa analytical rl-compatible
```

原始 `.npz` 是可再生的中间结果，不提交到 Git；PNG、此说明和 manifest 保留在仓库中。更换为真实 benchmark 后，必须新建有 case ID、配置、seed、evaluator fidelity、预算和源提交的输出目录，不能覆盖本目录。
