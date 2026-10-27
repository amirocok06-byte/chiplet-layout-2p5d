# 轻量 2D CNN 与 LinUCB：RTX 4060 8GB 实验设计

2026-10-28 已完成本机首轮 GPU 验收：
[完整结果](figures/stage1_e2_8gb_pilot/RESULTS.md) ·
[结果解释](figures/stage1_e2_8gb_pilot/INTERPRETATION.md)。
CNN与固定策略在本轮留出集打平，LinUCB较差；实现可运行与方法增益分别评价。

## 决策链路

研究问题：在相同布局状态和 65 次评价的宏动作预算下，空间状态是否能帮助
选择更合适的算法配置，以及选择后的反馈是否进一步改善决策？

`四通道数值 map → CNN 布局特征 → LinUCB 选 arm → 算法专家运行宏动作 → evaluator 收益 → 更新所选 arm`

三位专家仍为 SA、Analytical、RL。六个 arm 是专家和离散参数配置的组合：
`analytical-large / analytical-small / rl-explore / rl-local / sa-explore / sa-local`。
CNN 是共享编码器与离线收益预测器，LinUCB 是协调器的选择机制。
E2a 的动作 mask 固定为全局允许区域；本轮尚不学习局部 mask 选择。
legality 输入通道用于观察几何状态，精确动作约束仍由 worker/evaluator 执行。

## LinUCB 的具体含义

LinUCB = Linear Upper Confidence Bound（线性置信上界）。对每个 arm a，
以当前状态特征 x 为输入，拟合一个线性收益模型：

`A_a = λI + Σ x_i x_iᵀ, b_a = Σ r_i x_i, θ_a = A_a⁻¹ b_a`。

选择评分为 `p_a(x) = xᵀθ_a + α sqrt(xᵀA_a⁻¹x)`。
前项为预测收益，后项为探索奖励；α 越大越愿意尝试缺乏证据的 arm。
运行选中的宏动作并得到 r 后，仅更新所选 arm 的 A、b。

例如：解析小步长预测收益 0.08、探索奖励 0.01；SA 局部预测收益 0.06、
探索奖励 0.04，则 LinUCB 选择 SA（0.10 > 0.09）。这些数字仅用于教学。
训练期可以使用完整六臂标签；测试期间仅将所选 arm 的收益提供给在线更新。
测试 oracle 的其余标签只供评估使用。

CNN 特征上的条件线性关系是一项待检验的建模假设。LinUCB 原始理论的
假设不能自动转移成 CNN 特征、相关布局状态或当前有限样本上的保证。

## CNN 首版结构

| 层 | 输出（省略 batch） | 用途 |
| --- | --- | --- |
| 四通道输入 | 4×64×64 | priority、thermal_proxy、HPWL、legality |
| Conv 3×3, stride 2, padding 1 + ReLU | 16×32×32 | 学习局部跨通道关系 |
| Conv 3×3, stride 2, padding 1 + ReLU | 32×16×16 | 扩大局部感受范围 |
| AdaptiveAvgPool2d(4×4) | 32×4×4 | 保留粗粒度位置 |
| Flatten + Linear + ReLU | 32 | 布局 embedding |
| 拼接全局标量 | 36 | 32 embedding + 4 scalar |
| Linear | 6 | 每个 arm 的宏动作收益预测 |

总参数量为 21,870。LinUCB 上下文为 32 维 embedding、4 个标准化标量和
1 个截距，共 37 维；其六组 37×37 矩阵在 CPU 上更新。

保留 4×4 分区能表达中心/边缘差别。空间特征仍然是学习到的统计关联；
不能将激活图直接解释成物理因果或算法移动计划。

## 数据与训练

阶段 0 的 61 个合法态、9 个 Case 是 E2a 首轮数据；2,322 条 action 记录
包含重复 seed 和恢复态，不能当作 2,322 个独立训练样本。
每个 state 的六臂目标取三个 seed 的平均收益。首版只预测宏动作收益，
状态变化图预测留作后续扩展。
沿用阶段 0 已保存的 reward：合法输入且最终合法时，
`r = -ΔHPWL/hpwl_scale - 0.3×Δthermal_proxy/thermal_scale`；
最终非法时为 -1。这里 Δ 对应统一后处理后的返回布局，原始 worker 改善
与公共合法化贡献已在阶段 0 分列，因此本轮结论针对这一完整宏动作接口。

地图由保存的坐标重建。256×256 的展示地图保留；模型输入采用同一投影
函数在 64×64 上重新栅格化。这与对 256 图直接平均池化不是同一操作，
需在报告中注明，避免误报精确分辨率。

按 Case 切分，Case4 版本始终属于同组。所有地图、标量与 reward 标准化
仅由 train Case 估计，validation 选择 checkpoint，test 只用于最终评价。
CNN 用六臂标准化 reward 的 MSE 监督，预测后还原到原 reward 尺度再选 arm。
首版 AdamW：lr=0.001，weight_decay=0.0001，batch=8，最多 100 epochs，
固定随机种子；validation MSE 选最佳 checkpoint。小样本首版使用 FP32。
超参数搜索和训练成本单独报告，后续用 Case 留一验证检查结论稳定性。

CNN 完成后冻结。CNN greedy 使用其六输出 head；LinUCB 使用冻结 embedding
另拟合六个线性收益模型。LinUCB 的 α 在 validation 选择，候选 {0, 0.1, 0.5, 1}，
λ=1。上下文按训练数据定标；在线更新使用同一个训练期确定的全局 reward
尺度，保证六臂探索奖励在相同量纲。α=0 是对应的无探索对照。

## 本机 8GB 显存约束

2026-10-27 核验：RTX 4060，driver 560.94，驱动报告 CUDA 12.6，
物理总显存 8188 MiB。此处“8GB”指部署容量上限；无需填满显存。
系统为 i9-14900HX（24核/32线程）、32GB级RAM（操作系统报告
34,070,192,128 bytes，约31.73 GiB）；训练函数将PyTorch CPU线程数固定为2，
GPU batch固定为8，数据保留在CPU并逐batch传入单卡。

训练显式要求 CUDA。使用 `min(8192 MiB, physical_total) - 1024 MiB` 作为
PyTorch allocator 上限；实际可用显存较低时进一步缩小。对本机空闲条件下
上限为 7164 MiB，为 CUDA context、驱动和其他进程留出至少 1024 MiB。
allocator 限额只约束 PyTorch allocator，不是操作系统级全 GPU 隔离。

每次运行记录 GPU 名称、总/空闲显存、allocator 限额、batch、参数量、
峰值 allocated/reserved、训练时间以及版本。
每个 optimizer step 后另用 `mem_get_info` 采样整卡已用显存，记录
`sampled_device_peak_used_bytes`；这是包含其他进程的采样峰值，具有采样边界。
训练前配置限额，OOM 时终止
该运行并报告；若调整 batch，另立配置，不能悄悄改变对照预算。
不以占满显存或系统共享内存作为达标依据。正式测量需使用 NVML/nvidia-smi
同时观察整卡显存，PyTorch 峰值不等于包含所有 CUDA 外部分配的进程峰值。

算法 worker 的 65 次评价账本与 CNN 的训练显存属于两个不同资源边界，
需要分别报告。训练全部使用本机单卡，禁止用多卡结果冒充本机 8GB 结果。

## 对照与结论

固定最佳 arm、随机、标量 Ridge、CNN greedy、CNN+LinUCB 使用同一 Case
切分和 reward。CNN greedy 对比标量模型检验空间信息，CNN+LinUCB 对比
CNN greedy / α=0 检验探索。另加三通道与四通道消融检验 legality 的作用；
合法态 legality 通道多为零，主要增益应在恢复任务 E2b 中另外检验。

报告集合 top-1、regret、每 Case 收益和 action 成本；记录模型推理成本。
离线 replay 衡量冻结状态分布上的一步选择，不等同于多步在线布局闭环。
若当前 61 状态无法支持稳定结论，应增加独立 Case/状态，而非扩大网络。

## 执行入口

项目根目录中建立独立环境（Python 3.11+），CUDA 轮子来源显式固定：

```powershell
python -m venv --system-site-packages build/stage1-gpu-env
./build/stage1-gpu-env/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
./scripts/train_stage1_cnn.ps1 -Output build/stage1-cnn-first-run -Epochs 100 -Seed 11
```

预设 train=Case1/3/4/5/6、validation=Case7、test=Case8/9/10。
输出目录必须是新目录。`config.json` 保存切分与训练设置，`audit.json`
保存源码/标签 hash、环境和显存账本，`summary.json` 保存离线选择指标，
`decisions.jsonl` 保存逐状态 arm、收益、预算与历史运行成本，`cnn.pt`
保存权重、归一化参数和 arm 顺序。该单切分单 seed 运行用于实现验收；
正式结论还需多 seed、Case 分组的外层验证。

## 一手来源

- Li et al., WWW 2010, §3.1 / Algorithm 1：
  [LinUCB 原论文](https://www.schapire.net/papers/www10.pdf)。
- [PyTorch allocator 限额](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.set_per_process_memory_fraction.html)：
  限额按设备总显存比例设置，超过 allocator 限额触发 OOM。
- [PyTorch CUDA 内存说明](https://docs.pytorch.org/docs/main/notes/cuda.html)：
  allocated 与 reserved 具有不同统计边界。

上述文献支持算法定义和实现接口。CNN 架构、参数与本项目的实验对照是本项目
提出的可执行方案，其效果以随后实际留出实验为准。
