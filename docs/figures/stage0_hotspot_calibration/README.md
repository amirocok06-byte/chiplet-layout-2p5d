# 正式状态集的真实 HotSpot 校准

使用 TAP 本地 ELF executable，经 WSL `Ubuntu-24.04-G` 实际运行。固定配置：
[hotspot_stage0_block.config](../../../configs/hotspot_stage0_block.config)，SHA-256：
`30b3790b15be2b1eb5660e1a97b5302ab835a91f5afa3e78088aed838c21e5c7`。

|项|结果|
|---|---|
|成功物理模拟|27 个布局，9 个 Case，每 Case 3 个|
|校准训练|Case1/3/4/5/6/7/9，共 21 个布局|
|按 Case 留出|Case8、Case10，共 6 个布局|
|训练 MAE / RMSE|42.9523 / 51.6263 °C|
|留出 MAE / RMSE|58.9105 / 68.9954 °C|
|拟合斜率 / 截距|−5.02865 / 203.63429 °C|
|扫描时拒绝|30 个非法布局；达到每 Case 样本上限即停止扫描|

**校准未达可用精度。** 斜率为负且跨 Case 误差很大，不将代理改标为温度，也不把该映射用于算法热收益优化。
本次是平面硅 block 参考模型，空白区域用零功耗硅填充；不是 TAP 的完整 interposer/ubump 分层封装模型。
温度只代表该固定参考模型的输出。Case2 没有通过合法化的布局，未产生热样本。

模型为 `T_peak/C = slope × thermal_proxy_peak + intercept`，只用训练 Case 拟合，未在留出集上重新拟合。
布局按文件名排序，选择每 Case 前三个成功的严格合法状态，未按模拟温度挑选。
由于 HotSpot 拒绝任意正交集，它的输入筛选比带数值容差的公共合法化更严格；浮点狭缝也会保留为失败。

[calibration.json](calibration.json) 含全部系数、逐 Case 留出误差、split、样本来源与 hash；
[collection.json](collection.json) 保留成功与拒绝记录；各 `run_*` 目录保留 flp、功耗、config、命令和 steady.k。
早期 19 次探索模拟已永久保存到 [stage0_hotspot_pilot](../stage0_hotspot_pilot)，不再依赖临时目录。
完整模型边界见 [热校准文档](../../hotspot_calibration.md)。

```powershell
python scripts/calibrate_hotspot.py --states-root docs/figures/stage0_same_state --executable 'F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/4-开源源码/TAP-2.5D/util/hotspot' --config configs/hotspot_stage0_block.config --output build/hotspot-reproduction --wsl-distribution Ubuntu-24.04-G --layouts-per-case 3 --held-out Case8 --held-out Case10
```
