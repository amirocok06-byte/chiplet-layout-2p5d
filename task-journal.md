 # Task Journal / 任务日志

## 长期同步规定

1. 每次源码、配置、测试、文档、图表或 PDF 修改，都必须同步更新本地目录 `F:/GAME/Chiplet-Layout-2.5D` 与 GitHub 仓库 `amirocok06-byte/chiplet-layout-2p5d`。
2. 同步顺序为：本地修改 → 运行相关测试/构建 → 检查 diff → Git commit → push 到 GitHub → 核验远端提交。
3. 每次部署完成后，必须再次思考并检查是否需要同步修改相关源码、实验配置、测试、README、论文源文件、PDF、图表或 GitHub 内容；如果需要，必须补充修改、测试、提交和推送。
4. 实验结果、生成图和 PDF 必须带有可追溯的版本、配置、seed、evaluator 和预算信息；禁止只更新生成文件而不更新源文件。

## 本次记录（2026-10-20）

- 将三类 Agent 明确为 `SA-Agent`、`Analytical-Agent`、`RL-Agent`；`order/thermal/HPWL` 作为共享 map 通道，`PhysicsEvaluator` 作为唯一裁判。
- 增加 HPWL 贡献场、order rank 场和 order transition 场的形式化定义。
- 增加 `map_projector.py` 与三个状态/状态流/算法效果绘图脚本，并为 map projector 增加测试。
- 将公平预算、一步定义、可视化元数据要求和同步规定写入实验设计文档。

## 本次记录（2026-10-21）

- 将实验方案冻结为 v1.0：三类 Agent 固定为 `SA-Agent`、`Analytical-Agent`、`RL-Agent`；`PhysicsEvaluator` 固定为统一裁判；Contextual Bandit 负责选择 `(agent, config, mask, macro_budget)`。
- 将共享 `order` map 的语义修正为算法无关的 priority/precedence map；运行时 placement sequence 不再作为共享输入，只保留为算法轨迹诊断。
- 固定第一版主实验为有限 arm 配置库 + 2D CNN + Contextual Bandit，GRU/ConvLSTM/局部注意力和连续超参数优化只能作为后续递进消融。
- 新增 `docs/experimental_design_v1_freeze.md`，并同步修订状态设计、README、CONTRIBUTING、LaTeX/PDF。
- 从本次起，除非出现实现 bug、协议不可运行、全臂采样证明定义不可识别，或消融结果提供明确反证，否则不得修改上述核心定义；任何修改仍须按“本地修改 → 测试/构建 → diff → commit → push → 远端核验”同步。

## 本次记录（2026-10-24）

- 确认方案 A：保留 `256×256` 原始 map，使用 nearest 显示，叠加 `16×16` 宏网格（每格对应 `16×16` 原始像素），默认输出 300 DPI，并显示原始像素坐标。
- 固定首轮共同输入为 `2-Benchmarks/external/ATPlace2.5D/Case1..Case10`；`2-Benchmarks/cases` 与其哈希不同，只作独立副本校验，不能混合统计；`5-额外Benchmark/ThermalBench` 和 `thermo-3dic-surrogates` 暂作热模型验证集。
- 固定外部源码 commit 与运行边界，详见 `docs/stage0_data_source_freeze.md`。由于 ATPlace 核心加密、RLPlanner 本地归档缺少可直接运行的训练/布局源码、STAMP 依赖 ANSYS/SSH，当前三臂仍是本项目可审计的本地 worker，不宣称外部算法复现。
- 每次源码、配置、测试、文档、图表或 PDF 修改，继续执行本日志开头的本地同步、测试、commit、push 和远端核验规定；本次部署后也要复查相关文件是否需要同步。

## 本次记录（2026-10-24，五检查点真实轨迹）

- 使用冻结的 external/ATPlace2.5D Case1–10，新增连续 Metropolis SA、投影梯度和 NumPy 在线 REINFORCE。原 smoke worker 保留；新实现以本地算法族版本命名。
- 每个 Case/算法运行 800 次候选，保存 0/200/400/600/800 五检查点；每个检查点含三张 256×256 map，另保存全部 801 条参数与指标状态。
- 保留真实 percentage pin offset、毫米 HPWL、全部网及 DUMMY 功耗；priority 采用连通性定义并保留并列值，热场采用明确标注的空间代理。
- 显示采用浅色空白、分级 priority 与芯粒轮廓、热代理 PowerNorm、HPWL SymLogNorm；色阶跨算法/时刻固定，浮点残差只在显示时按机器精度掩码。
- 更正前条数据审计：50 个共有文件仅 LF/CRLF 换行不同，语义完全相同；external 多 10 份热配置。额外热学仓库尚无实际下载数据。
- 发现 Case4 wrapper 板为 37×37 mm，芯粒总面积 2064 mm²，面积利用率 150.77%，是不可行输入；保留它作为诊断，不作算法偏好结论。
- 新增总览页、五快照、状态流、指标/参数曲线、输入与源码 hash、调用/资源记录及每个 PDF 的对齐/字形/碰撞 QA；原 smoke 图同步更新。
- 部署后复查 README、数据冻结记录、五检查点设计、结果说明、绘图与轨迹代码、测试、生成图和 GitHub 可视化分支。主工作树原有 PDF 删除状态继续保留。
- 代码审查修正了不等矩形包含时的几何重叠报告：优化惩罚单列 penetration_penalty，overlap_fraction 按精确交集报告；新增包含测试并重新采集记录、同步相关图表。
- 最终验收：402 passed、7 skipped；30 条完整运行、24,030 条状态记录、150 张三联检查点、230 张 PNG/PDF/SVG；整包审核通过，230 份对齐/字体/碰撞校验无失败与警告。最终合法数 SA=1、Analytical=2、RL=3；其余保留为未达合法化的真实结果，不作偏好优劣结论。
