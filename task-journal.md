 # Task Journal / 任务日志

## 阶段 1 / E2a 启动（2026-10-27）

- 阶段 0 按冻结审计再次核验：129 个独立状态、2,322 条 action 标签、6 arm × 3 seed 完整网格、150,930 次算法评价、坐标/指标重放与四通道投影审核均通过；本阶段只读该数据。
- 固定 Contextual Bandit 的“一步”为 `a_t=(worker, configuration, wiremask, H_eval=65)` 宏动作：1 次起点评价加 64 次候选评价；SA/解析/RL 的内部循环均为不可跨算法直接比较的微步。时间和内存为共同停止上限及成本字段，不替代 evaluator 预算。
- 新增只读 E2 数据加载器：从阶段 0 的问题和坐标重建 `[priority, thermal_proxy, HPWL, legality]` 四通道，严格校验 `evaluator_calls=65`、完整 arm×seed 网格，并按 Case 切分，防止相邻 checkpoint 泄漏。
- 正式合法态标签已做只读验收：61 个状态、9 个 Case、6 个 arm、`4×64×64` 输入和统一 65 次预算。E2a 只处理 `legal_optimization`；恢复任务保留给 E2b。
- 本轮先实现固定/random/标量/CNN/LinUCB 的预注册对照接口；在完整验证、commit、push 与远端核验前，不声称已取得 CNN/Bandit 增益。

## 四通道状态与图例同步（2026-10-26）

- 按用户明确要求加入 M_overlap/legality，状态版本为 connectivity-priority-thermal-hpwl-legality-v2；三类算法 Agent 不随通道数变化。
- Priority 明确定义为 incident pin 次数/最大次数；背景-1、断连0、并列保留、重叠像素取最大q。新增通道手册，逐项说明输入、公式、单位、配色、色条、空间含义与推断边界。
- Legality 用精确 pair 交集面积覆盖加归一化边界越界投影，避免栅格同格假重叠；精确 evaluator 仍为合法性裁判。差分四列均为 after-before。
- 150个历史检查点重新投影为600张map，导出190幅四通道图；原坐标/指标与thermal、HPWL数组保持，历史三联图保留。129个同状态布局新增v2观察sidecar，新采状态默认声明v2。
- 网页增加完整公式/配色解释和逐芯粒priority数值表，PDF重编译为22页，并同步指定外部PDF与主目录文稿；此前主目录旧PDF删除状态由本次明确授权的新文稿更新替代，旧外部稿已备份在build目录。
- 整包审核通过：190图对齐/字体/碰撞无失败或警告；658 passed、7 skipped，差分修正后另行通过相关回归。三对四通道和priority消融写入方案，未声称已获得CNN/Bandit增益。

## 本次记录（2026-10-25，同状态分叉与物理评估）

- 在 `codex/stage0-visualization` 上继续实现；审计并保留 `codex/stage0-core` 的 paired sampling 接口，从 `codex/stage0-baselines` 引入 stage0_summary 模块与测试。
- Case4 原 37×37 mm 保留为不可行诊断；另立有本地来源 hash 的 `case4-maskplace-59mm-alternative-v1`，保持芯粒、功率和引脚物理值不变。它是不同几何问题，不冒称论文原始输入修正。
- 增加所有连续优化器共用的合法化接口，固定上限的确定性 MaxRects/shelf 搜索；保留合法输入，失败与面积证明分开，不把启发式失败称为不可行证明。
- 增加六个真实数值配置、同状态分叉、硬评价预算、原始与后处理收益分解；恢复可行性与合法到合法优化分别统计。
- HotSpot 真实 subprocess 适配、可追溯输入/输出与分组留出校准单独实现；实际 simulator 可用性和校准误差须以生成结果和热校准文档为准。
- 每条记录保存坐标、checksum、调用数、梯度/训练次数、合法率及成本；新增独立 replay audit 从坐标重算指标并核验 action×seed 完整网格。
- 本轮继续本地→测试→diff→commit→push→远端核验；部署后检查 README、数据协议、结果入口与主仓库入口。主工作树原有 PDF 删除状态继续保留。
- 实际采集 129 状态/2,322 标签，预算与坐标回放通过；另加 129 次 legalizer-only 控制评价。恢复任务 61/68 状态的后处理奖励打平，禁止生成伪唯一 winner；61 个合法优化状态均有数值动作均值差异，不等同显著性。
- 正式集合实际完成 27 次 HotSpot block 运行，Case8/Case10 留出 MAE=58.9105 °C，代理不升级为温度；原始输入输出、固定 config、split 和早期 19 次模拟证据永久入库。
- 全量验证：636 passed、7 skipped；热物理完整分层 package 校准和 CNN/Bandit 策略训练仍未完成。

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

## 本次记录（2026-10-24，结果对照与公开访问）

- 在 stage0_real_trajectories 新增 RESULTS_COMPARISON.md，逐 Case 比较 HPWL、热代理、精确重叠、综合目标、检查点排序、时间和峰值内存；同步导出 CSV/统计 JSON 与可复现生成器。
- 新增 paper_results_sources.md，依据 TAP、ATPlace 和 RLPlanner 全文核对作者报告数字，保留页码、表号、指标、比较对象与预算。单独解释 20.28% 综合目标、1.42× 比值及热模型加速不能当成端到端加速。
- 报告使用合法性标记，将 Case4 排除于 9-Case 汇总；区分独立轨迹的检查点排序与同状态反事实偏好，未把当前轨迹称作 CNN/Bandit 效果。
- 按用户明确要求，将 amirocok06-byte/chiplet-layout-2p5d 改为 public；GitHub 返回 visibility=PUBLIC、isPrivate=false。同步更新结果目录入口、README、生成器、审核元数据和 GitHub 分支。
- 验证采用生成器 --check、30 行 CSV 与 30 份 summary 对照、整包审核和文档链接检查；本轮不修改优化算法或重跑优化。部署后继续执行本日志的本地/远端同步复查规定。
