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
