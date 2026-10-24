# 外部参考项目 / External References

本目录只记录外部项目的来源、用途和复现边界，不包含外部项目源码。正式实验前应重新核对远程仓库的 commit、许可证和依赖版本。

This directory records external sources, intended use, and reproduction boundaries. It does not contain third-party source code. Before formal experiments, pin the remote commit, license, and dependency versions again.

| 项目 / Project | 官方地址 / Official URL | 本项目中的使用边界 / Boundary |
|---|---|---|
| TAP-2.5D | https://github.com/bu-icsg/TAP-2.5D | SA、给定布局评价、线长/热评价接口设计参考；不直接复制代码。 |
| ATPlace_pub | https://github.com/Brilight/ATPlace_pub | HPWL、pin offset、配置字段和公开 wrapper 作为接口参考；加密内核不作为透明实现。 |
| RLPlanner | https://github.com/weiweihook/RLPlanner | RL 方法叙事和快速热模型的背景参考；当前仓库的 RL-compatible worker 不声称复现它。 |
| MaskPlace2.5D | https://github.com/xunk33333/MaskPlace2.5D | 热模型、数据处理和 mask 概念参考；公开内容不等于完整 placement worker。 |
| STAMP2.5D | https://github.com/MicroSystemsDesignLab/STAMP2.5D | 2.5D 布局和热约束方向参考；采用前需单独核对许可证和 benchmark 兼容性。 |

Stage 0 的实际输入、固定 commit 和“可运行三臂/外部源码边界”见 [`docs/stage0_data_source_freeze.md`](../docs/stage0_data_source_freeze.md)。

The local design PDF is a project-authored experiment plan. The user-supplied external-drive PDF was treated as reference material and was not modified.
