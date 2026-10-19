# Desktop 发布副本与 GitHub Private 项目设计

## 目标

在桌面建立 `Chiplet-Layout-2.5D` 的可发布副本，并创建同名 Private GitHub 仓库。原工作区保持不移动、不清理、不覆盖。

## 发布边界

- 纳入当前已完成的 `chiplet_layout_workflow` 与 `chiplet_layout_experiment` 源码、测试、配置、实验设计文档和必要构建产物。
- 以双语 README 说明三类方法：模拟退火、解析式布局、RL-compatible worker；明确第一阶段是统一协议与可复现实验基线，不宣称复现外部项目的私有核心实现。
- 仅在 `references/` 中保留外部项目来源与复现边界，不复制或执行 F 盘第三方源码。
- 排除截图、缓存、临时目录、编辑器元数据、私有导入附件、个人研究资料和旧构建缓存。
- F 盘 PDF 不被修改；桌面副本使用工作区中的设计源文件和已生成 PDF。

## 目录职责

```text
src/              可安装 Python 包
tests/            回归测试与 Phase 1 实验测试
docs/             实验设计、MVP1 说明和发布设计记录
configs/          可公开的默认配置
schemas/          仓库级 schema 资源
prompts/          仓库级 prompt 模板
references/       外部项目链接、来源说明和许可证边界
data/sample/      小型、可公开的示例数据
scripts/          可重复运行的本地辅助脚本
.github/workflows GitHub Actions 最小 CI
```

## GitHub 操作

- 账号：`amirocok06-byte`
- 仓库：`amirocok06-byte/chiplet-layout-2p5d`
- 可见性：Private
- 初始分支：`main`
- 首次提交包含经过敏感文件筛选的发布副本；不从原工作区直接添加远程。

## 验收

- 发布副本不包含 `.git`、`.pytest_cache`、`.tmp-ai`、截图、导入附件或凭据文件。
- `python -m pytest -q`、`python -m compileall -q src` 和 `git diff --check` 通过。
- GitHub 仓库存在且为 Private，默认分支包含 README 和筛选后的项目结构。
