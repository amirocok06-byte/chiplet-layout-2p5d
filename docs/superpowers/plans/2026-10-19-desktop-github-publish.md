# Desktop 发布副本与 GitHub Private 项目 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动原工作区的前提下，建立桌面发布副本、补齐双语项目介绍，并创建 Private GitHub 仓库。

**Architecture:** 采用“原目录只读 + 桌面精选副本 + GitHub 初始提交”的边界。发布副本只包含当前 Phase 1 源码、测试、配置、设计文档和来源说明；敏感、临时、不可公开或与项目无关的文件排除在副本之外。

**Tech Stack:** PowerShell、Python 3.13、pytest、Git、GitHub CLI、GitHub Actions。

---

### Task 1: 建立发布目录和设计记录

**Files:**
- Create: `docs/superpowers/specs/2026-10-19-desktop-github-publish-design.md`
- Create: `docs/superpowers/plans/2026-10-19-desktop-github-publish.md`

- [x] 创建目录树并写入设计记录。
- [x] 复核设计记录没有占位符、敏感信息或与发布边界冲突的内容。

### Task 2: 复制并筛选项目内容

**Files:**
- Copy: `src/`, `tests/`, `schemas/`, `prompts/`, `pyproject.toml`
- Copy: `config/default.yaml` as `configs/default.yaml`
- Copy: `docs/experimental_design_contextual_bandit_2p5d.tex`, `docs/experimental_design_contextual_bandit_2p5d.pdf`, `docs/面向2.5D系统集成的芯粒布局方法-MVP1.md`
- Create: `references/README.md`, `data/sample/README.md`, `scripts/README.md`

- [x] 复制当前源码和测试，不复制原 `.git`、缓存、临时目录和未筛选根级资料。
- [x] 将 `config/default.yaml` 放入 `configs/`，并保留 `schemas/`、`prompts/` 在仓库根目录，以满足现有测试和源码路径契约。
- [x] 检查副本中没有密钥、令牌、私人路径或 F 盘第三方源码。

### Task 3: 生成双语发布文件

**Files:**
- Create: `README.md`
- Create: `CONTRIBUTING.md`
- Create: `.gitignore`
- Create: `.github/workflows/ci.yml`

- [x] README 中同时提供中文与英文的项目定位、三类方法说明、Phase 1 边界、快速开始和验证命令。
- [x] CONTRIBUTING 中说明数据、外部源码、实验结果和复现声明的发布边界。
- [x] CI 只运行当前不需要 PyTorch/CUDA 的测试和编译检查。

### Task 4: 本地验证和 Git 初始化

**Files:**
- Modify: `docs/superpowers/plans/2026-10-19-desktop-github-publish.md`
- Create: `.git/`

- [x] 在副本中运行 `python -m pytest -q`，结果为 `323 passed, 7 skipped`。
- [x] 在副本中运行 `python -m compileall -q src`。
- [ ] 运行发布清单检查和 `git diff --check`。
- [ ] 初始化 Git、创建 `main`、提交筛选后的内容。

### Task 5: 创建 Private GitHub 仓库并核验

**Files:**
- External: `https://github.com/amirocok06-byte/chiplet-layout-2p5d`

- [ ] 使用已登录的 GitHub CLI 创建 Private 仓库。
- [ ] 添加 `origin` 并推送 `main`。
- [ ] 核验仓库可见性、默认分支、远端提交和 README 内容。
- [ ] 在进度文件中记录本地路径、远程地址和验证结果，不记录任何令牌。
