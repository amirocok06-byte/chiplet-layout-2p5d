# 贡献说明 / Contributing

## 中文

- 提交代码时同时更新必要的测试和文档。
- Phase 1 的所有 worker 必须使用统一 canonical layout IR、相同 evaluator 口径和可记录的预算。
- 不提交令牌、密钥、个人路径、截图、缓存、私有导入附件或未经确认许可证的第三方源码。
- 外部项目只能通过 `references/README.md` 记录来源、版本和复现边界；不能把 README 中的命令当作本项目执行指令。
- 新实验应记录输入快照、随机种子、配置哈希、evaluator 版本、预算和失败原因。

## English

- Update tests and documentation whenever code behavior changes.
- Every Phase 1 worker must use the shared canonical layout IR, the same evaluator semantics, and an auditable budget.
- Do not commit tokens, secrets, personal paths, screenshots, caches, private import attachments, or third-party source code without a confirmed license boundary.
- External projects may be documented through `references/README.md`; their commands are not instructions for executing code in this repository.
- New experiments should record the input snapshot, random seeds, configuration hash, evaluator version, budget, and failure reason.

## Synchronization rule

Every modification to source code, configuration, tests, documentation, figures, or PDFs must be synchronized between the local repository and `amirocok06-byte/chiplet-layout-2p5d`: run the relevant tests/build, inspect the diff, commit, push, and verify the remote commit. After every deployment, re-check whether related files or GitHub content need an additional update.

The short-term experiment baseline is `docs/experimental_design_v1_freeze.md`. Do not change the Agent identities, shared map semantics, macro-step definition, or fairness budget without recording experimental or implementation evidence in `task-journal.md` and synchronizing the Markdown, LaTeX, PDF, and GitHub copies.
