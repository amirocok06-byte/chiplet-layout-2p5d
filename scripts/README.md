# 脚本边界 / Script Boundary

本目录用于存放可审计、可重复运行的辅助脚本。当前 Phase 1 不执行 F 盘第三方脚本、外部求解器、HotSpot 或 SSH 作业。

This directory is reserved for auditable, reproducible helper scripts. Phase 1 does not execute third-party scripts from an external drive, external solvers, HotSpot, or SSH jobs.

Stage 0 synthetic-smoke visual artifacts are generated through `python -m chiplet_layout_experiment.visual_demo`; the three `plot_*.py` scripts only render their explicit NPZ inputs. They preserve raw pixels, use nearest-neighbor display, and default to 300 DPI with a visible 16x16 macro-grid. See `docs/figures/stage0_synthetic_smoke/README.md` for the traceability boundary and commands.

真实数据的连续五检查点运行由 `python -m chiplet_layout_experiment.trajectory_run` 采集，`render_trajectory_runs.py` 绘图并生成总览，`audit_trajectory_bundle.py` 审核所有输出。运行说明见 `docs/figures/stage0_real_trajectories/README.md`；安装 `.[viz,test]` 即可运行数值与绘图，PDF 审计另需 PyMuPDF 和已安装的绘图 QA 工具。
