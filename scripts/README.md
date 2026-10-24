# 脚本边界 / Script Boundary

本目录用于存放可审计、可重复运行的辅助脚本。当前 Phase 1 不执行 F 盘第三方脚本、外部求解器、HotSpot 或 SSH 作业。

This directory is reserved for auditable, reproducible helper scripts. Phase 1 does not execute third-party scripts from an external drive, external solvers, HotSpot, or SSH jobs.

Stage 0 synthetic-smoke visual artifacts are generated through `python -m chiplet_layout_experiment.visual_demo`; the three `plot_*.py` scripts only render its explicit NPZ inputs. See `docs/figures/stage0_synthetic_smoke/README.md` for the traceability boundary and commands.
