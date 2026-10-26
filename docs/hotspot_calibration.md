# HotSpot calibration

## 正式采集与永久保存

正式 129 状态集合的校准已实际运行，结果见
[stage0_hotspot_calibration](figures/stage0_hotspot_calibration/README.md)：27 个成功布局，
按 Case 留出 Case8/Case10，训练 MAE=42.9523 °C、留出 MAE=58.9105 °C，斜率为负。
该结果不支持将现有空间代理换算为可信物理温度；优化标签继续明确标为 proxy。
真实模拟调用、输入、输出和拟合报告均已入库；固定配置见
[configs/hotspot_stage0_block.config](../configs/hotspot_stage0_block.config)。
下文早期 19 次探索的原始证据已从临时目录复制到
[stage0_hotspot_pilot](figures/stage0_hotspot_pilot)，临时绝对路径仅保留为历史调用 provenance。

目前验证的是平面 block 参考模型协议。下一次使用热收益作科学结论前，必须落实分层 interposer/ubump
材料与封装参数，并重新按 Case 留出评估模型误差，不能直接使用本次负斜率拟合。

## Actual runtime verified on 2026-10-25

The local simulator **runs successfully**. Nineteen real subprocess measurements
were completed: one two-chiplet protocol smoke, ten independently prepared real
ATPlace layouts, and eight layouts from `build/same-state-smoke`. No network
access, downloaded simulator, synthetic simulator temperatures, or substituted
thermal solver was used for these measurements.

Local source root:
`F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/4-开源源码`.

| Item | Verified location / value |
| --- | --- |
| Linux executable | `TAP-2.5D/util/hotspot` under the source root |
| Original configuration | `TAP-2.5D/util/hotspot.config` |
| Local calling-code reference | `TAP-2.5D/passive_interposer.py`, `run_hotspot` |
| WSL distribution | `Ubuntu-24.04-G` |
| Binary format | ELF x86-64, dynamic interpreter `/lib64/ld-linux-x86-64.so.2` |
| Dependencies | `libm.so.6`, `libc.so.6`; all resolved under WSL |
| Executable SHA-256 | `22440f9931bae8398461be7a522aa806e6181ed2571ab5c3f3c0b4b3d461a178` |
| Original config SHA-256 | `28bcf5d57cdc04da3aee74e66c88809615c9afc005e002ae3ae827ec7a748fd7` |

`ATPlace_pub/thermal/hotspot` and `MaskPlace2.5D/thermal/hotspot` have the same
executable hash. Their configs and Python callers are also locally available.
No HotSpot C/C++ implementation was found among the local source-root files;
the available distribution supplies binaries, configs, and placement wrappers.
The TAP caller uses grid/detailed-3D options and subtracts 273.15 from Kelvin.
This adapter deliberately exposes a standalone **block** reference model.

The original 30 mm spreader cannot cover all benchmark boards. For the two
case-calibration collections, one config was frozen *before* sampling: the TAP
config with `-s_sink 0.12` and `-s_spreader 0.10`, both in metres. Other file
settings were retained, including ambient 318.15 K and convection resistance
0.1 K/W. Every run explicitly overrides `-model_type block` and records that
override. The frozen file hash is:

`30b3790b15be2b1eb5660e1a97b5302ab835a91f5afa3e78088aed838c21e5c7`.

On this machine it is preserved at
`C:/Users/Lenovo/AppData/Local/Temp/hotspot-case-calibration-ae7v1c2p/frozen.config`.
Runtime artifacts were written outside the shared repository to respect the
four-file implementation scope. Temporary paths are machine-local evidence,
not permanent repository artifacts; preserve them before clearing Windows temp.

## Measured calibration results

The fit is ordinary least squares:
`predicted_peak_celsius = slope * thermal_proxy_peak + intercept_celsius`.
Coefficients use training samples only; held-out cases never participate in fitting.

| Collection | Training / held-out | Train MAE / RMSE (°C) | Held-out MAE / RMSE (°C) |
| --- | --- | --- | --- |
| Independent real ATPlace states, 10 runs | Case1: 4 and Case3: 3 / Case8: 3 | 2.92549 / 3.36640 | **70.05309 / 70.05683** |
| Shared smoke states, 8 runs | Case1: 3 / Case4: 5 | 0.27682 / 0.31437 | **129.89580 / 129.90830** |

The independent collection used `trajectory_data.load_case` from the local
`ATPlace_pub/cases` data and the shared legalizer. Case1 used seeds 42–45;
Case3 used 42–44; Case8 used 42, 44, 46 (intervening attempts failed strict
geometry checks). These are independently prepared states, not the later shared
same-state collection. Original board sizes were retained: Case1 42×42 mm,
Case3 39×39 mm, Case8 26×23 mm. Repeated within-case layouts were retained as
measurements; no geometry hash crosses the train/held-out boundary.

Its slope is `-4.507778528467325`, intercept `178.19137656569464 °C`, with
training proxy range `[0.9351891383908816, 2.47060589329097]`. Held-out Case8
HotSpot peaks are 94.11, 92.62, and 93.25 °C.

The shared smoke collection read `build/same-state-smoke/Case*/problem.json`
and `states/*.json` directly. Case4 carries the explicitly alternative 59×59 mm
geometry already recorded by that collection; calibration did not alter it.
The CLI selected the first five successful states per case in sorted file order;
only three Case1 states passed strict checks. Nine scanned states were rejected
for positive overlap. The Case4 cap leaves later states unexamined.

Its slope is `-469.9885106678564`, intercept `620.3004421279495 °C`, with training
proxy range `[0.9327999314275531, 0.9446622410625902]`.

| Case | Measured state ID prefix | Proxy | HotSpot chiplet peak (°C) | Split |
| --- | --- | ---: | ---: | --- |
| Case1 | `27ed786c` | 0.93480089 | 181.37 | train |
| Case1 | `7415bbd6` | 0.94466224 | 176.25 | train |
| Case1 | `a3b2421e` | 0.93279993 | 181.55 | train |
| Case4 | `0ea99a55` | 1.04337838 | 259.85 | held-out |
| Case4 | `2cebaef5` | 1.02782283 | 267.79 | held-out |
| Case4 | `407cea82` | 1.02805257 | 269.59 | held-out |
| Case4 | `4c9f6261` | 1.04250801 | 260.00 | held-out |
| Case4 | `97520052` | 1.03685678 | 259.86 | held-out |

Both fits have negative slopes and large held-out error. These measured results
do **not** establish a transferable proxy-to-temperature conversion. Keep the
trajectory proxy labelled as an uncalibrated proxy. The flat silicon block model
also does not reproduce TAP's layered interposer/ubump package. Temperatures are
reference-model outputs, not validated physical temperatures for a 2.5D package.

Evidence directories under the temp root above:

| Directory | Contents / canonical sample-list SHA-256 |
| --- | --- |
| `results` | Independent collection; `cbbad95aa765bedf2fb86cec4f7575d966b0c51b394d72c432a6a6991c2e3b94` |
| `same-state-smoke` | Shared collection; `b4c3af588d7a33db2ce9d359cb44d9c7c0985b0200f1f3914a58d7b28242d9ec` |

Each contains `calibration.json`, `collection.json`, and per-run input/output
files. Full state IDs, source-file hashes, geometry, powers, proxy version,
executable/config/adapter hashes, elapsed time, timestamp, actual command, and
Kelvin temperatures are preserved. A separate two-chiplet run at
`C:/Users/Lenovo/AppData/Local/Temp/hotspot-real-mqv_x18v/run` returned 328.75 K
and 337.94 K (64.79 °C peak), using the original TAP config.

## Running the saved-state collector

From the shared worktree, this exact command completed successfully:

```powershell
python scripts/calibrate_hotspot.py --states-root build/same-state-smoke --executable "F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/4-开源源码/TAP-2.5D/util/hotspot" --config C:/Users/Lenovo/AppData/Local/Temp/hotspot-case-calibration-ae7v1c2p/frozen.config --output C:/Users/Lenovo/AppData/Local/Temp/hotspot-case-calibration-ae7v1c2p/same-state-smoke --wsl-distribution Ubuntu-24.04-G --layouts-per-case 5 --held-out Case4
```

Choose a **new output directory** when rerunning. For the full collection replace
`--states-root` with `docs/figures/stage0_same_state`. Repeat `--held-out CaseN`
to reserve multiple cases, and optionally repeat `--case CaseN` to restrict the
sampled cases. Selection is sorted paths, first N successful legal states per
case; it is recorded and does not inspect temperatures to choose states.

To reproduce the frozen config from the original:

```python
import re
from pathlib import Path
base = Path("F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/4-开源源码/TAP-2.5D/util")
text = (base / "hotspot.config").read_text()
text = re.sub(r"(-s_sink\s+)\S+", r"\g<1>0.12", text)
text = re.sub(r"(-s_spreader\s+)\S+", r"\g<1>0.10", text)
Path("frozen.config").write_text(text, encoding="utf-8")
```

File hashes are byte-sensitive to line endings. This reproduces the measured
hash on Windows with the same original file. A Linux-created LF config may
have a different hash; use one unchanged config throughout any new experiment.

Native Linux runs omit `--wsl-distribution` and provide native executable/config
paths. WSL runs accept Windows drive paths and translate them into `/mnt/<drive>`
paths, preserving spaces and Unicode via an argument list without shell expansion.
No install or compilation is required in the verified WSL environment.

## Adapter and fitting contract

`HotSpotRunner([executable], config, timeout=60, wsl_distribution=...).measure`
accepts a `TracedProblem`, normalized lower-left positions, a new run directory,
and explicit `case_id` / `layout_id`. It does not mutate the shared objective.

- `.flp` rows are `name width height left bottom`, all geometry in **metres**:
  normalized coordinates × per-axis `board_mm` ÷ 1000.
- `.ptrace` is a name header plus one row of **Watts**. Dead space is tiled
  with zero-power silicon rectangles to retain the full physical board extent.
- Negative/nonfinite power, invalid shapes/names, nonpositive geometry, out-of-board
  and overlapping layouts are rejected before launching. Overlap is deliberately
  strict: any positive intersection is rejected, including floating-point slivers
  tolerated by the shared legalizer. No legalization or overlap repair is done.
  Upper-bound overshoot ≤1e-12 is clipped as roundoff and the exported geometry is
  recorded. Extremely thin dead-space slivers can also cause real HotSpot failures;
  those failures remain failures and never become samples.
- Config bytes are frozen at runner construction and copied into each run.
  External layer/init files, detailed package modeling, and leakage are rejected
  by this standalone adapter, so hidden thermal input files cannot escape the hash.
- Only the named original chiplets from `steady.k` determine peak temperature.
  Internal spreader/interface/dead-space nodes are excluded. Kelvin values must
  be finite and positive, every chiplet must occur exactly once, and Celsius is
  Kelvin minus 273.15. Missing/malformed output, nonzero exits, launch failures,
  and timeouts raise `HotSpotError`; no successful sample is written.
- Arguments are passed directly to `subprocess.run` with a timeout. On Windows,
  the timeout bounds the WSL launcher invocation; remote descendant teardown
  depends on WSL behavior. New run directories prevent reuse of stale output.
- `fit_calibration(samples, split_by="case", held_out=[...])` fits only training
  rows. It requires at least two nonconstant training observations, nonempty
  held-out groups, finite values, unique sample IDs, one config hash, executable
  hash, and proxy version. Identical layout hashes across the split are rejected.
- `split_by="layout"` reserves layout IDs instead; it can contain other layouts
  of the same case in training, so it measures layout transfer rather than unseen
  case transfer. In CLI reports layout IDs include the relative case/variant path
  and state filename stem. Case splitting keeps all variants of one case together
  when `problem.json` records `source.case`.
- Reports retain coefficients, proxy range, train/held-out IDs, MAE/RMSE in °C,
  per-held-out-case metrics, slope/extrapolation diagnostics, complete sample
  provenance, and a canonical JSON sample-list hash. There is no fit-on-all-data
  step after validation. Insufficient accepted data produces exit code 2 and
  preserves `collection.json` without claiming a fitted calibration.

## Tests

```powershell
python -m pytest tests/experiment/test_hotspot.py -q
```

The test-only Python executable fixtures exercise the real subprocess protocol,
metres/Watts conversion, dead-space area/extent, config freezing, provenance,
illegal-layout rejection, output parsing, timeout/error paths, grouped fitting,
leakage rejection, and CLI saved-state collection. Their invented temperatures
are exclusively unit-test data and never appear in either real calibration.
