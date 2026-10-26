"""Project archived positions into v2 maps without rerunning optimization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chiplet_layout_experiment.state_projection import CHANNELS, STATE_VERSION, project_state
from chiplet_layout_experiment.trajectory_data import load_case
from plot_snapshot_maps import plot_snapshot_maps
from plot_state_flow import plot_state_flow
from plot_algorithm_effects import plot_algorithm_effects
from plot_grid import compute_map_limits

METHODS = ("sa", "analytical", "rl")
NAMES = ("Connectivity priority", "Spatial thermal proxy", "HPWL contribution (mm/pixel)",
         "Overlap / boundary violation")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render(root, case, dpi):
    directory = root / case
    summaries = {a: read(directory/a/"summary.json") for a in METHODS}
    source = summaries['sa']['source']
    problem, _ = load_case(Path(source['source_directory']).parent, case,
                           seed=summaries['sa']['seed'])
    maps = {a: np.stack([project_state(problem, np.array(c['positions']))
                        for c in summaries[a]['checkpoints']]) for a in METHODS}
    degree = np.bincount(problem.pin_indices.ravel(), minlength=problem.n)
    write(directory/'state_v2'/'priority_values.json', {
        'definition': 'incident pin count / maximum incident pin count; ties preserved',
        'values': [{'chiplet': name, 'incident_pins': int(d), 'priority': float(q)}
                   for name, d, q in zip(problem.names, degree, problem.priority)]})
    limits = [compute_map_limits(np.stack([maps[a][:, i] for a in METHODS]), channel)
              for i, channel in enumerate(("order", "thermal", "hpwl", "legality"))]
    write(directory/'state_v2'/'display_scales.json', {
        'state_version': STATE_VERSION, 'channels': CHANNELS, 'limits': limits,
        'shared_across': 'three algorithms and all five frames',
        'priority_overlap_rule': 'maximum q among geometrically covering chiplets',
        'raster_is_not_validity_verdict': True})
    for a in METHODS:
        destination = directory/a/'state_v2'
        raw = destination/'raw'
        raw.mkdir(parents=True, exist_ok=True)
        boxes = [np.column_stack((c['positions'], problem.sizes)) for c in summaries[a]['checkpoints']]
        snapshots = []
        for index, c in enumerate(summaries[a]['checkpoints']):
            step = c['step']
            stem = f'checkpoint_{step:04d}'
            np.savez_compressed(raw/f'{stem}.npz', maps=maps[a][index])
            x = np.array(c['positions'])
            boundary = bool(np.all(x >= -1e-10) and np.all(x+problem.sizes <= 1+1e-10))
            write(destination/f'{stem}.json', {
                'state_version': STATE_VERSION, 'channels': CHANNELS,
                'map_shape': [4, 256, 256], 'step': step, 'positions': c['positions'],
                'metrics': c['metrics'], 'boundary_valid': boundary,
                'historical_snapshot': f'../{stem}.json',
                'historical_snapshot_sha256': hashlib.sha256((directory/a/f'{stem}.json').read_bytes()).hexdigest(),
                'maps_sha256': hashlib.sha256(maps[a][index].tobytes()).hexdigest(),
                'optimization_rerun': False})
            title = (f'{case} | {a} | iteration {step}/800 | '
                     f"overlap={c['metrics']['overlap_fraction']:.5f} | "
                     f"valid={c['metrics']['valid']} | boundary={boundary}")
            plot_snapshot_maps(raw/f'{stem}.npz', destination/f'{stem}.png', dpi=dpi,
                               title=title, channel_labels=NAMES, limits=limits,
                               chiplet_boxes=boxes[index])
            snapshots.append(stem)
        np.savez_compressed(raw/'state_flow.npz', maps=maps[a])
        plot_state_flow(raw/'state_flow.npz', destination/'state_flow.png', max_frames=5,
            dpi=dpi, frame_labels=[f"iter {c['step']}" for c in summaries[a]['checkpoints']],
            title=f'{case} | {a} | archived independent trajectory | {STATE_VERSION}',
            channel_labels=NAMES, limits=limits, chiplet_boxes=boxes)
        write(destination/'manifest.json', {'state_version': STATE_VERSION, 'snapshots': snapshots,
            'channels': CHANNELS, 'origin': '../summary.json', 'optimization_rerun': False})
        print(f'{case}/{a}: four-channel snapshots and state flow rendered', flush=True)
    effects = directory/'state_v2'/'effects.npz'
    np.savez_compressed(effects, before_maps=maps['sa'][0],
                        after_maps=np.stack([maps[a][-1] for a in METHODS]))
    plot_algorithm_effects(effects, directory/'state_v2'/'algorithm_effects.png', list(METHODS), dpi=dpi)


def gallery(root, cases):
    rows = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>2.5D 布局状态：四通道五检查点</title>',
        '<style>body{font-family:system-ui;max-width:1640px;margin:28px auto;padding:0 18px;color:#263746;background:#f5f7fa}section,header{background:white;padding:24px;margin:24px 0;border:1px solid #dae2eb;border-radius:12px}img{width:100%;height:auto}summary{cursor:pointer;padding:12px;font-weight:600}table{border-collapse:collapse}td,th{padding:8px 14px;border:1px solid #ccd6e0}a{color:#145c90}.note{padding:14px;background:#fff4db}code{background:#eef3f7;padding:2px 5px}</style>',
        '<header><h1>四通道布局状态 · 真实五检查点</h1>',
        '<p>从已有800次候选轨迹的0/200/400/600/800检查点重新投影，未重跑优化器。每帧四张256×256 map；同Case、跨算法和时间使用固定色阶。</p>',
        '<table><tr><th>通道</th><th>数值含义</th><th>可以判断什么</th></tr>',
        '<tr><td>M_priority</td><td>q_i=d_i/max(d)，d_i为incident pin数；空白−1，断连芯粒0，并列保留；覆盖冲突取最大q</td><td>连接度较高的芯粒在哪里；它不编码实际执行顺序，也不等于时序关键性</td></tr>',
        '<tr><td>M_thermal proxy</td><td>高斯功率密度扩散，无温度单位</td><td>代理场集中位置；HotSpot留出MAE58.91°C，当前未获可信温度校准</td></tr>',
        '<tr><td>M_HPWL</td><td>真实引脚包围盒上的线长贡献，mm/pixel</td><td>当前线长贡献集中位置；像素总和约等于HPWL</td></tr>',
        '<tr><td>M_overlap/legality</td><td>精确矩形交集按像素面积分配并归一化，加越界量的边界带投影；0表示未显示违反</td><td>空间约束异常位置；细微重叠与全局合法性以精确evaluator为准</td></tr></table>',
        '<p class="note">第四通道由精确几何交集生成，避免相邻芯粒落在同一像素内导致假重叠。它是观察通道，不等于候选动作合法掩码。三类Agent仍为SA、解析法、RL。</p>',
        '<p><a href="../../four_channel_state_design.md">四通道定义与消融方案</a> · <a href="../../experimental_design_contextual_bandit_2p5d.pdf">更新的实验方案PDF</a> · <a href="RESULTS_COMPARISON.md">原轨迹指标和论文对照</a> · <a href="../stage0_same_state/EXPERIMENT_FINDINGS.md">同状态动作收益与合法化控制</a></p>',
        '<details open><summary>四通道计算公式与图例：怎样读颜色和数值</summary>',
        '<p><b>1. Connectivity priority：</b>d_i为芯粒关联pin的出现次数，q_i=d_i/max(max(d),1)。像素覆盖多个芯粒时取最大q；无人覆盖为−1，断连芯粒为0。16级viridis：深紫低、蓝绿中、黄高，浅灰为空白，黑线为芯粒边界。连接数相同就应同色；颜色块随布局移动，但q值在netlist不变时固定。</p>',
        '<p><b>2. Spatial thermal proxy：</b>像素中心处计算Σ_i(P_i/A_i) exp[−½Σ_a(u_ga−c_ia)²/σ_ia²]，σ_ia²=s_ia²/12+0.08²；P以W、面积A以mm²给出幅值。inferno由黑紫到橙黄，PowerNorm γ=0.6；色条是原代理值，不是°C/K。标量峰值在芯粒中心采样，图在网格中心采样。</p>',
        '<p><b>3. HPWL contribution：</b>真实pin offset得到每网的mm制HPWL=L_n；将w_n L_n均匀分到其引脚包围盒的K_n个像素，M(g)=Σ_n w_n L_n/K_n。色条mm/pixel；magma亮处贡献较高，浅灰处无可显示贡献；SymLogNorm压缩动态范围。当前适配器为two-pin、w_n=1；此图不画实际走线。</p>',
        '<p><b>4. Overlap / boundary violation：</b>Σ_(i&lt;j) area(g∩R_i∩R_j)/area(g)/max(N−1,1)，加上沿板边一像素带投影的归一化越界量。线性Reds：0白，正值浅红至深红；板内红块提示重叠，板边红带提示越界。它不等于合法概率或精确重叠率，必须同时读标题中的overlap、boundary和valid。</p>',
        '<p><b>差分图：</b>四列均为ΔM=M_after−M_before；coolwarm中蓝色减少、红色增加、白色接近不变。priority的变化主要是位置变化。快照第四通道的“红=绝对违反”与差分图的“红=增加”需区分。所有色标按同Case全部算法与五帧固定；300 DPI/16×16宏网格不改变256×256数据。</p>',
        '<p><a href="../../channel_reference.md">完整含义、公式、单位、颜色及读图边界</a></p></details>',
        '<p>下面是单seed独立轨迹观察，不是每个检查点同状态分叉的算法胜出证据。历史三通道图与原始JSON继续保存，四通道产物在state_v2目录。</p></header>']
    for case in cases:
        result = read(root/case/'case_result.json')
        rows += [f'<section id="{case}"><h2>{case}</h2>',
                 f"<p>{result['chiplets']}芯粒 · 面积利用率{result['area_utilization']:.2%}</p>"]
        priority_file = root/case/'state_v2'/'priority_values.json'
        if priority_file.exists():
            values = read(priority_file)['values']
            rows += ['<details><summary>priority 数值如何得到：逐芯粒连接度与颜色值</summary><table><tr><th>芯粒</th><th>incident pin 数</th><th>q_i / 色图值</th></tr>']
            rows += [f'<tr><td>{item["chiplet"]}</td><td>{item["incident_pins"]}</td><td>{item["priority"]:.6f}</td></tr>'
                     for item in sorted(values, key=lambda d: (-d['priority'], d['chiplet']))]
            rows += ['</table><p>每个芯粒内部颜色由q_i决定；布局移动只改变其空间位置。重叠像素取最大q_i；低连接度不等于可忽略。</p></details>']
        if result['proven_infeasible_by_area']:
            rows += ['<p class="note">此处Case4保留原37×37 mm不可行诊断。59×59 mm替代几何是另一个版本，见同状态实验，不能混合比较。</p>']
        rows += [f'<img loading="lazy" src="{case}/metric_trends.png" alt="完整指标走势">']
        for a in METHODS:
            m = result['final_metrics'][a]
            rows += [f'<details><summary>{a} · HPWL {m["hpwl_mm"]:.2f} mm · overlap {m["overlap_fraction"]:.6f} · valid {m["valid"]}</summary>',
                f'<img loading="lazy" src="{case}/{a}/state_v2/state_flow.png" alt="四通道五检查点状态流">',
                f'<p><a href="{case}/{a}/summary.json">原始运行记录</a> · <a href="{case}/{a}/state_v2/manifest.json">四通道投影版本</a> · <a href="{case}/{a}/state_flow.png">历史三通道状态流</a></p>',
                f'<img loading="lazy" src="{case}/{a}/parameter_trends.png" alt="原始参数走势">']
            for step in (0, 200, 400, 600, 800):
                stem = f'{case}/{a}/state_v2/checkpoint_{step:04d}'
                rows += [f'<details><summary>iteration {step}</summary><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.json">坐标与精确指标</a></p><img loading="lazy" src="{stem}.png" alt="{case} {a} iteration{step} 四通道"></details>']
            rows += ['</details>']
        rows += [f'<details><summary>最终−初始：四通道变化</summary><img loading="lazy" src="{case}/state_v2/algorithm_effects.png" alt="独立轨迹变化"></details></section>']
    rows += ['</html>']
    (root/'index.html').write_text('\n'.join(rows), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--index-only', action='store_true')
    args = parser.parse_args()
    cases = args.cases or read(args.root/'manifest.json')['cases']
    if not args.index_only:
        for case in cases:
            render(args.root, case, args.dpi)
    # Always retain all cases when producing the gallery, even on partial refresh.
    gallery(args.root, read(args.root/'manifest.json')['cases'])
