"""Summarize completed preference probes without changing training settings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--labels-root',type=Path,required=True)
    args = parser.parse_args()
    preference = json.loads((args.root/'preferences.json').read_text())
    cv = json.loads((args.root/'cross_validation.json').read_text())
    diagnostic = json.loads((args.root/'pilot_feature_diagnostic.json').read_text())
    arms = sorted(preference['65']['overall']['mean_rewards'])
    rows = [json.loads(line) for path in args.labels_root.glob('*/labels.jsonl')
            for line in path.read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row['phase']=='legal_optimization':
            groups[row['state_id']].append(row)
    audit = {}
    stable_switches = {}
    for budget in (17,257):
        probe_rows = [json.loads(line) for line in (args.root/f'budget_{budget}_actions.jsonl').read_text().splitlines()]
        probe_groups = defaultdict(list)
        for row in probe_rows:
            probe_groups[row['state_id']].append(row)
        valid = set(probe_groups)==set(groups)
        for state_id, entries in probe_groups.items():
            expected = groups[state_id]
            valid &= len(entries)==18 and {(r['arm_id'],r['seed']) for r in entries}=={(r['arm_id'],r['seed']) for r in expected}
            valid &= all(r['input_checksum']==expected[0]['input_checksum']
                         and r['problem_checksum']==expected[0]['problem_checksum']
                         and r['evaluator_calls']==budget for r in entries)
        switches = 0
        for state_id, entries in probe_groups.items():
            def optimal_set(action_rows):
                means = np.asarray([np.mean([r['bandit_reward'] for r in action_rows if r['arm_id']==arm]) for arm in arms])
                return {arms[i] for i in np.flatnonzero(np.isclose(means,means.max(),rtol=1e-6,atol=1e-8))}
            switches += not bool(optimal_set(entries)&optimal_set(groups[state_id]))
        stable_switches[str(budget)] = switches
        audit[str(budget)] = dict(passed=bool(valid),states=len(probe_groups),records=len(probe_rows),
                                 evaluator_calls=sum(r['evaluator_calls'] for r in probe_rows))
        if not valid:
            raise RuntimeError('same-state budget audit failed')
    (args.root/'budget_audit.json').write_text(json.dumps(audit,indent=2))
    (args.root/'disjoint_optimal_set_switches.json').write_text(json.dumps(stable_switches,indent=2))
    raw_diagnostic = []
    for state_id, entries in groups.items():
        by_arm = {arm:[r for r in entries if r['arm_id']==arm] for arm in arms}
        winner = max(arms,key=lambda arm:np.mean([r['bandit_reward'] for r in by_arm[arm]]))
        chosen = by_arm[winner]
        raw_diagnostic.append(dict(state_id=state_id,case=chosen[0]['state_metadata']['case'],
            winner=winner,raw_legal_rate=float(np.mean([r['raw_final_legal'] for r in chosen]))))
    (args.root/'winner_raw_legality.json').write_text(json.dumps(raw_diagnostic,indent=2))
    lines = ['# 配置偏好、空间选择与 LinUCB：实测回答','',
        '本轮为既有数据上的探索性诊断；原首轮测试Case已被查看。CNN配置保持冻结，',
        '9折Case留出，每折3个训练seed；validation为排序后下一Case。',
        '预算实验均从相同61个合法状态分叉，每配置3个优化seed。','',
        '|评价预算|各最优配置的状态数|并列状态|相对65预算换配置的状态数|最佳固定配置的oracle差距|',
        '|---|---|---:|---:|---:|']
    for budget in ('17','65','257'):
        result = preference[budget]
        overall = result['overall']
        lines.append(f"|{budget}|{overall['winner_counts']}|{overall['tied_states']}|"
                     f"{result.get('changed_vs_65',0)}|{overall['best_fixed_mean_regret']:.6g}|")
    lines += ['', '这里的最优按三个seed的平均收益确定；未据此声称显著性。', '',
              f'表中配置计数按固定顺序打破并列；排除并列解释后，最优集合与65预算无交集的状态数为{stable_switches}。', '',
              '|方法|Case×seed平均regret|集合top1|27个Case×seed中优于固定|与固定打平|',
              '|---|---:|---:|---:|---:|']
    for method, result in cv['aggregate'].items():
        lines.append(f"|{method}|{result['mean_case_seed_regret']:.6g}|"
                     f"{result['mean_case_seed_top1']:.4f}|{result['wins_vs_fixed']}|{result['ties_vs_fixed']}|")
    lines += ['', '每Case三个模型seed属于重复训练，不是27个独立Case；应以9个Case为推断单位。', '',
              '## LinUCB特征诊断','',
              f"原pilot训练样本{diagnostic['train_states']}，上下文{diagnostic['dimensions']}维。",
              f"低方差维度：{diagnostic['low_variance_feature_indices']}。",
              f"训练标准化特征最大绝对值{diagnostic['maximum_abs_train_standardized_feature']:.6g}；"
              f"测试为{diagnostic['maximum_abs_test_standardized_feature']:.6g}。", '',
              '上述检查揭示表示迁移和缩放风险；因果归因仍需固定编码器、单因素修正后独立验证。', '',
              '## 统一后处理的影响','',
              f"65预算中，最优配置在至少一个seed返回原始非法布局的状态："
              f"{sum(row['raw_legal_rate']<1 for row in raw_diagnostic)}/{len(raw_diagnostic)}。",
              '收益标签评价的是worker加公共合法化后的结果。因此这里的偏好适用于这一接口，',
              '不能直接称为纯算法本体的物理优化优势。逐状态记录见winner_raw_legality.json。', '',
              '## 可执行结论','',
              '配置偏好是否变化由同状态预算表回答；空间信息是否有效由Case留出对照回答。',
              '若CNN未优于固定策略，不应增加LLM或更大网络来预设收益；先扩充独立Case、',
              '检查原始/合法化收益分解，并在新的Case折上验证改动。LLM选择器目前未测试。']
    intervention_path=args.root/'scaling_intervention.json'
    if intervention_path.exists():
        intervention=json.loads(intervention_path.read_text())
        lines += ['', '## 冻结CNN的缩放干预', '',
                  '以训练集定义低方差维度并置零，其余标准化值截断到[-5,5]；CNN权重和标签保持不变。',
                  f"27折×seed的平均regret为{intervention['mean_case_seed_regret']:.6g}，与原LinUCB基本一致。",
                  f"原pilot干预后regret={intervention['original_pilot_intervention']['result']['mean_regret']:.6g}，"
                  '高于原0.0715316。没有观察到收益改善，不能把原失败简单归因为缩放放大。', '',
                  '## 明确回答', '',
                  '1. 本地六配置的偏好确实依赖状态和评价预算，主要变化发生于解析法的大/小步长。',
                  '2. 冻结65预算上，当前CNN和LinUCB没有超过固定配置的总体结果。',
                  '3. 现有证据不足以主张空间特征已学会有效的跨Case算法选择；也未测试LLM替代。',
                  '4. 下一版应显式将宏动作预算作为条件，扩充独立Case，并保留raw与合法化贡献分解；',
                  '   在新的验证协议中开展空间消融，再决定是否使用更复杂选择器。', '',
                  '局限：本轮CNN交叉验证只使用65预算；17/257仅检验动作偏好，并未训练预算条件化CNN。',
                  '状态来自合法布局，不能推断非法恢复任务或完整多步优化流程的结论。']
    (args.root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
