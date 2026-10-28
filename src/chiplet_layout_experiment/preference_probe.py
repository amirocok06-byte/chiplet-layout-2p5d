"""Preference identifiability, same-state budget probes and grouped CNN checks."""
from __future__ import annotations
import argparse
import contextlib
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def preference_summary(rewards, arm_ids):
    rewards = np.asarray(rewards, dtype=float)
    oracle = rewards.max(1)
    winners = np.isclose(rewards, oracle[:, None], rtol=1e-6, atol=1e-8)
    best = int(np.argmax(rewards.mean(0)))
    return dict(states=len(rewards), winner_counts=dict(Counter(
        arm_ids[i] for i in rewards.argmax(1))),
        tied_states=int((winners.sum(1)>1).sum()),
        best_fixed_arm=arm_ids[best],
        best_fixed_mean_regret=float(np.mean(oracle-rewards[:, best])),
        mean_rewards=dict(zip(arm_ids, rewards.mean(0).tolist())))


def main():
    from .stage1_data import load_stage1_dataset, _problem_from_saved_case
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root, out = args.labels_root, args.output
    dataset = load_stage1_dataset(root)
    cases = sorted({s.case_group for s in dataset.states})
    protocol = dict(budgets=[17,65,257], seeds=[11,22,33], epochs=100,
                    optimizer_seeds=[20261025,20261026,20261027],
                    cases=cases, validation='next Case in sorted cyclic order',
                    test='leave one Case out; exploratory existing-data analysis',
                    original_test_cases_previously_inspected=True)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    base = np.stack([s.rewards for s in dataset.states])
    preference = {'65': dict(overall=preference_summary(base, dataset.arm_ids),
        by_case={case:preference_summary(np.stack([s.rewards for s in dataset.states
                                                if s.case_group==case]), dataset.arm_ids)
                 for case in cases})}
    from .counterfactual import sample_state, finite_catalog
    for budget in (17,257):
        values = []
        with (out/f'budget_{budget}_actions.jsonl').open('w',encoding='utf-8') as file:
            for index, state in enumerate(dataset.states):
                case_dir = root/state.case_group
                source_rows = [json.loads(line) for line in (case_dir/'labels.jsonl').read_text().splitlines()]
                source = next(row for row in source_rows if row['state_id']==state.state_id)
                positions = np.asarray(json.loads((case_dir/source['state_metadata']['state_file']).read_text())['positions'])
                problem = _problem_from_saved_case(case_dir, positions)
                rows = sample_state(problem, positions, finite_catalog(budget),
                                    seeds=protocol['optimizer_seeds'], state_metadata=source['state_metadata'])
                rewards = []
                for arm in dataset.arm_ids:
                    rewards.append(np.mean([r['bandit_reward'] for r in rows if r['arm_id']==arm]))
                values.append(rewards)
                for row in rows:
                    assert row['evaluator_calls']==budget
                    # Store full counterfactual action evidence, including returned coordinates.
                    file.write(json.dumps(row)+'\n')
                print(f'budget={budget} states={index+1}/{len(dataset.states)}',flush=True)
        array = np.asarray(values)
        preference[str(budget)] = dict(overall=preference_summary(array,dataset.arm_ids),
            changed_vs_65=int((array.argmax(1)!=base.argmax(1)).sum()),
            by_case={case:preference_summary(array[[i for i,s in enumerate(dataset.states)
                                                   if s.case_group==case]],dataset.arm_ids)
                     for case in cases})
    (out/'preferences.json').write_text(json.dumps(preference,indent=2))
    from .stage1_cnn_train import main as train
    results = []
    for index, case in enumerate(cases):
        validation = cases[(index+1)%len(cases)]
        training = [c for c in cases if c not in (case,validation)]
        for seed in protocol['seeds']:
            dest = out/f'fold_{case}_seed{seed}'
            with contextlib.redirect_stdout(io.StringIO()):
                train(['--labels-root',str(root),'--output',str(dest),
                       '--train-cases',*training,'--validation-cases',validation,
                       '--test-cases',case,'--device','cuda','--epochs','100','--seed',str(seed)])
            summary = json.loads((dest/'summary.json').read_text())
            audit = json.loads((dest/'audit.json').read_text())
            results.append(dict(case=case,seed=seed,summary=summary,
                                peak_reserved_bytes=audit['peak_reserved_bytes']))
            print(f'fold={case} seed={seed} complete',flush=True)
    methods = [k for k in results[0]['summary'] if k!='cnn_prediction']
    aggregate = {method:dict(
        mean_case_seed_regret=float(np.mean([r['summary'][method]['mean_regret'] for r in results])),
        mean_case_seed_top1=float(np.mean([r['summary'][method]['set_top1'] for r in results])),
        wins_vs_fixed=sum(r['summary'][method]['mean_regret']<r['summary']['fixed']['mean_regret']-1e-8 for r in results),
        ties_vs_fixed=sum(abs(r['summary'][method]['mean_regret']-r['summary']['fixed']['mean_regret'])<=1e-8 for r in results)
    ) for method in methods}
    (out/'cross_validation.json').write_text(json.dumps(dict(folds=results,aggregate=aggregate),indent=2))
    # Diagnose the preserved original pilot without refitting it or tuning on test.
    import torch
    from .stage1_models import SmallRewardCNN, FittedCNN
    pilot = Path(__file__).resolve().parents[2]/'docs/figures/stage1_e2_8gb_pilot'
    saved = torch.load(pilot/'cnn.pt',map_location='cpu',weights_only=True)
    network = SmallRewardCNN(); network.load_state_dict(saved['state_dict'])
    transforms = {k:v.numpy() for k,v in saved['transforms'].items()}
    fitted = FittedCNN(network,**transforms,audit={})
    config = json.loads((pilot/'config.json').read_text())
    def features(groups):
        states = [s for s in dataset.states if s.case_group in groups]
        return fitted.predict(np.stack([s.maps for s in states]),np.stack([s.scalars for s in states]))[1]
    tx, qx = features(config['train_cases']), features(config['test_cases'])
    sd = tx.std(0); mean = tx.mean(0)
    scaled = (qx-mean)/np.maximum(sd,1e-6)
    diagnostic = dict(train_states=len(tx),dimensions=tx.shape[1],
        low_variance_feature_indices=np.where(sd[:-1]<1e-6)[0].tolist(),
        maximum_abs_test_standardized_feature=float(np.abs(scaled[:,:-1]).max()),
        maximum_abs_train_standardized_feature=float(np.abs((tx-mean)/np.maximum(sd,1e-6))[:,:-1].max()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'pilot_feature_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    print(json.dumps(dict(preferences=preference,aggregate=aggregate,diagnostic=diagnostic),indent=2))


if __name__=='__main__':
    main()
