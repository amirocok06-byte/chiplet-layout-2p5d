"""Exploratory train-only scaling intervention on frozen CNN checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .stage1_data import load_stage1_dataset
from .stage1_models import FittedCNN, SmallRewardCNN, LinUCB, configure_device
from .stage1_cnn_train import selection_report


def bounded_scale(train, other):
    mean, std = train.mean(0), train.std(0)
    active = std >= 1e-6
    mean[-1], std[-1], active[-1] = 0., 1., True
    value = np.clip((other-mean)/np.maximum(std,1e-6),-5.,5.)
    value[:,~active] = 0.
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--labels-root',type=Path,required=True)
    args=parser.parse_args()
    d=load_stage1_dataset(args.labels_root)
    maps=np.stack([s.maps for s in d.states]); scalars=np.stack([s.scalars for s in d.states])
    rewards=np.stack([s.rewards for s in d.states])
    configure_device('cuda')
    results=[]
    pilot_result=None
    for folder in sorted(args.root.glob('fold_*'))+[args.root.parent/'stage1_e2_8gb_pilot']:
        config=json.loads((folder/'config.json').read_text())
        saved=torch.load(folder/'cnn.pt',map_location='cpu',weights_only=True)
        network=SmallRewardCNN().to('cuda'); network.load_state_dict(saved['state_dict'])
        fitted=FittedCNN(network,**{k:v.numpy() for k,v in saved['transforms'].items()},audit={})
        _,x=fitted.predict(maps,scalars)
        indices={part:np.array([i for i,s in enumerate(d.states) if s.case_group in config[part+'_cases']])
                 for part in ('train','validation','test')}
        tx=x[indices['train']]; tr=rewards[indices['train']]
        mean,std=float(tr.mean()),max(float(tr.std()),1e-6)
        values={part:bounded_scale(tx,x[index]) for part,index in indices.items()}
        def replay(part,alpha):
            model=LinUCB(len(d.arm_ids),37,alpha)
            for features,targets in zip(values['train'],tr):
                for arm,target in enumerate(targets):
                    model.update(arm,features,(target-mean)/std)
            selected=[]
            for features,target in zip(values[part],rewards[indices[part]]):
                arm=model.select(features); selected.append(arm)
                model.update(arm,features,(target[arm]-mean)/std)
            return selection_report(rewards[indices[part]],selected)
        alpha=min((0.,.1,.5,1.),key=lambda a:replay('validation',a)['mean_regret'])
        result=replay('test',alpha)
        entry=dict(case=','.join(config['test_cases']),seed=config['seed'],alpha=alpha,result=result)
        if folder.name=='stage1_e2_8gb_pilot':
            pilot_result=entry
        else:
            results.append(entry)
        print(folder.name, result['mean_regret'],flush=True)
    output=dict(intervention='drop train-constant dimensions; clip train-standardized context to [-5,5]; frozen CNN',
                interpretation='exploratory post-result diagnostic; not a new confirmatory test',
                original_pilot_intervention=pilot_result,
                folds=results,mean_case_seed_regret=float(np.mean([r['result']['mean_regret'] for r in results])),
                mean_case_seed_top1=float(np.mean([r['result']['set_top1'] for r in results])))
    (args.root/'scaling_intervention.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='folds'},indent=2))


if __name__=='__main__': main()
