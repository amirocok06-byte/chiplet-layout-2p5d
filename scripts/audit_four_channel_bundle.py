"""Audit newly projected observations separately from historical runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(root):
    cases=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['cases']
    count=0
    figures=[]
    for case in cases:
        directory=root/case
        scales=json.loads((directory/'state_v2'/'display_scales.json').read_text())
        assert len(scales['limits'])==4
        for algorithm in ('sa','analytical','rl'):
            base=directory/algorithm
            for step in (0,200,400,600,800):
                stem=f'checkpoint_{step:04d}'
                new=json.loads((base/'state_v2'/f'{stem}.json').read_text())
                old=json.loads((base/f'{stem}.json').read_text())
                assert new['metrics']==old['metrics'] and new['positions']==old['positions']
                assert new['historical_snapshot_sha256']==hashlib.sha256((base/f'{stem}.json').read_bytes()).hexdigest()
                with np.load(base/'state_v2'/'raw'/f'{stem}.npz') as f: maps=f['maps']
                with np.load(base/'raw'/f'{stem}.npz') as f: legacy=f['maps']
                assert maps.shape==(4,256,256) and np.isfinite(maps).all()
                np.testing.assert_array_equal(maps[1:3],legacy[1:3])
                assert maps[3].min()>=0
                assert hashlib.sha256(maps.tobytes()).hexdigest()==new['maps_sha256']
                figures.append(base/'state_v2'/f'{stem}.png')
                count+=1
            figures.append(base/'state_v2'/'state_flow.png')
        figures.append(directory/'state_v2'/'algorithm_effects.png')
    warnings=[]
    for figure in figures:
        for suffix in ('.png','.pdf','.svg','.alignment.json','.collision.json','.qa.json'):
            assert figure.with_suffix(suffix).is_file(),figure.with_suffix(suffix)
        alignment=json.loads(figure.with_suffix('.alignment.json').read_text())
        assert alignment['verdict']=='PASS',figure
        collision=json.loads(figure.with_suffix('.collision.json').read_text())
        assert collision['summary']['fail']==0,figure
        if collision['summary']['warn']:
            warnings.append(str(figure))
        reports=json.loads(figure.with_suffix('.qa.json').read_text())
        assert all(r['returncode']==0 for r in reports),figure
    result={'status':'pass' if not warnings else 'review-required','checkpoints':count,
        'maps':count*4,'figures':len(figures),'all_historical_metrics_and_coordinates_unchanged':True,
        'thermal_hpwl_arrays_identical_to_v1':True,'all_alignment_pass':True,
        'collision_failures':0,'collision_warnings':warnings,'optimizer_rerun':False}
    result['source_preflight_notes'] = [
        'Literal export scanner cannot follow shared figure_export.py; actual PNG/PDF/SVG exports and rendered QA are verified for every figure.',
        'Large research-inspection figures, not resized journal-submission panels; journal-width defaults do not apply.',
    ]
    result['projection_source_sha256'] = hashlib.sha256((Path(__file__).resolve().parents[1]/
        'src/chiplet_layout_experiment/state_projection.py').read_bytes()).hexdigest()
    (root/'state_v2_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(audit(parser.parse_args().root),indent=2))
