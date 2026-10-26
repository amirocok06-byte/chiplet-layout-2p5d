"""Export new CNN contexts linked to unchanged same-state action labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from chiplet_layout_experiment.state_projection import CHANNELS, STATE_VERSION, project_state
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def export(root):
    records = []
    for path in sorted(root.glob('Case*/problem.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        p = TracedProblem(names=tuple(data['names']), initial_positions=np.zeros((len(data['names']), 2)),
            **{key: np.array(data[key]) for key in ('board_mm','sizes','powers','pin_indices','pin_offsets')})
        destination = path.parent/'state_v2'
        destination.mkdir(exist_ok=True)
        for state_file in sorted((path.parent/'states').glob('*.json')):
            state = json.loads(state_file.read_text(encoding='utf-8'))
            maps = project_state(p, np.array(state['positions']))
            raw = destination/'raw'
            raw.mkdir(exist_ok=True)
            np.savez_compressed(raw/f'{state_file.stem}.npz', maps=maps)
            record = {'state_version': STATE_VERSION, 'channels': CHANNELS,
                'map_shape': list(maps.shape), 'case': data['source']['case'],
                'source_state': f'../states/{state_file.name}',
                'source_state_sha256': hashlib.sha256(state_file.read_bytes()).hexdigest(),
                'maps_sha256': hashlib.sha256(maps.tobytes()).hexdigest(),
                'action_labels': '../labels.jsonl', 'input_checksum': state['checksum'],
                'case_split_group': data['source']['case'], 'evaluator_calls': 0,
                'existing_labels_unchanged': True}
            (destination/state_file.name).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
            records.append(record)
    manifest = {'state_version': STATE_VERSION, 'contexts': len(records), 'records': records,
        'map_shape': [4,256,256], 'projection_source_sha256': hashlib.sha256(
            (Path(__file__).resolve().parents[1]/'src/chiplet_layout_experiment/state_projection.py').read_bytes()).hexdigest(),
        'note': 'Map exports only; no optimizer rerun, no new reward and no CNN training.'}
    (root/'state_v2_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'contexts': len(records), 'shape':[4,256,256], 'evaluator_calls':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    export(parser.parse_args().root)
