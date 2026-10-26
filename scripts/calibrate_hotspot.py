"""Measure saved legal states with real HotSpot and fit a grouped calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chiplet_layout_experiment.hotspot import HotSpotError, HotSpotRunner, fit_calibration
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states-root', type=Path, required=True,
                        help='tree of case directories containing problem.json and states/*.json')
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='one frozen standalone HotSpot config')
    parser.add_argument('--output', type=Path, required=True, help='new output directory; never overwritten')
    parser.add_argument('--wsl-distribution', help='Windows only; e.g. Ubuntu-24.04-G')
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--layouts-per-case', type=int, default=3)
    parser.add_argument('--case', action='append', help='optional case IDs to include')
    parser.add_argument('--split-by', choices=('case', 'layout'), default='case')
    parser.add_argument('--held-out', action='append', required=True, help='case/layout ID, repeatable')
    args = parser.parse_args(argv)
    if args.layouts_per_case < 1:
        parser.error('--layouts-per-case must be positive')
    runner = HotSpotRunner([str(args.executable.resolve())], args.config,
                          timeout=args.timeout, wsl_distribution=args.wsl_distribution)
    problems = sorted(args.states_root.resolve().rglob('problem.json'))
    if not problems:
        parser.error('no problem.json under --states-root')
    args.output.mkdir(parents=True, exist_ok=False)
    samples, rejected, counts = [], [], {}
    for problem_path in problems:
        saved = json.loads(problem_path.read_text(encoding='utf-8-sig'))
        case_id = str(saved.get('source', {}).get('case', problem_path.parent.name))
        if args.case and case_id not in args.case:
            continue
        for state_path in sorted((problem_path.parent / 'states').glob('*.json')):
            if counts.get(case_id, 0) >= args.layouts_per_case:
                break
            state = json.loads(state_path.read_text(encoding='utf-8-sig'))
            # A relative parent distinguishes geometry variants of one case.
            layout_id = (str(problem_path.parent.relative_to(args.states_root.resolve()))
                         .replace('\\', '/') + '/' + state_path.stem)
            positions = np.asarray(state['positions'], dtype=float)
            problem = TracedProblem(
                names=tuple(saved['names']), board_mm=np.asarray(saved['board_mm']),
                sizes=np.asarray(saved['sizes']), powers=np.asarray(saved['powers']),
                pin_indices=np.asarray(saved['pin_indices']), pin_offsets=np.asarray(saved['pin_offsets']),
                initial_positions=positions,
            )
            try:
                sample = runner.measure(problem, positions, args.output / f'run_{len(samples):04d}_{len(rejected):04d}',
                                        case_id=case_id, layout_id=layout_id)
            except (ValueError, HotSpotError) as exc:
                rejected.append({'case_id': case_id, 'layout_id': layout_id, 'reason': str(exc)})
                print(f'rejected {case_id}/{state_path.stem}: {exc}', file=sys.stderr, flush=True)
                continue
            sample['source_problem'] = str(problem_path)
            sample['source_state'] = str(state_path)
            sample['source_problem_sha256'] = hashlib.sha256(problem_path.read_bytes()).hexdigest()
            sample['source_state_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
            write_json(Path(sample['run_directory']) / 'sample.json', sample)
            samples.append(sample)
            counts[case_id] = counts.get(case_id, 0) + 1
            print(f'{case_id} {state_path.stem}: {sample["peak_celsius"]:.4f} C', flush=True)
    write_json(args.output / 'collection.json', {'samples': samples, 'rejected': rejected})
    try:
        report = fit_calibration(samples, split_by=args.split_by, held_out=args.held_out)
    except ValueError as exc:
        print(f'Calibration not fitted: {exc}; collection preserved at {args.output}', file=sys.stderr)
        return 2
    report['rejected'] = rejected
    report['selection'] = {'method': 'sorted paths, first N successful legal states per case',
                           'layouts_per_case': args.layouts_per_case,
                           'included_cases': args.case, 'states_root': str(args.states_root.resolve())}
    write_json(args.output / 'calibration.json', report)
    print(json.dumps({'fit': report['fit'], 'held_out_metrics': report['held_out_metrics']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
