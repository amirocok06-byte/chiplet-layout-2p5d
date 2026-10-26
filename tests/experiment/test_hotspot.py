"""Subprocess fixtures here are TEST ONLY; their output is not calibration data."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from chiplet_layout_experiment.hotspot import (
    HotSpotError, HotSpotRunner, fit_calibration, parse_temperatures,
)
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def problem():
    return TracedProblem(
        ("a", "b"), np.array([10., 8.]), np.array([[.2, .2], [.2, .2]]),
        np.array([4., 8.]), np.array([[0, 1]]),
        np.array([[[.1, .1], [.1, .1]]]), np.array([[.1, .2], [.6, .5]]),
    )


@pytest.fixture
def runner(tmp_path):
    config = tmp_path / "package.config"
    config.write_text("-ambient 300\n-model_type block\n")
    fake = tmp_path / "TEST_ONLY_fake_hotspot.py"
    fake.write_text('''import sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
row = Path(args['-f']).read_text().splitlines()[0].split()
assert row[0] == 'a'
assert all(abs(a-b) < 1e-15 for a,b in zip(map(float,row[1:]), [.002,.0016,.001,.0016]))
trace = Path(args['-p']).read_text().splitlines()
assert trace[0].split()[:2] == ['a', 'b']
assert list(map(float, trace[1].split()[:2])) == [4., 8.]
assert all(float(v) == 0 for v in trace[1].split()[2:])
Path(args['-steady_file']).write_text('a 310.15\\nb 320.15\\niface_a 900\\n')
print('test-only simulator')
''')
    return HotSpotRunner([sys.executable, str(fake)], config, timeout=5)


def test_real_subprocess_protocol_and_provenance(runner, tmp_path):
    p = problem()
    sample = runner.measure(p, p.initial_positions, tmp_path / "run", case_id="case1", layout_id="l1")
    assert sample['peak_celsius'] == pytest.approx(47.)
    assert sample['temperatures_kelvin'] == {'a': 310.15, 'b': 320.15}
    assert sample['config_sha256'] == hashlib.sha256(runner.config.read_bytes()).hexdigest()
    assert sample['thermal_proxy_peak'] == p.evaluate(p.initial_positions)['thermal_proxy_peak']
    assert sample['proxy_version'] == p.VERSION
    saved = json.loads((tmp_path / 'run' / 'sample.json').read_text())
    assert saved == sample
    assert saved['command'] and saved['floorplan_sha256'] and saved['executable_sha256']
    with pytest.raises(FileExistsError):
        runner.measure(p, p.initial_positions, tmp_path / 'run', case_id='c', layout_id='l')


@pytest.mark.parametrize('positions', [
    [[.1, .2], [.15, .2]], [[-.01, .2], [.6, .5]],
    [[.1, .2], [.9, .5]], [[float('nan'), .2], [.6, .5]], [[.1, .2]],
])
def test_illegal_layout_rejected_before_launch(runner, tmp_path, positions):
    with pytest.raises(ValueError):
        runner.measure(problem(), positions, tmp_path / 'illegal', case_id='c', layout_id='l')
    assert not (tmp_path / 'illegal').exists()


@pytest.mark.parametrize('text', ['a nan\nb 300', 'a 0\nb 300', 'a 300', 'a 300\na 301\nb 302', 'a nope\nb 300'])
def test_temperature_validation(text):
    with pytest.raises(HotSpotError):
        parse_temperatures(text, ('a', 'b'))


@pytest.mark.parametrize('body,match', [
    ('import sys; print("failed", file=sys.stderr); sys.exit(7)', 'exit 7'),
    ('import time; time.sleep(4)', 'timed out'),
    ('print("no output")', 'steady'),
])
def test_subprocess_failure_and_timeout(runner, tmp_path, body, match):
    Path(runner.command[1]).write_text(body)
    runner.timeout = .2 if 'sleep' in body else 5
    with pytest.raises(HotSpotError, match=match):
        runner.measure(problem(), problem().initial_positions, tmp_path / 'bad', case_id='c', layout_id='l')
    assert not (tmp_path / 'bad' / 'sample.json').exists()


def samples():
    return [dict(sample_id=str(i), case_id=f'c{i//2}', layout_id=f'l{i}',
                 layout_sha256=f'geometry{i}', thermal_proxy_peak=float(i),
                 peak_celsius=2.*i+30., config_sha256='config',
                 executable_sha256='exe', proxy_version='v1') for i in range(6)]


def test_fit_uses_training_only_and_reports_holdout_provenance():
    data = samples()
    data[-1]['peak_celsius'] += 10
    report = fit_calibration(data, split_by='case', held_out=['c2'])
    assert report['fit']['slope'] == pytest.approx(2.)
    assert report['fit']['intercept_celsius'] == pytest.approx(30.)
    assert report['held_out_metrics']['mae_celsius'] == pytest.approx(5.)
    assert report['held_out_metrics']['rmse_celsius'] == pytest.approx(np.sqrt(50))
    assert report['train_sample_ids'] == ['0', '1', '2', '3']
    assert report['held_out_sample_ids'] == ['4', '5']
    assert len(report['samples']) == 6
    assert report['samples_sha256']


def test_layout_holdout_and_no_geometry_leakage():
    data = samples()
    report = fit_calibration(data, split_by='layout', held_out=['l5'])
    assert report['held_out_sample_ids'] == ['5']
    data[-1]['layout_sha256'] = data[0]['layout_sha256']
    with pytest.raises(ValueError, match='leak'):
        fit_calibration(data, split_by='case', held_out=['c2'])


@pytest.mark.parametrize('mutation', ['config', 'executable', 'proxy', 'constant', 'nan', 'duplicate'])
def test_invalid_fit_rejected(mutation):
    data = samples()
    if mutation in ('config', 'executable'):
        data[-1][mutation + '_sha256'] = 'other'
    elif mutation == 'proxy':
        data[-1]['proxy_version'] = 'other'
    elif mutation == 'constant':
        for row in data:
            row['thermal_proxy_peak'] = 1.
    elif mutation == 'nan':
        data[0]['peak_celsius'] = float('nan')
    else:
        data[-1]['sample_id'] = data[0]['sample_id']
    with pytest.raises(ValueError):
        fit_calibration(data, split_by='case', held_out=['c2'])


def test_unknown_or_empty_holdout_rejected():
    for held_out in ([], ['absent'], ['c0', 'c1', 'c2']):
        with pytest.raises(ValueError):
            fit_calibration(samples(), split_by='case', held_out=held_out)


def test_dead_space_preserves_board_extent_and_has_zero_power(runner, tmp_path):
    p = problem()
    runner.measure(p, p.initial_positions, tmp_path / 'run', case_id='c', layout_id='l')
    rows = [line.split() for line in (tmp_path / 'run/layout.flp').read_text().splitlines()]
    geometry = np.array([[float(v) for v in row[1:]] for row in rows])
    assert (geometry[:, :2] > 0).all()
    assert np.sum(geometry[:, 0] * geometry[:, 1]) == pytest.approx(.010*.008)
    np.testing.assert_allclose((geometry[:, :2] + geometry[:, 2:]).max(axis=0), [.010, .008])
    for i in range(len(geometry)):
        lo = np.maximum(geometry[i, 2:], geometry[i+1:, 2:])
        hi = np.minimum(geometry[i, 2:] + geometry[i, :2], geometry[i+1:, 2:] + geometry[i+1:, :2])
        assert np.maximum(hi-lo, 0).prod(axis=1).sum() < 1e-20


@pytest.mark.parametrize('field,value', [
    ('names', ('a', 'a')), ('names', ('a b', 'c')), ('names', ('a', '__hs_gap_1')),
    ('powers', np.array([-1., 3.])), ('powers', np.array([float('inf'), 3.])),
    ('board_mm', np.array([0., 8.])), ('sizes', np.array([[0., .2], [.2, .2]])),
])
def test_invalid_problem_rejected(runner, tmp_path, field, value):
    p = problem()
    setattr(p, field, value)
    with pytest.raises(ValueError):
        runner.measure(p, p.initial_positions, tmp_path / 'bad', case_id='c', layout_id='l')
    assert not (tmp_path / 'bad').exists()


def test_config_bytes_are_frozen_at_runner_creation(runner, tmp_path):
    original = runner.config.read_bytes()
    runner.config.write_text('-ambient 999\n')
    sample = runner.measure(problem(), problem().initial_positions, tmp_path / 'frozen', case_id='c', layout_id='l')
    assert sample['config_sha256'] == hashlib.sha256(original).hexdigest()
    assert (tmp_path / 'frozen/hotspot.config').read_bytes() == original


@pytest.mark.parametrize('option', [
    '-grid_layer_file layers.lcf', '-init_file init.k', '-package_model_used 1', '-leakage_used 1',
])
def test_external_config_dependencies_rejected(tmp_path, option):
    config = tmp_path / 'config'
    config.write_text(option)
    with pytest.raises(ValueError):
        HotSpotRunner(['hotspot'], config)


def test_missing_executable_has_explicit_error(tmp_path):
    config = tmp_path / 'config'
    config.write_text('-ambient 300')
    runner = HotSpotRunner([str(tmp_path / 'missing')], config)
    with pytest.raises(HotSpotError, match='executable unavailable'):
        runner.measure(problem(), problem().initial_positions, tmp_path / 'bad', case_id='c', layout_id='l')


def test_cli_collects_saved_states_and_case_holdout(runner, tmp_path, monkeypatch):
    # The executable below is deliberately synthetic, for CLI plumbing ONLY.
    Path(runner.command[1]).write_text('''import sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
Path(args['-steady_file']).write_text('a 310.15\\nb 320.15\\n')
''')
    states_root = tmp_path / 'states'
    p = problem()
    for case, offsets in [('train', [.6, .4]), ('test', [.65])]:
        directory = states_root / case
        (directory / 'states').mkdir(parents=True)
        saved = {key: np.asarray(getattr(p, key)).tolist() for key in (
            'board_mm', 'sizes', 'powers', 'pin_indices', 'pin_offsets')}
        saved.update(names=list(p.names), source={'case': case})
        (directory / 'problem.json').write_text(json.dumps(saved))
        for i, offset in enumerate(offsets):
            positions = p.initial_positions.copy()
            positions[1, 0] = offset
            (directory / 'states' / f'{i}.json').write_text(json.dumps({'positions': positions.tolist()}))
    cli = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/calibrate_hotspot.py'))
    monkeypatch.setitem(cli['main'].__globals__, 'HotSpotRunner', lambda *a, **kw: runner)
    output = tmp_path / 'cli-output'
    result = cli['main']([
        '--states-root', str(states_root), '--executable', runner.command[1],
        '--config', str(runner.config), '--output', str(output), '--held-out', 'test'])
    assert result == 0
    report = json.loads((output / 'calibration.json').read_text())
    assert report['held_out_sample_ids'] == ['test/test/0']
    assert report['train_metrics']['count'] == 2
    assert all(s['source_state_sha256'] and s['source_problem_sha256'] for s in report['samples'])
    assert all(json.loads((Path(s['run_directory'])/'sample.json').read_text()) == s for s in report['samples'])
