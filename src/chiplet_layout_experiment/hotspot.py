"""Auditable HotSpot block-model runs and grouped affine proxy calibration.

No simulator fallback is provided. Coordinates follow TracedProblem: fractions
of board axes, with physical board dimensions in millimetres and power in W.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np


class HotSpotError(RuntimeError):
    """Simulator execution or output failed; no calibration sample was made."""


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _write_json(path: Path, value):
    path.write_text(_json(value) + '\n', encoding='utf-8')


def parse_temperatures(text: str, names) -> dict[str, float]:
    """Read named steady-state Kelvin rows, ignoring package/internal nodes."""
    expected = set(names)
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in expected:
            continue
        name = fields[0]
        if name in result or len(fields) != 2:
            raise HotSpotError(f'duplicate or malformed steady temperature: {line}')
        try:
            value = float(fields[1])
        except ValueError as exc:
            raise HotSpotError(f'invalid Kelvin temperature: {line}') from exc
        if not math.isfinite(value) or value <= 0:
            raise HotSpotError(f'invalid Kelvin temperature: {line}')
        result[name] = value
    if set(result) != expected:
        raise HotSpotError(f'missing steady temperatures: {sorted(expected - result.keys())}')
    return result


def _geometry(problem, positions):
    x = np.asarray(positions, dtype=float)
    sizes = np.asarray(problem.sizes, dtype=float)
    board = np.asarray(problem.board_mm, dtype=float)
    power = np.asarray(problem.powers, dtype=float)
    n = len(problem.names)
    if (not n or x.shape != (n, 2) or sizes.shape != (n, 2)
            or board.shape != (2,) or power.shape != (n,)):
        raise ValueError('invalid geometry/power array shapes')
    if (not all(np.isfinite(a).all() for a in (x, sizes, board, power))
            or (sizes <= 0).any() or (board <= 0).any() or (power < 0).any()):
        raise ValueError('geometry must be finite and positive; powers finite and nonnegative')
    if len(set(problem.names)) != n or any(
        not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name.startswith('__hs_')
        for name in problem.names
    ):
        raise ValueError('chiplet names must be unique safe tokens, without __hs_ prefix')
    if (x < 0).any() or (x + sizes > 1. + 1e-12).any():
        raise ValueError('illegal layout: outside board')
    # Clip only roundoff at the upper boundary, never repair an illegal layout.
    x = np.minimum(x, 1. - sizes)
    for i in range(n):
        if (np.minimum(x[i] + sizes[i], x[i+1:] + sizes[i+1:])
                - np.maximum(x[i], x[i+1:]) > 0).all(axis=1).any():
            raise ValueError('illegal layout: overlapping chiplets')
    return x, sizes, board, power


def _floorplan(problem, x, sizes, board, power):
    """Tile board dead space with zero-power silicon so extents stay fixed."""
    blocks = [(name, *size, *pos, float(watts))
              for name, size, pos, watts in zip(problem.names, sizes, x, power)]
    edges = sorted(set([0., 1., *x[:, 0], *(x[:, 0] + sizes[:, 0])]))
    for left, right in zip(edges, edges[1:]):
        middle = (left + right) / 2
        intervals = sorted((pos[1], pos[1] + size[1]) for pos, size in zip(x, sizes)
                           if pos[0] <= middle < pos[0] + size[0])
        bottom = 0.
        for lo, hi in intervals + [(1., 1.)]:
            if lo > bottom:
                blocks.append((f'__hs_gap_{len(blocks)}', right-left, lo-bottom,
                               left, bottom, 0.))
            bottom = max(bottom, hi)
    lines = []
    for name, width, height, left, bottom, _ in blocks:
        values = np.array([width, height, left, bottom]) * board[[0, 1, 0, 1]] / 1000.
        lines.append(name + ' ' + ' '.join(format(v, '.17g') for v in values))
    floorplan = '\n'.join(lines) + '\n'
    trace = ' '.join(b[0] for b in blocks) + '\n'
    trace += ' '.join(format(b[-1], '.17g') for b in blocks) + '\n'
    return floorplan, trace


class HotSpotRunner:
    """Run a native executable command or Linux executable via WSL.

    command is an argument list (e.g. [executable], or test-only Python fixture).
    WSL mode takes ONE Windows-accessible executable path, translating all run
    paths itself. Config must be standalone: external layer/package/init files
    and leakage are rejected to keep the recorded config hash meaningful.
    """

    def __init__(self, command, config, *, timeout=60., wsl_distribution=None):
        if not command or isinstance(command, (str, bytes)):
            raise ValueError('command must be a nonempty argument list')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        self.command = [str(v) for v in command]
        self.config = Path(config).resolve()
        self.timeout = timeout
        self.wsl_distribution = wsl_distribution
        if wsl_distribution and len(self.command) != 1:
            raise ValueError('WSL mode requires a single executable path')
        self.config_bytes = self.config.read_bytes()
        options = {}
        for line in self.config_bytes.decode('utf-8-sig').splitlines():
            fields = line.split('#')[0].split()
            if len(fields) >= 2:
                options[fields[0]] = fields[1]
        for key in ('-grid_layer_file', '-init_file'):
            if options.get(key, '(null)') != '(null)':
                raise ValueError(f'{key} external files are unsupported by standalone calibration')
        if options.get('-package_model_used', '0') != '0' or options.get('-leakage_used', '0') != '0':
            raise ValueError('external package model/leakage unsupported by standalone calibration')

    def _path(self, path):
        value = str(Path(path).resolve())
        if self.wsl_distribution:
            match = re.fullmatch(r'([A-Za-z]):[\\/](.*)', value)
            if not match:
                raise ValueError('WSL paths must be on a Windows drive')
            return '/mnt/' + match[1].lower() + '/' + match[2].replace('\\', '/')
        return value

    def measure(self, problem, positions, output_dir, *, case_id, layout_id):
        """Persist inputs/logs and return an auditable successful sample.

        Run directories must be new: stale steady files can never be reused.
        On failure logs/inputs remain, but sample.json is never created.
        """
        if not case_id or not layout_id:
            raise ValueError('case_id and layout_id are required')
        x, sizes, board, power = _geometry(problem, positions)
        floorplan, trace = _floorplan(problem, x, sizes, board, power)
        executable = Path(shutil.which(self.command[0]) or self.command[0]).resolve()
        try:
            executable_hash = _hash(executable.read_bytes())
        except OSError as exc:
            raise HotSpotError(f'executable unavailable: {executable}') from exc
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=False)
        (out / 'layout.flp').write_text(floorplan, encoding='utf-8')
        (out / 'power.ptrace').write_text(trace, encoding='utf-8')
        (out / 'hotspot.config').write_bytes(self.config_bytes)
        command = [str(executable), *self.command[1:]]
        if self.wsl_distribution:
            command = ['wsl', '-d', self.wsl_distribution, '--', self._path(executable)]
        command += ['-c', self._path(out / 'hotspot.config'), '-f', self._path(out / 'layout.flp'),
                    '-p', self._path(out / 'power.ptrace'),
                    '-steady_file', self._path(out / 'steady.k'), '-model_type', 'block']
        _write_json(out / 'invocation.json', {'command': command, 'timeout_seconds': self.timeout})
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, cwd=out, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            (out / 'stdout.log').write_bytes(exc.stdout or b'')
            (out / 'stderr.log').write_bytes(exc.stderr or b'')
            raise HotSpotError(f'HotSpot timed out after {self.timeout}s; logs: {out}') from exc
        except OSError as exc:
            raise HotSpotError(f'HotSpot could not start: {exc}') from exc
        (out / 'stdout.log').write_bytes(proc.stdout)
        (out / 'stderr.log').write_bytes(proc.stderr)
        if proc.returncode:
            raise HotSpotError(f'HotSpot exit {proc.returncode}: {proc.stderr.decode(errors="replace")[-2000:]}')
        try:
            temperatures = parse_temperatures((out / 'steady.k').read_text(), problem.names)
        except OSError as exc:
            raise HotSpotError(f'HotSpot did not produce a readable steady file: {out}') from exc
        geometry = {'names': list(problem.names), 'board_mm': board.tolist(),
                    'sizes': sizes.tolist(), 'positions': x.tolist()}
        layout_hash = _hash(_json(geometry).encode())
        sample = {
            'sample_id': f'{case_id}/{layout_id}', 'case_id': str(case_id), 'layout_id': str(layout_id),
            'layout_sha256': layout_hash, 'geometry': geometry, 'powers_watts': power.tolist(),
            'thermal_proxy_peak': float(problem._thermal(x)[0]), 'proxy_version': problem.VERSION,
            'temperatures_kelvin': temperatures, 'peak_celsius': max(temperatures.values()) - 273.15,
            'config_sha256': _hash(self.config_bytes), 'executable_sha256': executable_hash,
            'floorplan_sha256': _hash((out / 'layout.flp').read_bytes()),
            'power_trace_sha256': _hash((out / 'power.ptrace').read_bytes()),
            'steady_sha256': _hash((out / 'steady.k').read_bytes()),
            'command': command, 'run_directory': str(out),
            'elapsed_seconds': time.perf_counter() - started,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'model': 'HotSpot block; dead space = zero-power silicon; chiplet peak only',
            'effective_overrides': {'model_type': 'block'},
            'adapter_sha256': _hash(Path(__file__).read_bytes()),
        }
        _write_json(out / 'sample.json', sample)
        return sample


def fit_calibration(samples, *, split_by='case', held_out):
    """Fit y[C] = slope*x[proxy] + intercept on training groups ONLY.

    Explicit held-out cases are recommended. Layout mode holds out layout_id
    across cases. Physical geometry hashes must never cross either split.
    No refit on held-out data is performed; all rows/provenance are in report.
    """
    samples = list(samples)
    if split_by not in ('case', 'layout'):
        raise ValueError('split_by must be case or layout')
    field = split_by + '_id'
    groups = set(held_out)
    available = {s[field] for s in samples}
    if not groups or not groups <= available:
        raise ValueError('held-out groups must be nonempty and present in samples')
    ids = [s['sample_id'] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate sample IDs')
    for key in ('config_sha256', 'executable_sha256', 'proxy_version'):
        if len({s[key] for s in samples}) != 1:
            raise ValueError(f'calibration requires one frozen {key}')
    train = [s for s in samples if s[field] not in groups]
    test = [s for s in samples if s[field] in groups]
    if len(train) < 2 or not test:
        raise ValueError('need at least two training samples and one held-out sample')
    if {s['layout_sha256'] for s in train} & {s['layout_sha256'] for s in test}:
        raise ValueError('layout leakage across training and held-out samples')
    all_values = np.array([[s['thermal_proxy_peak'], s['peak_celsius']] for s in samples], float)
    if not np.isfinite(all_values).all():
        raise ValueError('calibration values must be finite')
    x = np.array([s['thermal_proxy_peak'] for s in train], float)
    y = np.array([s['peak_celsius'] for s in train], float)
    if np.ptp(x) <= np.finfo(float).eps * max(1., float(abs(x).max())):
        raise ValueError('training proxy is constant; slope is not identifiable')
    centre = x.mean()
    slope = float(np.dot(x-centre, y-y.mean()) / np.dot(x-centre, x-centre))
    intercept = float(y.mean() - slope*centre)

    def metrics(rows):
        error = np.array([slope*s['thermal_proxy_peak'] + intercept - s['peak_celsius'] for s in rows])
        return {'count': len(rows), 'mae_celsius': float(abs(error).mean()),
                'rmse_celsius': float(np.sqrt(np.mean(error**2)))}

    return {
        'schema_version': 1, 'target': 'peak chiplet steady-state Celsius',
        'fit': {'slope': slope, 'intercept_celsius': intercept, 'method': 'ordinary least squares',
                'training_proxy_range': [float(x.min()), float(x.max())]},
        'split_by': split_by, 'held_out_groups': sorted(groups),
        'train_sample_ids': [s['sample_id'] for s in train],
        'held_out_sample_ids': [s['sample_id'] for s in test],
        'train_metrics': metrics(train), 'held_out_metrics': metrics(test),
        'held_out_metrics_by_case': {
            case: metrics([s for s in test if s['case_id'] == case])
            for case in sorted({s['case_id'] for s in test})},
        'diagnostics': {
            'positive_slope': slope > 0,
            'held_out_extrapolation_count': sum(
                not float(x.min()) <= s['thermal_proxy_peak'] <= float(x.max()) for s in test)},
        'config_sha256': samples[0]['config_sha256'],
        'executable_sha256': samples[0]['executable_sha256'],
        'proxy_version': samples[0]['proxy_version'],
        'samples_sha256': _hash(_json(samples).encode()), 'samples': samples,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
