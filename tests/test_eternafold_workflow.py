"""Native numerical checks for the EternaFold experiment adapters."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import tempfile

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ef_workflow', REPO / 'scripts/eternafold/workflow.py')
ef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ef)
BINARY = Path(os.environ.get('EF_TEST_BINARY', REPO / 'qsub_logs/eternafold/eternafold-traina-branches-20261022/bin/contrafold'))


def run(tmp, name, input_list, *, initial=None, env=None, maxiter=6):
    folder = tmp / name
    folder.mkdir(exist_ok=True)
    command = [str(BINARY), 'train', '--numdatasources', '5', '--regularize', '.01',
               '--maxiter', str(maxiter), '--hyperparam_data', '.5', '--examplefile', str(input_list)]
    if initial is not None:
        command += ['--initweights', str(initial)]
    subprocess.run(command, cwd=folder, env={**os.environ, **(env or {})},
                   capture_output=True, text=True, check=True, timeout=90)
    return folder


def objective(tmp, name, lst, params):
    out = tmp / (name + '.value')
    run(tmp, name, lst, initial=params, env={'EF_OBJECTIVE_OUT': str(out)})
    return float(out.read_text()), ef.read_params(str(out) + '.gradient')


def zero_params(path):
    original = REPO / 'reports/eternafold-feasibility-20261022/evidence/pretrain_A_5/optimize.params.final'
    params = {k: 0. for k in ef.read_params(original)}
    save_params(path, params)
    return params


def save_params(path, params):
    path.write_text(''.join(f'{k} {v:.17g}\n' for k, v in params.items()))


def test_positive_transform_masks_before_conversion():
    values, mask = ef.positive_evidence('ACGUA', 'DMS', np.array([-2., 0., 10., -999., np.nan]))
    assert mask.tolist() == [True, True, False, False, False]
    assert 1e-5 < values[0] < values[1]
    assert np.all(values[~mask] == -1)


def test_structural_audit_does_not_turn_invalid_pairs_into_unpaired():
    assert ef.pair_audit('GAAAC', [5, 0, 0, 0, 1]) == []
    assert 'pair_distance_lt_4' in ef.pair_audit('GAAC', [4, 0, 0, 1])
    assert 'noncanonical_pair' in ef.pair_audit('GAAAA', [5, 0, 0, 0, 1])


def test_packed_weighted_objective_and_gradient_equal_text_inputs(tmp_path):
    params = tmp_path / 'initial.params'
    zero_params(params)
    sequence = 'GCAAAAGC'
    pack = ef.PackWriter(tmp_path / 'weighted.efpack')
    native_values = []
    for i, (value, weight) in enumerate([(0.4, .25), (1.2, .75)]):
        values = np.full(len(sequence), value, dtype='<f4')
        pack.add(f'p{i}', sequence, weight=weight, channel=0, values=values)
        text = tmp_path / f'text{i}.bpseq'
        text.write_text(''.join(f'{j} {b} e5 {float(v):.17g} -1 -1 -1 -1\n'
                                for j, (b, v) in enumerate(zip(sequence, values), 1)))
        lst = tmp_path / f'text{i}.lst'
        lst.write_text(str(text) + '\n')
        native_values.append((weight, *objective(tmp_path, f'text{i}', lst, params)))
    pack.finish()
    packed_value, packed_gradient = objective(tmp_path, 'packed', ef.pack_list(pack.path), params)
    assert math.isclose(packed_value, sum(w*f for w, f, _ in native_values), abs_tol=1e-10)
    for key, value in packed_gradient.items():
        assert math.isclose(value, sum(w*g[key] for w, _, g in native_values), abs_tol=1e-10)


def test_negative_chemical_nll_retains_gradient_and_matches_finite_difference(tmp_path):
    params_path = tmp_path / 'initial.params'
    params = zero_params(params_path)
    # Identical narrow Gamma distributions for paired/unpaired; density at one > 1.
    for key in params:
        if key.startswith('log_score_evidence0_k_'):
            params[key] = math.log(100.)
        if key.startswith('log_score_evidence0_theta_'):
            params[key] = math.log(.01)
    save_params(params_path, params)
    pack = ef.PackWriter(tmp_path / 'density.efpack')
    pack.add('density', 'GCAAAAGC', channel=0, values=np.ones(8))
    pack.finish()
    lst = ef.pack_list(pack.path)
    value, gradient = objective(tmp_path, 'negative', lst, params_path)
    assert value < 0
    candidates = [k for k in gradient if abs(gradient[k]) > .01]
    assert candidates
    for i, key in enumerate(candidates[:4]):
        eps = 1e-5
        plus, minus = dict(params), dict(params)
        plus[key] += eps
        minus[key] -= eps
        pp, mp = tmp_path / f'plus{i}.params', tmp_path / f'minus{i}.params'
        save_params(pp, plus)
        save_params(mp, minus)
        fp, _ = objective(tmp_path, f'plus{i}', lst, pp)
        fm, _ = objective(tmp_path, f'minus{i}', lst, mp)
        assert math.isclose(gradient[key], (fp-fm)/(2*eps), rel_tol=2e-5, abs_tol=2e-5)


def test_exact_lbfgs_resume_matches_uninterrupted_training(tmp_path):
    source = json.loads((REPO / 'reports/eternafold-feasibility-20261022/evidence/smoke_results.json').read_text())
    pack = ef.PackWriter(tmp_path / 'a.efpack')
    for i, path in enumerate(source['A_examples']):
        sequence, pairs = ef.read_bpseq(path)
        pack.add(f'A{i}', sequence, pairs=pairs, weight=.25)
    pack.finish()
    lst = ef.pack_list(pack.path)
    full = run(tmp_path, 'full', lst)
    checkpoint = tmp_path / 'split/state.bin'
    env = {'EF_LBFGS_STATE': str(checkpoint), 'EF_SIGNATURE': 'resume-regression', 'EF_SLICE_ITERATIONS': '2'}
    split = run(tmp_path, 'split', lst, env=env)
    assert checkpoint.exists()
    assert not Path(str(checkpoint) + '.done').exists()
    for _ in range(4):
        split = run(tmp_path, 'split', lst, env=env)
        if Path(str(checkpoint) + '.done').exists():
            break
    assert Path(str(checkpoint) + '.done').exists()
    a, b = ef.read_params(full / 'optimize.params.final'), ef.read_params(split / 'optimize.params.final')
    assert max(abs(a[k]-b[k]) for k in a) < 1e-12


def test_structural_gradient_with_unequal_evidence_distributions(tmp_path):
    path = tmp_path / 'initial.params'
    params = zero_params(path)
    for key in params:
        if key.startswith('log_score_evidence0_theta_') and key.endswith('0'):
            params[key] = -.8
    save_params(path, params)
    pack = ef.PackWriter(tmp_path / 'unequal.efpack')
    values = np.array([.1, .2, 1.3, 1.5, 1.1, .8, .2, .1], dtype='<f4')
    pack.add('unequal', 'GCAAAAGC', channel=0, values=values)
    pack.finish()
    lst = ef.pack_list(pack.path)
    _, gradient = objective(tmp_path, 'unequal', lst, path)
    keys = sorted((k for k in params if not k.startswith('log_score_evidence')),
                  key=lambda k: abs(gradient[k]), reverse=True)[:3]
    assert abs(gradient[keys[0]]) > .01
    for i, key in enumerate(keys):
        pp, mp = dict(params), dict(params)
        pp[key] += 1e-4
        mp[key] -= 1e-4
        plus, minus = tmp_path / f'sp{i}.params', tmp_path / f'sm{i}.params'
        save_params(plus, pp)
        save_params(minus, mp)
        fp, _ = objective(tmp_path, f'sp{i}', lst, plus)
        fm, _ = objective(tmp_path, f'sm{i}', lst, minus)
        assert math.isclose(gradient[key], (fp-fm)/2e-4, rel_tol=2e-4, abs_tol=2e-4)


if __name__ == '__main__':
    import inspect
    if not BINARY.exists():
        raise SystemExit('Build the pinned EternaFold adapter first')
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith('test_')]
    for name, fn in tests:
        with tempfile.TemporaryDirectory(prefix='ef-native-test-') as tmp:
            fn(Path(tmp)) if inspect.signature(fn).parameters else fn()
        print(name + ': PASS', flush=True)
    print(f'{len(tests)} checks passed', flush=True)
