"""Native checks for mixed structure/chemical likelihoods and raw positives."""
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/eternafold'))
import joint_workflow as joint
import workflow as ef

spec = importlib.util.spec_from_file_location('ef_native_checks', REPO / 'tests/test_eternafold_workflow.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_raw_positives_follow_native_cutoff():
    raw = np.array([-2., 0., 1e-7, .2, 1.3, -999., np.nan])
    values, mask = joint.native_evidence('ACACACG', 'DMS', raw)
    assert mask.tolist() == [False, False, False, True, True, False, False]
    assert np.array_equal(values[mask], raw[mask].astype('<f4'))
    assert np.all(values[~mask] == -1)


def mixed_fixture(tmp):
    params = tmp / 'zero.params'
    helper.zero_params(params)
    sequence, pairs = 'GCAAAAGC', [8, 7, 0, 0, 0, 0, 2, 1]
    evidence = np.array([.1, .2, 1.2, 1.5, .9, .7, .2, .1], dtype='<f4')
    for name, structural in [('structure', True), ('chemical', False)]:
        pack = ef.PackWriter(tmp / (name + '.efpack'))
        if structural:
            pack.add(name, sequence, pairs=pairs, weight=.4)
        else:
            pack.add(name, sequence, channel=0, values=evidence, weight=.6)
        pack.finish()
        ef.pack_list(pack.path)
    inputs = tmp / 'mixed.lst'
    inputs.write_text('@EFPACK:' + str(tmp / 'structure.efpack') + '\n@EFPACK:' + str(tmp / 'chemical.efpack') + '\n')
    return params, inputs


def test_mixed_objective_and_gradient_equal_task_sum(tmp_path):
    params, inputs = mixed_fixture(tmp_path)
    fs, gs = helper.objective(tmp_path, 'struct_eval', tmp_path / 'structure.efpack.lst', params)
    fc, gc = helper.objective(tmp_path, 'chem_eval', tmp_path / 'chemical.efpack.lst', params)
    fm, gm = helper.objective(tmp_path, 'mixed_eval', inputs, params)
    assert math.isclose(fm, fs + fc, abs_tol=1e-10)
    assert max(abs(gm[k] - gs[k] - gc[k]) for k in gm) < 1e-10
    assert any(abs(v) > .01 for k, v in gm.items() if k.startswith('log_score_evidence'))
    assert any(abs(v) > .01 for k, v in gm.items() if not k.startswith('log_score_evidence'))


def test_mixed_input_still_rejects_unparsable_structures(tmp_path):
    params, inputs = mixed_fixture(tmp_path)
    pack = ef.PackWriter(tmp_path / 'bad.efpack')
    pack.add('bad', 'GAAAA', pairs=[5, 0, 0, 0, 1])
    pack.finish()
    bad_inputs = tmp_path / 'bad.lst'
    bad_inputs.write_text(inputs.read_text() + '@EFPACK:' + str(pack.path) + '\n')
    try:
        helper.objective(tmp_path, 'reject', bad_inputs, params)
    except subprocess.CalledProcessError as exc:
        assert 'Unannounced exclusion' in exc.stderr
    else:
        raise AssertionError('Invalid structural labels were silently accepted')


if __name__ == '__main__':
    import inspect
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith('test_')]
    for name, fn in tests:
        with tempfile.TemporaryDirectory(prefix='ef-joint-check-') as tmp:
            fn(Path(tmp)) if inspect.signature(fn).parameters else fn()
        print(name + ': PASS', flush=True)
    print(f'{len(tests)} joint checks passed', flush=True)
