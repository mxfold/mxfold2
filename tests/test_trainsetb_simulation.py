"""Scientific invariants of the RNAgym-calibrated synthetic experiment."""
import numpy as np
import unittest
from scipy.stats import ks_2samp

from scripts.rnagym.simulate_trainsetb_reactivity import (
    coverage_counts, simulate, structure_auc,
)
from mxfold2.rnagym import measurement_mask


def test_dms_coverage_is_fraction_of_all_bases_and_excludes_non_ac():
    bases = np.array(list('ACGUN' * 200))
    rates = {'A': .8, 'C': .4}
    allocation = coverage_counts(bases, 'DMS', .3, rates)
    assert sum(allocation.values()) == 300
    assert allocation['A'] > allocation['C']
    with unittest.TestCase().assertRaisesRegex(ValueError, 'applicable-base'):
        coverage_counts(bases, 'DMS', .41, rates)


def test_missing_values_zero_negative_reactivities_and_seed():
    bases = np.array(list('ACGUN' * 1000))
    paired = np.arange(len(bases)) % 2 == 0
    samples = {b: np.linspace(-.2, 2., 2000) for b in 'AC'}
    rates = {'A': .8, 'C': .8}
    first = simulate(bases, paired, 'DMS', .25, samples, rates, np.random.default_rng(17))
    second = simulate(bases, paired, 'DMS', .25, samples, rates, np.random.default_rng(17))
    np.testing.assert_array_equal(first, second)
    valid = measurement_mask(''.join(bases), 'DMS', first)
    assert valid.sum() == 1250
    assert (first[valid] < 0).any()
    assert (first[~np.isin(bases, list('AC'))] == -999).all()
    assert (first[~valid] == -999).all()
    empty = simulate(bases, paired, 'DMS', 0, samples, rates, np.random.default_rng(17))
    assert (empty == -999).all()


def test_empirical_marginal_preserved_with_noisy_structure_signal():
    bases = np.array(list('ACGU' * 10000))
    paired = np.arange(len(bases)) % 7 < 4
    samples = {b: np.random.default_rng(i).gamma(1.5, .3, 30000)
               for i, b in enumerate('ACGU')}
    values = simulate(bases, paired, '2A3', .61, samples,
                      dict.fromkeys('ACGU', .6), np.random.default_rng(42))
    assert (values > -900).sum() == 24400
    for base in 'ACGU':
        mask = (bases == base) & (values > -900)
        assert ks_2samp(values[mask], samples[base]).statistic < .001
        # Wu provides informative but imperfect labels, with overlapping tails.
        assert .6 < structure_auc(values[mask], paired[mask]) < .95


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_dms_coverage_is_fraction_of_all_bases_and_excludes_non_ac,
        test_missing_values_zero_negative_reactivities_and_seed,
        test_empirical_marginal_preserved_with_noisy_structure_signal,
    ))
