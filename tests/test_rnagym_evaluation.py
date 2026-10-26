import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.rnagym.evaluate_rnagym import (
    chemical_metrics, unpaired_from_bpp, unpaired_from_edges,
)


class RNAgymEvaluationTest(unittest.TestCase):
    def test_triangular_probability_is_added_to_both_endpoints(self):
        np.testing.assert_allclose(unpaired_from_edges([(1, 4, .8)], 4), [.2, 1, 1, .2])

    def test_symmetric_entries_are_not_double_counted(self):
        np.testing.assert_allclose(
            unpaired_from_edges([(1, 4, .8), (4, 1, .8)], 4), [.2, 1, 1, .2])

    def test_all_partners_contribute(self):
        np.testing.assert_allclose(
            unpaired_from_edges([(1, 3, .3), (2, 3, .4)], 3), [.7, .6, .3])

    def test_native_sparse_text(self):
        with patch.object(Path, 'open', return_value=io.StringIO('1 G 4:0.800\n2 A\n3 A\n4 C\n')):
            np.testing.assert_allclose(unpaired_from_bpp(Path('unused'), 4), [.2, 1, 1, .2])

    def test_invalid_values_fail(self):
        for edge in [(0, 2, .5), (1, 5, .5), (1, 1, .5), (1, 2, float('nan')), (1, 2, 1.1)]:
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                unpaired_from_edges([edge], 4)
        with self.assertRaises(ValueError):
            unpaired_from_edges([(1, 2, .4), (2, 1, .7)], 2)

    def test_empty_edges(self):
        np.testing.assert_array_equal(unpaired_from_edges([], 3), np.ones(3))

    def test_metrics_match_official_sklearn_convention(self):
        true = np.array([.1, .2, .2, .4, .9, 1.2, np.nan])
        pred = np.array([.3, .7, .1, .8, .9, .7, .4])
        result = chemical_metrics(true, pred)
        true = np.clip(true[:-1], 0, 1)
        pred = np.clip(pred[:-1], 0, 1)
        # Exact oracle: AUC includes one positive/negative tie; TP=2,
        # TN=3, FP=0, FN=1 after independent median thresholding.
        self.assertAlmostEqual(result['auc'], 17 / 18)
        self.assertAlmostEqual(result['mcc'], 1 / np.sqrt(2))
        self.assertAlmostEqual(result['f1'], 29 / 35)


if __name__ == '__main__':
    unittest.main()
