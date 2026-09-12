import unittest

from scripts.implicit_mle.correctness_check import DEFAULT_SEEDS, run_checks


class ImplicitMLECorrectnessTest(unittest.TestCase):
    """End-to-end toy checks shared across tests to keep runtime bounded."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_checks(
            DEFAULT_SEEDS,
            gradient_samples=1000,
            update_samples=500,
            train_samples=64,
            train_steps=80,
        )

    def test_exact_gradient_matches_numerical_gradient(self):
        result = self.report["gradient_sanity"]
        self.assertGreater(result["numerical_vs_exact_cosine"], 0.999)
        self.assertLess(result["numerical_vs_exact_relative_error"], 0.005)

    def test_imle_gradient_matches_exact_direction_across_five_seeds(self):
        result = self.report["gradient_sanity"]
        self.assertEqual(len(result["imle_per_seed"]), 5)
        self.assertGreater(result["imle_mean_cosine"], 0.95)
        self.assertLess(result["imle_mean_relative_error"], 0.25)

    def test_rank_imle_gradient_matches_exact_direction_across_five_seeds(self):
        result = self.report["rank_gradient_sanity"]
        self.assertEqual(len(result["imle_per_seed"]), 5)
        self.assertGreater(result["imle_mean_cosine"], 0.9)
        # I-MLE is biased for this nonlinear extension; record and bound the scale error.
        self.assertLess(result["imle_mean_relative_error"], 1.5)

    def test_rank_loss_prefers_correct_to_every_nondegenerate_wrong_structure(self):
        result = self.report["rank_loss"]
        self.assertTrue(result["correct_below_every_nondegenerate_wrong"])

    def test_rank_loss_prefers_correct_to_all_unpaired_structure(self):
        self.assertTrue(self.report["rank_loss"]["correct_below_all_unpaired"])

    def test_one_step_increases_target_score_and_margin_for_every_seed(self):
        self.assertTrue(all(item["pass"] for item in self.report["one_step_update"]))

    def test_perfect_fake_shape_is_learnable(self):
        result = self.report["fake_shape"]
        self.assertGreaterEqual(result["mean_f1"], 0.8)
        self.assertGreaterEqual(result["exact_recovery_rate"], 0.8)

    def test_wu_like_noisy_shape_partially_recovers_structure(self):
        result = self.report["wu_like_noisy_shape"]
        self.assertGreaterEqual(result["mean_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
