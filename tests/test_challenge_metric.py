import unittest

import numpy as np

from rare26_nct_tso.challenge_metric import (
    challenge_style_ppv,
    percentile_ranks,
    ppv_at_min_recall,
)


class ChallengeMetricTests(unittest.TestCase):
    def test_perfect_ranking_has_unit_ppv(self):
        labels = np.array([0] * 100 + [1] * 10)
        scores = np.array(list(range(100)) + list(range(100, 110)))
        self.assertEqual(ppv_at_min_recall(labels, scores), 1.0)

    def test_challenge_style_result_is_reproducible(self):
        labels = np.array([0] * 200 + [1] * 20)
        scores = np.linspace(0.0, 1.0, labels.size)
        first = challenge_style_ppv(labels, scores, iterations=20, seed=7)
        second = challenge_style_ppv(labels, scores, iterations=20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["sampled_positives_per_iteration"], 2)

    def test_percentile_ranks_average_ties(self):
        actual = percentile_ranks([1.0, 1.0, 3.0, 2.0])
        expected = np.array([0.25, 0.25, 0.875, 0.625])
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()

