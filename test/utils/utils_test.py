import unittest
from utils import (
    cosine_similarity,
    covariance,
    variance,
    pearson_correlation_terms,
    pearson_correlation,
    knn,
    avg,
    increment_avg
)


class UtilsTest(unittest.TestCase):
    def test_cosine_similarity(self):
        sim = cosine_similarity(0, 5, 5)
        sim1 = cosine_similarity(1, 5, 5)
        sim2 = cosine_similarity(2, 5, 5)
        sim3 = cosine_similarity(3, 5, 5)
        sim4 = cosine_similarity(4, 5, 5)
        sim5 = cosine_similarity(5, 5, 5)
        self.assertAlmostEqual(sim, 0, delta=0.01)
        self.assertAlmostEqual(sim1, 0.2, delta=0.01)
        self.assertAlmostEqual(sim2, 0.4, delta=0.01)
        self.assertAlmostEqual(sim3, 0.6, delta=0.01)
        self.assertAlmostEqual(sim4, 0.8, delta=0.01)
        self.assertAlmostEqual(sim5, 1.0, delta=0.01)

    def test_covariance(self):
        cov = covariance(
            [0, 4],
            [8, None, None, None, 7],
            [7, None, 1, None, 6], 7.5, 7)
        self.assertEqual(cov, 0.5)

    def test_variance(self):
        var = variance([0, 4], [8, None, None, None, 7], 7.5)
        self.assertEqual(var, 0.5)

    def test_pearson_correlation_terms(self):
        cov, vf, vs, pc = pearson_correlation_terms(
            [0, 4],
            [8, None, None, None, 7],
            [7, None, 1, None, 6], 7.5, 7)
        self.assertEqual(cov, 0.5)
        self.assertEqual(vf, 0.5)
        self.assertEqual(vs, 1)
        self.assertAlmostEqual(pc, 0.707, delta=0.001)

    def test_pearson_correlation(self):
        corr = pearson_correlation(0.1, 0.5, 0.5)
        corr2 = pearson_correlation(0.1, 0.5, 0)
        self.assertAlmostEqual(corr, 0.2, delta=0.001)
        self.assertEqual(corr2, 0)

    def test_knn(self):
        elems = knn(0, [0, 1, 2, 3, 4, 5], 2, lambda x, y: x**2 - y)
        self.assertEqual(elems, [1, 0])

    def test_avg(self):
        average = avg([5, 3, None, None, 1])
        self.assertEqual(average, 3)

    def test_increment_average(self):
        increment = increment_avg(3, 6, [5, 3, None, None, 1, 6])
        self.assertEqual(increment, avg([5, 3, None, None, 1, 6]))


if __name__ == "__main__":
    unittest.main()
