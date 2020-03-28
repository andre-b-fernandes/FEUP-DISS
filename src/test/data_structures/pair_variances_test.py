import unittest
from src.data_structures import PairVariances


class PairVariancesTest(unittest.TestCase):
    def test_init(self):
        pv = PairVariances()
        self.assertEqual(pv.first_variance, 0)
        self.assertEqual(pv.second_variance, 0)

    def test_set_variance(self):
        pv = PairVariances()
        pv.set_variance(0, 1, 0.5, 0.6)
        self.assertEqual(pv.first_variance, 0.6)
        self.assertEqual(pv.second_variance, 0.5)
        self.assertEqual(pv.variance(0, 1), 0.5)
        self.assertEqual(pv.variance(1, 0), 0.6)


if __name__ == "__main__":
    unittest.main()
