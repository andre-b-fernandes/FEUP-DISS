import unittest
from random import randint
from algorithms.collaborative_filtering\
    .matrix_factorization.explicit_feedback import MFExplicitSGD


class MFExplicitSGDTest(unittest.TestCase):

    def test_initialization(self):
        dimension = 10
        matrix = [[randint(1, 10) for _i in range(0, dimension)]
                  for _c in range(0, dimension)]
        cf = MFExplicitSGD(matrix, lf=4)
        self.assertEqual(len(cf.matrix), dimension)
        lr = 0.01
        reg = 0.1
        self.assertEqual(cf.learning_rate, lr)
        self.assertEqual(cf.reg_factor, reg)

    def test_empty_model(self):
        cf = MFExplicitSGD()
        cf.predict(196, 203)
        self.assertEqual(len(cf.u), 197)
        self.assertEqual(len(cf.v), cf.latent_factors)
        self.assertEqual(len(cf.u[196]), cf.latent_factors)
        self.assertEqual(len(cf.v[1]), 204)

    def test_recommendation(self):
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, 2, 9, None, 1],
            [None, 1, 8, None, None],
            [7, None, 1, None, 6],
        ]
        cf = MFExplicitSGD(matrix)
        self.assertIn(0, cf.recommend(3, 3))


if __name__ == "__main__":
    unittest.main()
