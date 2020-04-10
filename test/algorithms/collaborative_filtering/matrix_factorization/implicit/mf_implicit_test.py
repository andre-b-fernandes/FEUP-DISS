import unittest
from random import choice
from algorithms.collaborative_filtering\
    .matrix_factorization.implicit_feedback import MatrixFactorizationImplicit


class MatrixFactorizationExplicitTest(unittest.TestCase):

    def test_initialization(self):
        dimension = 10
        matrix = [[choice([1, None]) for _i in range(0, dimension)]
                  for _c in range(0, dimension)]
        cf = MatrixFactorizationImplicit(matrix, lf=4, lr=0.1, reg=0.5)
        self.assertEqual(len(cf.matrix), dimension)
        self.assertEqual(cf.reg_factor, 0.5)
        self.assertEqual(cf.learning_rate, 0.1)

    def test_empty_model(self):
        cf = MatrixFactorizationImplicit()
        cf.predict(196, 203)
        self.assertEqual(len(cf.u()), 197)
        self.assertEqual(len(cf.v()), cf.latent_factors)
        self.assertEqual(len(cf.u()[196]), cf.latent_factors)
        self.assertEqual(len(cf.v()[1]), 204)

    def test_recommendation(self):
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, 2, 9, None, 1],
            [None, 1, 8, None, None],
            [7, None, 1, None, 6],
        ]
        cf = MatrixFactorizationImplicit(matrix)
        self.assertIn(0, cf.recommend(3, 3))


if __name__ == "__main__":
    unittest.main()
