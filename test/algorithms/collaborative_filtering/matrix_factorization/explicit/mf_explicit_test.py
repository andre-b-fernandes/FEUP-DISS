import unittest
from random import randint
from utils import avg
from algorithms.collaborative_filtering\
    .matrix_factorization.explicit_feedback import MatrixFactorizationExplicit


class MatrixFactorizationExplicitTest(unittest.TestCase):

    def test_initialization(self):
        dimension = 10
        matrix = [[randint(1, 10) for _i in range(0, dimension)]
                  for _c in range(0, dimension)]
        cf = MatrixFactorizationExplicit(matrix, lf=4)
        self.assertEqual(len(cf.matrix), dimension)
        elements = [
            element for row in cf.preprocessed_matrix() for element in row]
        self.assertAlmostEqual(avg(elements), 0, delta=0.00001)

    def test_predict(self):
        dimension = 10
        matrix = [[randint(1, 10) for _i in range(0, dimension)]
                  for _c in range(0, dimension)]
        cf = MatrixFactorizationExplicit(matrix, lf=4)
        for user_id in range(dimension):
            with self.subTest(i=user_id):
                avg_user = avg(matrix[user_id])
                for item_id in range(len(matrix[user_id])):
                    avg_item = avg(cf.matrix.col(item_id))
                    with self.subTest(i=item_id):
                        prep = cf.predict_prep(user_id, item_id)
                        real = prep + 0.5*(avg_user + avg_item)
                        self.assertEqual(cf.predict(user_id, item_id), real)

    def test_empty_model(self):
        cf = MatrixFactorizationExplicit()
        cf.predict_prep(196, 203)
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
        cf = MatrixFactorizationExplicit(matrix)
        self.assertIn(0, cf.recommend(3, 1))


if __name__ == "__main__":
    unittest.main()
