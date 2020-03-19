import unittest
from random import choice
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.lsh_neighborhood import LSHBased


class LSHBasedTest(unittest.TestCase):
    def test_asserts(self):
        matrix = [1]
        sign_mat = [1]
        buckets = [1]
        n_perms = 1
        n_bands = 1
        lh_cf = LSHBased(matrix, sign_mat, buckets, n_perms, n_bands)
        self.assertEqual(lh_cf.matrix, matrix)
        self.assertEqual(lh_cf.signature_matrix(), sign_mat)
        self.assertEqual(lh_cf.n_bands, n_bands)
        self.assertEqual(lh_cf.n_permutations, n_perms)
        self.assertEqual(lh_cf.buckets(), buckets)

    def test_model_initialization(self):
        dimension = 10
        matrix = [[choice([1, None]) for _ in range(dimension)] for _ in range(dimension)]
        lh_based = LSHBased(matrix)
        self.assertEqual(lh_based.n_permutations, len(lh_based.signature_matrix()))
        self.assertEqual(len(lh_based.signature_matrix().loc[0]), len(matrix[0]))
        self.assertNotEqual(len(lh_based.buckets()), 0)

    def test_recommendation(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 2, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        lh_based = LSHBased(matrix, n_perms=20)
        lh_based.new_stream((1,1))
        self.assertIn(1, lh_based.recommend(4))
