import unittest
from random import choice
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback.user_based import UserLSHMinHash


class LSHBasedTest(unittest.TestCase):
    def test_model_initialization(self):
        dimension_users = 10
        dimension_items = 15
        matrix = [[choice([1, None]) for _ in range(dimension_items)]
                  for _ in range(dimension_users)]
        lh_based = UserLSHMinHash(matrix)
        self.assertEqual(lh_based.n_permutations,
                         len(lh_based.signature_matrix))
        self.assertEqual(len(lh_based.signature_matrix[0]),
                         dimension_users)
        self.assertNotEqual(len(lh_based.buckets), 0)

    def test_recommendation(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        lh_based = UserLSHMinHash(matrix, n_perms=120)
        lh_based.new_rating((1, 1))
        self.assertIn(1, lh_based.recommend(4, 5))
