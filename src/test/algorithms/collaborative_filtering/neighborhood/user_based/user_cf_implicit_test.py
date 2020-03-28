import unittest
from random import choice
from src.utils.utils import cosine_similarity
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.user_based_cf import UserBasedImplicitCF


class UserBasedImplicitCFTest(unittest.TestCase):
    def _test_similarity_user(self, cf, user_id):
        users = range(len(cf.matrix))
        for another_user_id in users:
            number_rated_items_user = len(cf.co_rated_between(user_id,
                                                              user_id))
            number_rated_items_another_user = len(cf.co_rated_between(
                another_user_id, another_user_id))
            number_of_co_rated_items = len(cf.co_rated_between(
                user_id, another_user_id))
            sim = cosine_similarity(number_of_co_rated_items,
                                    number_rated_items_user,
                                    number_rated_items_another_user)
            self.assertAlmostEqual(cf.similarity_between(user_id,
                                                         another_user_id),
                                   sim, delta=0.0001)

    def test_model_initialization(self):
        dimension = 10
        matrix = [[choice([None, 1]) for _i in range(0, dimension)]
                  for _c in range(0, dimension)]
        cf = UserBasedImplicitCF(matrix)
        self.assertEqual(len(matrix), len(cf.neighbors()))
        for i in range(dimension):
            with self.subTest(i=i):
                self._test_similarity_user(cf, i)

    def test_similarities(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, 1, 1, None, 1],
            [None, 1, 1, None, None],
            [1, None, 1, None, 1],
        ]
        cf = UserBasedImplicitCF(matrix)
        self._test_similarity_user(cf, 0)
        self._test_similarity_user(cf, 1)
        self._test_similarity_user(cf, 2)
        self._test_similarity_user(cf, 3)
        self._test_similarity_user(cf, 4)

    def test_new_rating(self):
        user_id = 2
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, None, 2, None, None],
            [None, 2, 9, None, 1],
            [7, None, 2, None, 6],
        ]
        cf = UserBasedImplicitCF(matrix)
        self._test_similarity_user(cf, user_id)
        self.assertNotIn(0, cf.co_rated_between(user_id, 4))
        cf.new_rating((user_id, 0))
        self._test_similarity_user(cf, user_id)
        cf.new_rating((user_id, 4))
        self._test_similarity_user(cf, user_id)
        cf.new_rating((user_id, 1))
        self._test_similarity_user(cf, user_id)
        cf.new_rating((user_id, 3))
        self._test_similarity_user(cf, user_id)

    def test_recommendation(self):
        user_id = 2
        matrix = [
            [1, None, None, None, 1, None, None, None, 1],
            [1, None, 1, None, 1, None, 1, None, 1],
            [None, None, 1, None, None, None, 1, None, None],
            [None, 1, 1, None, 1, None, 1, None, None],
            [1, None, 1, None, None, None, None, 1, None],
        ]
        cf = UserBasedImplicitCF(matrix, n_neighbors=2)
        self.assertNotIn(user_id, cf.neighborhood_of(user_id))
        self.assertIn(1, cf.neighborhood_of(user_id))
        self.assertIn(3, cf.neighborhood_of(user_id))
        self.assertIn(8, cf.recommend(user_id, 3))


if __name__ == "main":
    unittest.main()
