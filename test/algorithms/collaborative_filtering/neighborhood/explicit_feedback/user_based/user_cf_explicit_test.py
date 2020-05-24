import unittest
from random import randint
from algorithms.collaborative_filtering.\
    neighborhood.explicit_feedback.user_based import UserBasedNeighborhood
from utils.utils import pearson_correlation_terms, avg


class UserBasedExplicitCFTest(unittest.TestCase):

    def _test_similarity_terms(self, cf, user_id, another_user_id):
        terms = pearson_correlation_terms(
            cf.co_rated_between(user_id, another_user_id),
            cf.matrix[user_id], cf.matrix[another_user_id],
            cf.avg_rating(user_id), cf.avg_rating(another_user_id))

        self.assertAlmostEqual(cf.variance(user_id, another_user_id),
                               round(terms[1], 5), delta=0.0001)
        self.assertAlmostEqual(cf.variance(another_user_id, user_id),
                               round(terms[2], 5), delta=0.0001)
        self.assertAlmostEqual(cf.covariance_between(user_id, another_user_id),
                               round(terms[0], 5), delta=0.0001)
        self.assertAlmostEqual(cf.similarity_between(user_id, another_user_id),
                               round(terms[3], 5), delta=0.0001)

    def _test_similarity_terms_users(self, cf, user_id):
        members = list(range(0, len(cf.matrix)))
        members.remove(user_id)
        for another_user_id in members:
            self._test_similarity_terms(cf, user_id, another_user_id)

    def test_model_initialization(self):
        dimension = 10
        matrix = [[randint(1, 10) for _i in range(0, dimension)]
                  for _c in range(0, dimension)]
        cf = UserBasedNeighborhood(matrix)
        self.assertEqual(len(cf.avg_ratings), len(matrix))
        self.assertEqual(len(cf.neighbors), len(matrix))
        for i in range(0, dimension):
            with self.subTest(i=i):
                self._test_similarity_terms_users(cf, i)
                self.assertEqual(avg(cf.matrix[i]), cf.avg_rating(i))

    def test_similarities(self):
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, 2, 9, None, 1],
            [None, 1, 9, None, None],
            [7, None, 1, None, 6],
        ]
        cf = UserBasedNeighborhood(matrix)
        self._test_similarity_terms_users(cf, 0)
        self._test_similarity_terms_users(cf, 1)
        self._test_similarity_terms_users(cf, 2)
        self._test_similarity_terms_users(cf, 3)
        self._test_similarity_terms_users(cf, 4)

    def test_new_rating(self):
        user_id = 2
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, None, 2, None, None],
            [None, 2, 9, None, 1],
            [7, None, 2, None, 6],
        ]
        cf = UserBasedNeighborhood(matrix)
        self._test_similarity_terms_users(cf, user_id)
        self.assertNotIn(0, cf.co_rated_between(user_id, 4))
        cf.new_rating((user_id, 0, 7))
        self._test_similarity_terms_users(cf, user_id)
        cf.new_rating((user_id, 4, 6))
        self._test_similarity_terms_users(cf, user_id)
        cf.new_rating((user_id, 1, 8))
        self._test_similarity_terms_users(cf, user_id)
        cf.new_rating((user_id, 3, 2))
        self._test_similarity_terms_users(cf, user_id)

    def test_predict_rating(self):
        matrix = [
            [8, None, None, None, 7, None, None, None, 3],
            [7, None, 1, None, 6, None, 9, None, 4],
            [None, None, 2, None, None, None, 9, None, None],
            [None, 2, 9, None, 1, None, 5, None, None],
            [7, None, 2, None, None, None, None, 8, None],
        ]
        cf = UserBasedNeighborhood(matrix, n_neighbors=2)
        self.assertEqual(cf.predict(0, 2), 1.5)

    def test_recommendation(self):
        user_id = 2
        matrix = [
            [8, None, None, None, 7, None, None, None, 3],
            [7, None, 1, None, 6, None, 9, None, 4],
            [None, None, 2, None, None, None, 9, None, None],
            [None, 2, 9, None, 1, None, 5, None, None],
            [7, None, 2, None, None, None, None, 8, None],
        ]
        cf = UserBasedNeighborhood(matrix, n_neighbors=2)
        self.assertIn(0, cf.recommend(user_id, 1))


if __name__ == '__main__':
    unittest.main()
