import unittest
import numpy as np
import time
from random import randint
from src.algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedCollaborativeFiltering
from src.utils.utils import pearson_correlation_terms
class UserBasedCollaborativeFilteringTest(unittest.TestCase):
    def _test_similarity_terms(self, cf, user_id, another_user_id):
        terms = pearson_correlation_terms(cf.co_rated_between(user_id, another_user_id), cf.matrix[user_id], cf.matrix[another_user_id], cf.avg_rating(user_id), cf.avg_rating(another_user_id))
        self.assertEqual(cf.variance(user_id,another_user_id), terms[2])
        self.assertEqual(cf.variance(another_user_id,user_id), terms[1])
        self.assertEqual(cf.covariance_between(user_id, another_user_id), terms[0])
        self.assertEqual(cf.similarity_between(user_id, another_user_id), round(terms[3],5))

    def test_asserts(self):
        matrix = np.array([1])
        similarities = np.array([1])
        avg_ratings = {"teste":3}
        co_rated = np.array([1])
        cf = UserBasedCollaborativeFiltering(matrix, similarities, avg_ratings, co_rated)
        self.assertEqual(matrix, cf.matrix)
        self.assertEqual(similarities, cf.similarities())
        self.assertEqual(co_rated, cf.co_rated())
        self.assertEqual(avg_ratings, cf.avg_ratings())
        self.assertEqual(type(cf.similarities()), np.ndarray )
        self.assertEqual(type(cf.co_rated()), np.ndarray)
        self.assertEqual(type(cf.avg_ratings()), dict)
    
    def test_model_initialization(self):
        dimension = 10
        matrix = [[ randint(1,10) for _i in range(0,dimension)] for _c in range(0,dimension) ]
        cf = UserBasedCollaborativeFiltering(matrix)
        self.assertEqual(len(cf.similarities()), len(matrix))
        self.assertEqual(len(cf.co_rated()), len(matrix))
        self.assertEqual(len(cf.avg_ratings()), len(matrix))
        for i, tup in enumerate(cf.avg_ratings().items()):
            with self.subTest(i=i):
                self.assertEqual(tup[1], sum( matrix[tup[0]])/len(matrix[tup[0]]))
    
    def test_similarities(self):
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, 2, 9, None, 1],
            [None, 1, 9, None, None],
            [7, None, 1, None, 6],
        ]
        cf = UserBasedCollaborativeFiltering(matrix)
        self.assertLess(cf.similarity_between(0,1), cf.similarity_between(0,2)) #This is really weird
        self.assertEqual(cf.similarity_between(0,1), cf.similarity_between(0, 4))
        self.assertEqual(cf.similarity_between(0,3), 0)
        self.assertEqual(cf.similarity_between(1,4), 1)
    
    def test_new_rating(self):
        user_id = 3
        another_user_id = 4
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, 2, 9, None, 1],
            [None, None, 2, None, None],
            [7, None, 2, None, 6],
        ]
        cf = UserBasedCollaborativeFiltering(matrix)
        self.assertNotIn(0, cf.co_rated_between(user_id, another_user_id))
        cf.new_stream(user_id, 0, 7)
        self._test_similarity_terms(cf, user_id, another_user_id)
        cf.new_stream(user_id, 4, 6)
        self._test_similarity_terms(cf, user_id, another_user_id)
        cf.new_stream(user_id, 1, 8)
        self._test_similarity_terms(cf, user_id, another_user_id)
        
    #TODO
    # def test_update_rating(self):
    #     user_id = 0
    #     another_user_id = 4
    #     matrix = [
    #         [1, None, 1, None, 1],
    #         [7, None, 1, None, 6],
    #         [None, 2, 9, None, 1],
    #         [None, None, 1, None, None],
    #         [7, None, 1, None, 6],
    #     ]
    #     cf = UserBasedCollaborativeFiltering(matrix)
    #     print("")
    #     print(cf.similarity_terms_between(user_id, another_user_id))
    #     cf.new_stream(user_id, 0, 7)
    #     print(cf.similarity_terms_between(user_id, another_user_id))
    #     self._test_similarity_terms(cf, user_id, another_user_id)
        
            
if __name__ == '__main__':
    unittest.main()