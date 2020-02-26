import unittest
import numpy as np
from random import randint
from src.algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedCollaborativeFiltering

class UserBasedCollaborativeFilteringTest(unittest.TestCase):
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
        
        
if __name__ == '__main__':
    unittest.main()