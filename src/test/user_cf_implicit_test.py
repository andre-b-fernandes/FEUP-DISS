import unittest
from random import choice
from src.utils.utils import cosine_similarity
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.user_based_cf import UserBasedImplicitCF

class UserBasedImplicitCFTest(unittest.TestCase):
    def test_asserts(self):
        matrix = [1]
        similarities = [1]
        co_rated = [1]
        cf = UserBasedImplicitCF(matrix, similarities, co_rated)
        self.assertEqual(matrix, cf.matrix)
        self.assertEqual(similarities, cf.similarities())
        self.assertEqual(co_rated, cf.co_rated())
    
    def test_model_initialization(self):
        dimension = 10
        matrix = [[ choice([None, 1]) for _i in range(0,dimension)] for _c in range(0,dimension) ]
        cf = UserBasedImplicitCF(matrix)
        print("")
        print(cf.recommend(0))

        
if __name__ == "main":
    unittest.main()