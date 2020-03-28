import unittest
from src.algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import UserBasedImplicitCF
from src.evaluators.prequential.implicit_feedback.\
    prequential_evaluator import PrequentialEvaluatorImplicit


class PrequentialEvaluatorImplicitTest(unittest.TestCase):
    def test_evaluate(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        lsh_based = UserBasedImplicitCF(matrix)
        evaluator = PrequentialEvaluatorImplicit(lsh_based)
        self.assertTrue(evaluator.evaluate(0, 2))
        self.assertFalse(evaluator.evaluate(0, 0))

    def test_new_rating(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        lsh_based = UserBasedImplicitCF(matrix)
        evaluator = PrequentialEvaluatorImplicit(lsh_based)
        self.assertEqual(evaluator.new_rating((0, 2)), 0.0)


if __name__ == "main":
    unittest.main()
