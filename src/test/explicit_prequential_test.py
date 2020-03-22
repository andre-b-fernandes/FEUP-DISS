import unittest
from src.algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedExplicitCF
from src.evaluators.prequential.explicit_feedback.prequential_evaluator import PrequentialEvaluatorExplicit


class PrequentialEvaluatorExplicitTest(unittest.TestCase):
    def test_evaluate(self):
        matrix = [
            [8, None, None, None, 7],
            [7, None, 1, None, 6],
            [None, 2, 9, None, 1],
            [None, 1, 9, None, None],
            [7, None, 1, None, 6],
        ]
        cf = UserBasedExplicitCF(matrix)
        evaluator = PrequentialEvaluatorExplicit(cf)
        self.assertEqual(evaluator.evaluate(0, 2, 3), 0.2)

    def test_new_stream(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        cf = UserBasedExplicitCF(matrix)
        evaluator = PrequentialEvaluatorExplicit(cf)
        self.assertEqual(evaluator.new_stream((0, 2, 3)), 0.2)


if __name__ == "main":
    unittest.main()
