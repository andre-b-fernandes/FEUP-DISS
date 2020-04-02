import unittest
from algorithms.collaborative_filtering.neighborhood.\
    explicit_feedback import UserBasedExplicitCF
from evaluators.prequential.explicit_feedback.\
    prequential_evaluator import PrequentialEvaluatorExplicit


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
        err, _elap = evaluator.evaluate(0, 2, 3)
        self.assertEqual(err, 0.4)

    def test_new_rating(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        cf = UserBasedExplicitCF(matrix)
        evaluator = PrequentialEvaluatorExplicit(cf)
        err, _elap, _elap2 = evaluator.new_rating((0, 2, 3))
        self.assertEqual(err, 0.4)


if __name__ == "main":
    unittest.main()
