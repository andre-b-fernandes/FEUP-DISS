import unittest
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.lsh_neighborhood import LSHBased
from src.evaluators.prequential.implicit_feedback.prequential_evaluator import PrequentialEvaluatorImplicit


class PrequentialEvaluatorImplicitTest(unittest.TestCase):
    def test_evaluate(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        lsh_based = LSHBased(matrix)
        evaluator = PrequentialEvaluatorImplicit(lsh_based)
        self.assertTrue(evaluator.evaluate(0, 2))
        self.assertFalse(evaluator.evaluate(0, 0))

    def test_new_stream(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        lsh_based = LSHBased(matrix, n_perms=20)
        evaluator = PrequentialEvaluatorImplicit(lsh_based)
        self.assertEqual(evaluator.new_stream((0, 2)), 0.0)


if __name__ == "main":
    unittest.main()
