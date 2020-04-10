import unittest
from graphic import EvaluationAnimation
from evaluators.prequential.\
    implicit_feedback import PrequentialEvaluatorImplicit
from algorithms.collaborative_filtering.matrix_factorization.\
    implicit_feedback import MatrixFactorizationImplicit


class EvaluationAnimationTest(unittest.TestCase):
    def test_initialization(self):
        cf = MatrixFactorizationImplicit()
        ev = PrequentialEvaluatorImplicit(cf)
        stream = [(0, 0), (1, 1), (2, 2)]
        anim = EvaluationAnimation(stream, ev)
        self.assertEqual(anim.evaluator, ev)
        self.assertEqual(anim.stream, stream)


if __name__ == "__main__":
    unittest.main()
