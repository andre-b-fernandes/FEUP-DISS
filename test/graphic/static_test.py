import unittest
from graphic import EvaluationStatic
from stream.file_stream.explicit import FileStreamExplicit
from evaluators.prequential.\
    explicit_feedback import PrequentialEvaluatorExplicit
from algorithms.collaborative_filtering.matrix_factorization.\
    explicit_feedback import MFExplicitSGD


class StaticTest(unittest.TestCase):
    def test_init(self):
        fs = FileStreamExplicit("test/test_dataset/test.data", sep="\t")
        cf = MFExplicitSGD()
        ev = PrequentialEvaluatorExplicit(cf)
        st = EvaluationStatic(fs.stream, ev)
        self.assertEqual(st.stream, fs.stream)
        self.assertEqual(st.x, range(0, len(fs.stream)))
        self.assertEqual(st.evaluator, ev)
        self.assertEqual(st.err_rate, [])
        self.assertEqual(st.elap_nr, [])
        self.assertEqual(st.elap_rec, [])

    def test_evaluate(self):
        fs = FileStreamExplicit("test/test_dataset/test.data", sep="\t")
        cf = MFExplicitSGD()
        ev = PrequentialEvaluatorExplicit(cf)
        st = EvaluationStatic(fs.stream, ev)
        st.evaluate()
        self.assertEqual(len(st.err_rate), len(st.stream))
        self.assertEqual(len(st.elap_rec), len(st.stream))
        self.assertEqual(len(st.elap_nr), len(st.stream))


if __name__ == "__main__":
    unittest.main()
