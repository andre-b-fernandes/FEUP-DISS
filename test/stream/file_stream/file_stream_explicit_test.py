from stream.file_stream.explicit import FileStreamExplicit
from algorithms.collaborative_filtering.matrix_factorization.\
    explicit_feedback import MatrixFactorizationExplicit
import unittest


class FileStreamTest(unittest.TestCase):

    def test_initialization(self):
        fs = FileStreamExplicit("test/test_dataset/test.data", sep="\t")
        self.assertEqual(len(fs.stream), 10)

    def test_process_stream(self):
        fs = FileStreamExplicit("test/test_dataset/test.data", sep="\t")
        cf = MatrixFactorizationExplicit()
        model = fs.process_stream(cf)
        self.assertEqual(len(model.matrix), 306)
        self.assertEqual(type(model), MatrixFactorizationExplicit)


if __name__ == "__main__":
    unittest.main()
