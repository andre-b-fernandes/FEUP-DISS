from stream.file_stream.implicit import FileStreamImplicit
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import ItemLSH
import unittest


class FileStreamTest(unittest.TestCase):

    def test_initialization(self):
        fs = FileStreamImplicit("test/test_dataset/test.data", sep="\t")
        self.assertEqual(len(fs.stream), 10)

    def test_process_stream(self):
        fs = FileStreamImplicit("test/test_dataset/test.data", sep="\t")
        cf = ItemLSH()
        model = fs.process_stream(cf)
        self.assertEqual(len(model.matrix), 306)
        self.assertEqual(type(model), ItemLSH)


if __name__ == "__main__":
    unittest.main()
