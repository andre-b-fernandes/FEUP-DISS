import unittest
from algorithms.collaborative_filtering.\
    neighborhood.implicit_feedback.item_based import ItemBasedNeighborhood
from stream.file_stream.implicit import FileStreamImplicit


class ItemNeighborhoodTest(unittest.TestCase):
    def _test_neighborhood(self, cf, identifier, neighborhood):
        self.assertEqual(cf.neighborhood_of(identifier), neighborhood)
        self.assertNotIn(identifier, neighborhood)

    def _test_neighbors(self, cf, neighbors):
        for item, neighborhood in zip(cf.items, neighbors):
            self._test_neighborhood(cf, item, neighborhood)

    def test_neighborhood(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        cf = ItemBasedNeighborhood(matrix, n_neighbors=2)
        self._test_neighbors(cf, [[2, 4], [2, 4], [0, 4], [2, 4], [2, 0]])

    def test_parallel_process(self):
        cf = ItemBasedNeighborhood()
        fs = FileStreamImplicit("test/test_dataset/test.data", sep="\t")
        cf.parallel_process_stream(fs.stream, n_cores=2)
        self.assertEqual(len(cf.matrix), 306)
        self.assertEqual(len(cf.matrix[196]), 475)


if __name__ == "__main__":
    unittest.main()
