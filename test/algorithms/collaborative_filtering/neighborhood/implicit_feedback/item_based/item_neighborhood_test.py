import unittest
from algorithms.collaborative_filtering.\
    neighborhood.implicit_feedback.item_based import ItemBasedNeighborhood


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


if __name__ == "__main__":
    unittest.main()
