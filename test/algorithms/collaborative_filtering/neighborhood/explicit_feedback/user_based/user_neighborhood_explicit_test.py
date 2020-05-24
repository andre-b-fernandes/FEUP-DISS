import unittest
from algorithms.collaborative_filtering.neighborhood.\
    explicit_feedback.user_based import UserBasedNeighborhood


class UserNeighborhoodTest(unittest.TestCase):
    def _test_neighborhood(self, cf, identifier, neighborhood):
        self.assertEqual(cf.neighborhood_of(identifier), neighborhood)
        self.assertNotIn(identifier, neighborhood)

    def _test_neighbors(self, cf, neighbors):
        for user, neighborhood in zip(cf.users, neighbors):
            self._test_neighborhood(cf, user, neighborhood)

    def test_neighborhood(self):
        matrix = [
            [1, None, None, None, 1, None, None, None, 1],
            [1, None, 1, None, 1, None, 1, None, 1],
            [None, None, 1, None, None, None, 1, None, None],
            [None, 1, 1, None, 1, None, 1, None, None],
            [1, None, 1, None, None, None, None, 1, None],
        ]
        cf = UserBasedNeighborhood(matrix, n_neighbors=2)
        self._test_neighbors(cf, [[3, 4], [3, 4], [3, 4], [2, 4], [2, 3]])


if __name__ == "__main__":
    unittest.main()
