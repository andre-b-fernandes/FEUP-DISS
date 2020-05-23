import unittest
from algorithms.collaborative_filtering.\
    neighborhood.explicit_feedback.user_based import UserBasedNeighborhood


class UserNeighborhoodTest(unittest.TestCase):
    def test_neighborhood(self):
        user_id = 2
        matrix = [
            [8, None, None, None, 7, None, None, None, 3],
            [7, None, 1, None, 6, None, 9, None, 4],
            [None, None, 2, None, None, None, 9, None, None],
            [None, 2, 9, None, 1, None, 5, None, None],
            [7, None, 2, None, None, None, None, 8, None],
        ]
        cf = UserBasedNeighborhood(matrix, n_neighbors=2)
        self.assertNotIn(user_id, cf.neighborhood_of(user_id))
        self.assertIn(1, cf.neighborhood_of(user_id))
        self.assertIn(4, cf.neighborhood_of(user_id))
        self.assertIn(2, cf.neighborhood_of(1))
        self.assertIn(4, cf.neighborhood_of(1))

    def test_recommendation(self):
        user_id = 2
        matrix = [
            [8, None, None, None, 7, None, None, None, 3],
            [7, None, 1, None, 6, None, 9, None, 4],
            [None, None, 2, None, None, None, 9, None, None],
            [None, 2, 9, None, 1, None, 5, None, None],
            [7, None, 2, None, None, None, None, 8, None],
        ]
        cf = UserBasedNeighborhood(matrix, n_neighbors=2)
        self.assertIn(0, cf.recommend(user_id, 1))


if __name__ == "__main__":
    unittest.main()
