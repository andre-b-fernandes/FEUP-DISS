import unittest
from algorithms.collaborative_filtering.\
    neighborhood.implicit_feedback.user_based import UserBasedClustering


class UserBasedClusteringTest(unittest.TestCase):
    def test_clustering(self):
        matrix = [
            [1, None, None, None, 1, None, None, None, 1],
            [1, None, 1, None, 1, None, 1, None, 1],
            [None, None, 1, None, None, None, 1, None, None],
            [None, 1, 1, None, 1, None, 1, None, None],
            [1, None, 1, None, None, None, None, 1, None],
        ]
        cf = UserBasedClustering(matrix, n_neighbors=2, centroids=[1, 3])
        self.assertEqual(len(cf.centroids), len(cf.clusters))
        self.assertEqual(cf.clusters[0], {0, 1, 4})
        self.assertEqual(cf.clusters[1], {2, 3})
        self.assertEqual(cf.cluster_map[0], 0)
        self.assertEqual(cf.cluster_map[1], 0)
        self.assertEqual(cf.cluster_map[2], 1)
        self.assertEqual(cf.cluster_map[4], 0)
        self.assertEqual(cf.cluster_map[3], 1)

    def test_increment(self):
        matrix = [
            [1, None, None, None, 1, None, None, None, 1],
            [1, None, 1, None, 1, None, 1, None, 1],
            [None, None, 1, None, None, None, 1, None, None],
            [None, 1, 1, None, 1, None, 1, None, None],
            [1, None, 1, None, None, None, None, 1, None],
        ]
        cf = UserBasedClustering(matrix, n_neighbors=2, centroids=[1, 3])
        cf.new_rating((5, 1))
        self.assertEqual(2, len(cf.centroids))
        self.assertEqual(cf.cluster_map[5], 1)


if __name__ == "__main__":
    unittest.main()
