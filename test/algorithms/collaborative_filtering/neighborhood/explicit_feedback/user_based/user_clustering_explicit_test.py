import unittest
from algorithms.collaborative_filtering.\
    neighborhood.explicit_feedback.user_based import UserBasedClustering


class UserBasedClusteringTest(unittest.TestCase):
    def test_clustering(self):
        matrix = [
            [8, None, None, None, 7, None, None, None, 3],
            [7, None, 1, None, 6, None, 9, None, 4],
            [None, None, 2, None, None, None, 9, None, None],
            [None, 2, 9, None, 1, None, 5, None, None],
            [7, None, 2, None, None, None, None, 8, None],
        ]
        cf = UserBasedClustering(matrix, n_neighbors=2, centroids=[1, 3])
        self.assertEqual(len(cf.centroids), len(cf.clusters))
        self.assertEqual(cf.clusters[0], {0, 1, 2, 4})
        self.assertEqual(cf.clusters[1], {3})
        self.assertEqual(cf.cluster_map[0], 0)
        self.assertEqual(cf.cluster_map[1], 0)
        self.assertEqual(cf.cluster_map[2], 0)
        self.assertEqual(cf.cluster_map[4], 0)
        self.assertEqual(cf.cluster_map[3], 1)

    def test_increment(self):
        matrix = [
            [8, None, None, None, 7, None, None, None, 3],
            [7, None, 1, None, 6, None, 9, None, 4],
            [None, None, 2, None, None, None, 9, None, None],
            [None, 2, 9, None, 1, None, 5, None, None],
            [7, None, 2, None, None, None, None, 8, None],
        ]
        cf = UserBasedClustering(matrix, n_neighbors=2, centroids=[1, 3])
        cf.new_rating((5, 1, 2))
        self.assertEqual(3, len(cf.centroids))
        self.assertEqual(cf.cluster_map[5], 2)
        cf.new_rating((5, 2, 9))
        self.assertEqual(cf.cluster_map[5], 1)
        self.assertEqual(cf.clusters[1], {3, 5})


if __name__ == "__main__":
    unittest.main()
