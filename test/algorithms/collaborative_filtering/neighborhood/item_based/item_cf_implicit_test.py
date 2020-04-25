import unittest
from algorithms.collaborative_filtering.\
    neighborhood.implicit_feedback import ItemBasedImplicitCF


class ItemBasedImplicitCFTest(unittest.TestCase):
    def test_initialization(self):
        pass
        # cf = ItemBasedImplicitCF(n_neighbors=10)
        # self.assertEqual(cf.n_neighbors, 10)
        # self.assertEqual(len(cf.matrix), 0)
        # self.assertEqual(len(cf.intersections_matrix()), 0)
        # self.assertEqual(len(cf.matrix), 0)
        # self.assertEqual(len(cf.matrix), 0)


if __name__ == "__main__":
    unittest.main()
