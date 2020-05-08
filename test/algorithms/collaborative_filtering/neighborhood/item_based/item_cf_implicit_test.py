import unittest
from algorithms.collaborative_filtering.\
    neighborhood.implicit_feedback import ItemBasedImplicitCF
from stream.file_stream.implicit import FileStreamImplicit


class ItemBasedImplicitCFTest(unittest.TestCase):
    def test_asserts(self):
        cf = ItemBasedImplicitCF(n_neighbors=10)
        self.assertEqual(cf.n_neighbors, 10)
        self.assertEqual(len(cf.inv_index), 0)
        self.assertEqual(len(cf.l1_norms), 0)

    def test_initialization(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        cf = ItemBasedImplicitCF(matrix)
        self.assertEqual(cf.l1_norm_of(0), 3)
        self.assertEqual(cf.l1_norm_of(3), 0)
        self.assertEqual(cf.l1_norm_of(4), 4)
        self.assertEqual(cf.inv_index_of(0), {0, 4})
        self.assertEqual(cf.inv_index_of(1), {0, 2, 4})
        self.assertEqual(cf.inv_index_of(2), {2})
        self.assertEqual(cf.intersections_between(0, 4), 3)
        self.assertEqual(cf.intersections_between(1, 3), 0)
        self.assertEqual(cf.intersections_between(0, 2), 2)

    def test_new_rating(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        cf = ItemBasedImplicitCF(matrix)
        self.assertAlmostEqual(cf.similarity_between(0, 0), 1.0, delta=0.0001)
        self.assertAlmostEqual(cf.similarity_between(1, 1), 1.0, delta=0.0001)
        self.assertAlmostEqual(cf.similarity_between(2, 2), 1.0, delta=0.0001)
        self.assertAlmostEqual(cf.similarity_between(3, 3), 0, delta=0.0001)
        self.assertAlmostEqual(cf.similarity_between(0, 3), 0, delta=0.0001)
        cf.new_rating((0, 3))
        self.assertAlmostEqual(cf.similarity_between(0, 3), 0.577, delta=0.001)
        cf.new_rating((1, 3))
        self.assertAlmostEqual(cf.similarity_between(0, 3), 0.816, delta=0.001)

    def test_recommendation(self):
        matrix = [
            [1, None, None, None, 1],
            [1, None, 1, None, 1],
            [None, None, 1, None, None],
            [None, 1, 1, None, 1],
            [1, None, 1, None, 1],
        ]
        cf = ItemBasedImplicitCF(matrix, n_neighbors=2)
        self.assertEqual(cf.neighborhood_of(0), [2, 4])
        self.assertIn(2, cf.recommend(0, 3))
        self.assertNotIn(0, cf.recommend(0, 3))

    def test_parallel_process(self):
        cf = ItemBasedImplicitCF()
        fs = FileStreamImplicit("test/test_dataset/test.data", sep="\t")
        cf.parallel_process_stream(fs.stream, n_cores=2)
        self.assertEqual(len(cf.matrix), 306)
        self.assertEqual(len(cf.matrix[196]), 475)


if __name__ == "__main__":
    unittest.main()
