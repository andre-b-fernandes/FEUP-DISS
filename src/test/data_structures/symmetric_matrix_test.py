import unittest
from src.data_structures import SymmetricMatrix


class SymmetricMatrixTest(unittest.TestCase):
    def test_init(self):
        sm = SymmetricMatrix()
        self.assertEqual(len(sm), 0)
        self.assertEqual(len(sm._data), 0)

    def test_symmetry(self):
        dimension = 5
        sy = SymmetricMatrix(dimension)
        self.assertEqual(len(sy._data), dimension**2/2 + dimension/2)
        self.assertEqual(len(sy), dimension)
        sy[(3, 3)] = dimension
        self.assertEqual(sy[(3, 3)], dimension)


if __name__ == "__main__":
    unittest.main()
