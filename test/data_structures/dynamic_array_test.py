import unittest
from data_structures import DynamicArray


class DynamicArrayTest(unittest.TestCase):
    def test_empty(self):
        da = DynamicArray()
        self.assertEqual(da._size, 0)
        self.assertEqual(da._data, [])

    def test_matrix(self):
        matrix = [
            DynamicArray([1, None, None]),
            DynamicArray([None, None, None]),
            DynamicArray([None, None, 1])
        ]
        da = DynamicArray(
            matrix, default_value=DynamicArray(default_value=None))

        self.assertEqual(da._size, len(matrix))
        self.assertEqual(da._data, matrix)
        self.assertEqual(type(da._data[0]), DynamicArray)

    def test_new_values(self):
        matrix = [
            DynamicArray([1, None, None]),
            DynamicArray([None, None, None]),
            DynamicArray([None, None, 1])
        ]
        da = DynamicArray(
            matrix, default_value=DynamicArray(default_value=None))
        self.assertEqual(da._size, len(matrix))
        da[4][4] = 1
        self.assertEqual(len(da), 5)
        self.assertEqual(len(da[4]), 5)
        self.assertEqual(da[4][4], 1)
        da.append(DynamicArray([None, None, 1, None]))
        self.assertEqual(len(da[5]), 4)


if __name__ == "__main__":
    unittest.main()
