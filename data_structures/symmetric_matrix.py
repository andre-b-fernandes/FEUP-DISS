from data_structures import DynamicArray


class SymmetricMatrix:
    def __init__(self, size=0, value=lambda: None):
        if size < 0:
            raise ValueError('size cannot be negative')

        self._size = size
        self._data = DynamicArray(
            [value() for i in range((size + 1) * size // 2)],
            value)

    def __len__(self):
        return self._size

    def __setitem__(self, position, value):
        index = self._get_index(position)
        self._data[index] = value

    def __getitem__(self, position):
        index = self._get_index(position)
        return self._data[index]

    def _get_index(self, position):
        row, column = position
        if column > row:
            row, column = column, row
        index = (0 + row) * (row + 1) // 2 + column
        return index

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        ret = ""
        for i in range(0, self._size):
            ret += "[  "
            for c in range(0, i + 1):
                ret += str(self.__getitem__([i, c]))
                ret += "  "
            ret += "]\n"
        return ret
