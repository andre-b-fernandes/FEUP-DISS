from copy import deepcopy
import numpy as np
class SymmetricMatrix:

    def __init__(self, size, value=None):
        if size < 0:
            raise ValueError('size has to be positive')

        self._size = size
        self._default = value
        self._data = np.array([value for i in range((size + 1) * size // 2)])

    def __len__(self):
        return self._size

    def __setitem__(self, position, value):
        if position[0] >= self._size:
            self._add_elements(position)
        index = self._get_index(position)
        self._data[index] = value

    def __getitem__(self, position):
        if position[0] >= self._size:
            self._add_elements(position)
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

    def _add_elements(self, position):
        row, _col = position
        increments = row - self._size + 1
        for increment in range(increments):
            self._size += increment + 1
            self._data = np.concatenate((self._data, [deepcopy(self._default) for _ in range(self._size)]))
