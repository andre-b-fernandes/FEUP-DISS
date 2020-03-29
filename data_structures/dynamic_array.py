from copy import deepcopy


class DynamicArray:
    def __init__(self, data=None, default_value=None):
        self.default_value = default_value
        self._data = data or list()
        self._size = len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self._size

    def __getitem__(self, position):
        if isinstance(position, slice):
            return self._data[position]
        if position >= self._size:
            self._add_elements(position)
        return self._data[position]

    def __setitem__(self, position, value):
        if position >= self._size:
            self._add_elements(position)
        self._data[position] = value

    def _add_elements(self, position):
        increments = position - self._size + 1
        ext = [deepcopy(self.default_value) for _ in range(increments)]
        self._data.extend(ext)
        self._size += increments

    def append(self, value):
        self._data.append(value)
        self._size += 1
