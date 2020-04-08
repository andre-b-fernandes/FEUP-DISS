class DynamicArray:
    def __init__(self, data=None, default_value=lambda: None):
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
            self.extend(position)
        return self._data[position]

    def __setitem__(self, position, value):
        if position >= self._size:
            self.extend(position)
        self._data[position] = value

    def extend(self, position):
        increments = position - self._size + 1
        ext = [self.default_value() for _ in range(increments)]
        self._data.extend(ext)
        self._size += increments

    def append(self, value):
        self._data.append(value)
        self._size += 1

    def col(self, position):
        return [elem[position] for elem in self._data]

    def set_col(self, position, value):
        for i in range(len(value)):
            self._data[i][position] = value[i]
