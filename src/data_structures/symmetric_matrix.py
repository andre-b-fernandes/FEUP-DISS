class SymmetricMatrix:
    def __init__(self, size, value=None):
        if size <= 0:
            raise ValueError('size has to be positive')
 
        self._size = size
        self._data = [value for i in range((size + 1) * size // 2)]
    
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
