class DynamicArray:
    """
    Description
        A class which implements a dynamic data structure
        under the disguise of a list.
    """
    def __init__(self, data=None, default_value=lambda: None):
        """
        Description
            DynamicArray's constructor.

        Arguments
            :param data: The actual metadata to encapsulate.
            :type data: list
            :param default_value: A function which returns a default\
                value when extending the metadata for uninitialized\
                    positions.
            :type default_value: function
        """
        self.default_value = default_value
        self._data = data
        if self._data is None:
            self._data = list()
        self._size = len(self._data)

    def __iter__(self):
        """
        Description.
            Overriding the __iter__ function.
        """
        return iter(self._data)

    def __len__(self):
        """
        Description.
            Overriding the __len__ function.
        """
        return self._size

    def __getitem__(self, position):
        """
        Description.
            Overriding the __getitem__ function.

        Arguments
            :param position: The index to access.
            :type position: slice or int
        """
        if isinstance(position, slice):
            return self._data[position]
        if position >= self._size:
            self.extend(position)
        return self._data[position]

    def __setitem__(self, position, value):
        """
        Description.
            Overriding the __setitem__ function.

        Arguments
            :param position: The index to access.
            :type position: int
        """
        if position >= self._size:
            self.extend(position)
        self._data[position] = value

    def extend(self, position):
        """
        Description
            A function which allocates more memory for the internal
            _data list.

        Arguments
            :param position: The index which was accessed.
            :type position: int
        """
        increments = position - self._size + 1
        ext = [self.default_value() for _ in range(increments)]
        self._data.extend(ext)
        self._size += increments

    def append(self, value):
        """
        Description
            A function which appends a new value into _data.

        Arguments
            :param value: The new value.
            :type value: Any
        """
        self._data.append(value)
        self._size += 1

    def col(self, position):
        """
        Description
            When _data contains other DynamicArray objects, making
            this a matrix, it returns a column list object.

        Arguments
            :param position: The column index.
            :type position: int
        """
        return [elem[position] for elem in self._data]

    def set_col(self, position, value):
        """
        Description
            When _data contains other DynamicArray objects, making
            this a matrix, this function sets the value for a column.

        Arguments
            :param position: The column index.
            :type position: int
            :param value: The column.
            :type value: list
        """
        for i in range(len(value)):
            self._data[i][position] = value[i]
