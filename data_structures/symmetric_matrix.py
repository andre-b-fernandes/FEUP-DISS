from data_structures import DynamicArray


class SymmetricMatrix:
    """
    Description
        A class which implements a matrix object with
        symmetry characteristics.
    """
    def __init__(self, size=0, value=lambda: None):
        """
        Description
            SymmetricMatrix's constructor.

        Arguments
            :param size: The initial size.
            :type size: list
            :param default_value: A function which returns a default\
                value when extending the metadata for uninitialized\
                    positions.
            :type default_value: function
        """
        if size < 0:
            raise ValueError('size cannot be negative')

        self._data = DynamicArray(
            [value() for i in range((size + 1) * size // 2)],
            value)

    def __setitem__(self, position, value):
        """
        Description.
            Overriding the __setitem__ function.

        Arguments
            :param position: The index to access.
            :type position: tuple
            :param value: The value to set.
            :type value: int
        """
        index = self._get_index(position)
        self._data[index] = value

    def __getitem__(self, position):
        """
        Description.
            Overriding the __getitem__ function.

        Arguments
            :param position: The index to access.
            :type position: tuple
        """
        index = self._get_index(position)
        return self._data[index]

    def _get_index(self, position):
        """
        Description
            A function which computes the resulting index
            out of a provided (row, column) tuple.

        Arguments
            :param position: A tuple of the form (row, column).
            :type position: tuple
        """
        row, column = position
        if column > row:
            row, column = column, row
        index = (0 + row) * (row + 1) // 2 + column
        return index

    def __iter__(self):
        """
        Description.
            Overriding the __iter__ function.
        """
        return iter(self._data)
