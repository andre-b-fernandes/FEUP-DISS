class ElementVariances:
    def __init__(self, first_id, second_id, first_variance, second_variance):
        if first_id < second_id:
            first_id, second_id = second_id, first_id
        self._data = dict()
        self._data[first_id] = first_variance
        self._data[second_id] = second_variance