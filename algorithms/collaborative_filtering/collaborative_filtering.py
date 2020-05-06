from abc import ABC
from data_structures import DynamicArray


class CollaborativeFiltering(ABC):
    """
        The definition of a Collaborative Filtering Model.\n

        Attributes
        ----------

        matrix : array
            The ratings matrix
        model: array
            The collaborative filtering model
    """
    def __init__(self, matrix):
        """
            Constructor

            Parameters
            ----------
            matrix: array
                Ratings matrix.
        """
        self.items = set(
            {item_id for row in matrix for item_id in range(len(row))})
        self.users = set({user_id for user_id in range(len(matrix))})
        self.matrix = DynamicArray(default_value=lambda: DynamicArray())
        for row in matrix:
            self.matrix.append(DynamicArray(row))
        self.model = dict()

    def _init_model(self, model, callback):
        if len(model) == 0:
            return callback()
        else:
            return model
