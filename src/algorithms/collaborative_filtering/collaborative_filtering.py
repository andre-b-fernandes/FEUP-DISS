from abc import ABC
from src.data_structures.dynamic_array import DynamicArray


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
        self.matrix = DynamicArray(matrix, default_value=DynamicArray())
        self.n_users = len(self.matrix)
        self.n_items = len(self.matrix[0]) if self.n_users > 0 else 0
        self.model = dict()

    def _init_model(self, model, model_name, callback):
        if len(model) == 0:
            callback()
        else:
            self.model[model_name] = model
