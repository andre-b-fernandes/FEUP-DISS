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
        self.matrix = DynamicArray(default_value=DynamicArray(
            default_value=None))
        for row in matrix:
            self.matrix.append(DynamicArray(row, default_value=None))
        self.model = dict()

    def _init_model(self, model, model_name, callback):
        if len(model) == 0:
            callback()
        else:
            self.model[model_name] = model
