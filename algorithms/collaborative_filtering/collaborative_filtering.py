from abc import ABC
from data_structures import DynamicArray


class CollaborativeFiltering(ABC):
    """
    Description
        The CollaborativeFiltering abstract class. Intended
        for generalizing collaborative filtering algorithms.
    """
    def __init__(self, matrix):
        """
        Description
            CollaborativeFiltering's constructor.

        Arguments
            :param matrix: The ratings matrix.
            :type matrix: list.
        """
        self.items = set(
            {item_id for row in matrix for item_id in range(len(row))})
        self.users = set({user_id for user_id in range(len(matrix))})
        self.matrix = DynamicArray(default_value=lambda: DynamicArray())
        for row in matrix:
            self.matrix.append(DynamicArray(row))

    def _init_model(self, model, callback):
        """
        Description
            A function which returns a collaborative filtering model,
            initializing it if empty.

        Arguments
            :param model: The collaborative filtering model. E.g list of
            average ratings. Has to have __len__ implemented.
            :type model: Any.
            :param callback: The function which returns a computed model.
            :type callback: function.
        """
        if len(model) == 0:
            return callback()
        else:
            return model
