import numpy as np
from ...model import CollaborativeFiltering

class UserBasedCollaborativeFiltering(CollaborativeFiltering):
    """
        The definition of the user based collaborative filtering algorithm.
        It extends the collaborative filtering base class.
        This class is to be used for explicit feedback.

        Attributes
        ----------

        matrix : array
            The ratings matrix.
        model: array.
            The user-user similarities.

        
    """
    def __init__(self, matrix, model = []):
        """
            Constructor

            Parameters
            ----------
            matrix : array
                The ratings matrix.
            model: array.
                The user-user similarities.
        """
        super(matrix, model)
        # If no model was provided(no previously calculated one), it has to be calculated.
        if len(model) == 0:
            self._init_model()
        else:
            self._validate_model()
    
    # Assuming user in rows and items in columns
    def _init_model(self):
        for index , user in enumerate(self.matrix):
            copy_matrix = np.copy(self.matrix)
            np.delete(copy_matrix, index, 0)
            for another_user in copy_matrix:
                sim = self.calculate_pearson_similarity(user, another_user)
                

    def _validate_model(self):
        self._validate_matrix(self.model)
    
    #Uses the Pearson Similarity Coefficient to calculate the similarities between both users.
    def calculate_pearson_similarity(self, user, another_user):
        co_ratings = [ user_tuple[0] for user_tuple, another_user_tuple in zip(enumerate(user), enumerate(another_user)) if (user_tuple[1] != None and another_user_tuple[1] != None) ]
        return 0
    
