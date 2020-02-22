import numpy as np
from math import sqrt
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
        self.model = np.zeros((len(self.matrix), len(self.matrix)))
        for index , user in enumerate(self.matrix):
            copy_matrix = np.copy(self.matrix)
            np.delete(copy_matrix, index, 0)
            for another_index, another_user in enumerate(copy_matrix):
                sim = self.calculate_pearson_similarity(user, another_user)
                self.model[index][another_index] = sim
                

    def _validate_model(self):
        self._validate_matrix(self.model)
    
    #Uses the Pearson Similarity Coefficient to calculate the similarities between both users.
    def calculate_pearson_similarity(self, user, another_user):
        #id's of co_rated items
        co_ratings = [ user_tuple[0] for user_tuple, another_user_tuple in zip(enumerate(user), enumerate(another_user)) if (user_tuple[1] != None and another_user_tuple[1] != None) ]
        user_avg = sum(user)/len(user)
        another_user_avg = sum(another_user)/len(another_user)

        covariance = sum([ (user[item_id] - user_avg)*(another_user[item_id] - another_user_avg) for item_id in co_ratings])
        std_dev_user = sqrt(sum([ (user[item_id] - user_avg)**2 for item_id in co_ratings]))
        std_dev_another_user = sqrt(sum([ (another_user[item_id] - another_user_avg)**2 for item_id in co_ratings]))
        
        return (covariance/(std_dev_user * std_dev_another_user))
    
