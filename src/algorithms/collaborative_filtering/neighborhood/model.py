from ..model import CollaborativeFiltering
from ....data_structures.symmetric_matrix import SymmetricMatrix

CO_RATED_KEY = "co_rated"

class NeighborhoodUserCF(CollaborativeFiltering):
    def __init__(self, matrix, co_rated):
        super().__init__(matrix)
        self._init_model(co_rated, CO_RATED_KEY, self._init_co_rated)
    
    def _init_model(self, model, model_name, callback):
        if len(model) == 0:
            callback()
        else:
            self.model[model_name] = model
    
    #initializing the co rated items with the item id's
    def _init_co_rated(self):
        self.model[CO_RATED_KEY] = SymmetricMatrix(len(self.matrix), set())
        for index , user in enumerate(self.matrix):
            for another_index in range(0, index + 1):
                another_user = self.matrix[another_index]
                self.model[CO_RATED_KEY][(index,another_index)] = set([ user_tuple[0] for user_tuple, another_user_tuple in zip(enumerate(user), enumerate(another_user)) if (user_tuple[1] is not None and another_user_tuple[1] is not None) ]) 
    
    #updating the co_rated matrix inside the model
    def _update_co_rated(self, user_id, item_id):
        for another_user_id in range(0,len(self.matrix)):
            if self.matrix[another_user_id][item_id] is not None:
                self.model[CO_RATED_KEY][(user_id,another_user_id)].add(item_id)