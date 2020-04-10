from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import SymmetricMatrix
from data_structures import DynamicArray
from utils import knn
from copy import copy


CO_RATED_KEY = "co_rated"
SIMILARITIES_KEY = "similarities"
NEIGHBORS_KEY = "neighbors"


class NeighborhoodUserCF(CollaborativeFiltering):
    def __init__(self, matrix, co_rated, neighbors, n_neighbors):
        super().__init__(matrix)
        self._init_model(co_rated, CO_RATED_KEY, self._init_co_rated)
        self.similarity_default = 0.0  # for initialization
        self.n_neighbors = n_neighbors
        self.users = {u_id for u_id in range(
            len(self.matrix)) if len(self.co_rated_between(u_id, u_id)) > 0}

    def _init_similarities(self):
        self.model[SIMILARITIES_KEY] = SymmetricMatrix(
            len(self.matrix), copy(self.similarity_default))
        for user_id in range(0, len(self.matrix)):
            for another_user_id in range(0, user_id + 1):
                self._init_similarity(user_id, another_user_id)

    # initializing the co rated items with the item id's
    def _init_co_rated(self):
        self.model[CO_RATED_KEY] = SymmetricMatrix(len(self.matrix), set())
        for index, user in enumerate(self.matrix):
            for another_index, another_user in enumerate(
                    self.matrix[0:index + 1]):
                self.model[CO_RATED_KEY][(index, another_index)] = set([
                    user_tuple[0]
                    for user_tuple, another_user_tuple
                    in zip(enumerate(user), enumerate(another_user))
                    if (user_tuple[1] is not None and another_user_tuple[1]
                        is not None)])

    # initialize neighborhood models
    def _init_neighborhood(self):
        self.model[NEIGHBORS_KEY] = DynamicArray(
            [self._neighborhood(
                u_id) for u_id in self.users], default_value=lambda: list())

    # updating the co_rated matrix inside the model
    def _update_co_rated(self, user_id, item_id, comp):
        for another_user_id in range(0, len(self.matrix)):
            if comp(self.matrix[another_user_id][item_id]):
                self.model[CO_RATED_KEY][(user_id, another_user_id)].add(
                    item_id)

    def _neighborhood(self, user_id):
        candidates = self.users.difference({user_id})
        return knn(user_id, candidates, self.n_neighbors,
                   self.similarity_between)

    def similarities(self):
        return self.model[SIMILARITIES_KEY]

    def co_rated(self):
        return self.model[CO_RATED_KEY]

    def co_rated_between(self, user_id, another_user_id):
        return self.model[CO_RATED_KEY][(user_id, another_user_id)]

    def _init_similarity(self, user_id, another_user_id):
        raise NotImplementedError("The method is not implemented!")

    def similarity_between(self, user, another_user):
        raise NotImplementedError("The method is not implemented!")

    def neighbors(self):
        return self.model[NEIGHBORS_KEY]

    def neighborhood_of(self, user_id):
        return self.model[NEIGHBORS_KEY][user_id]
