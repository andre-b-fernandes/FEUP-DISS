from src.algorithms.collaborative_filtering.collaborative_filtering import CollaborativeFiltering
from src.data_structures.symmetric_matrix import SymmetricMatrix
from src.data_structures.dynamic_array import DynamicArray
from src.utils.utils import knn
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
            for another_index in range(0, index + 1):
                another_user = self.matrix[another_index]
                self.model[CO_RATED_KEY][(index, another_index)] = set([
                    user_tuple[0]
                    for user_tuple, another_user_tuple
                    in zip(enumerate(user), enumerate(another_user))
                    if (user_tuple[1] is not None and another_user_tuple[1]
                        is not None)])

    # initialize neighborhood models
    def _init_neighborhood(self):
        self.model[NEIGHBORS_KEY] = DynamicArray(
            [self._neighborhood(user_id) for user_id in range(
                0, len(self.matrix))], default_value=list())

    # updating the co_rated matrix inside the model
    def _update_co_rated(self, user_id, item_id):
        for another_user_id in range(0, len(self.matrix)):
            if self.matrix[another_user_id][item_id] is not None:
                self.model[CO_RATED_KEY][(user_id, another_user_id)].add(
                    item_id)

    def _neighborhood(self, user_id):
        candidates = list(range(0, len(self.matrix)))
        candidates.remove(user_id)
        return knn(user_id, candidates, self.n_neighbors,
                   self.similarity_between)

    def recommend(self, user_id, n_products):
        item_ids = [i for i in range(0, len(self.matrix[user_id]))
                    if self.matrix[user_id][i] is None]
        return sorted(item_ids,
                      key=lambda item_id:
                      self._activation_weight(user_id, item_id))[-n_products:]

    def _activation_weight(self, user_id, item_id):
        nbs = self.neighborhood_of(user_id)
        return sum([self.similarity_between(user_id, another_user_id)
                    for another_user_id in nbs
                    if self.matrix[another_user_id][item_id]
                    is not None]) / len(nbs)

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
