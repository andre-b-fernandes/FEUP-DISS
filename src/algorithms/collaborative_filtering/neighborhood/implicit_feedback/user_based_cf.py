from src.algorithms.collaborative_filtering.neighborhood import (
    NeighborhoodUserCF,
    SIMILARITIES_KEY,
    NEIGHBORS_KEY
)
from src.utils import cosine_similarity as cos_sim


class UserBasedImplicitCF(NeighborhoodUserCF):
    def __init__(self, matrix=[], similarities=[], co_rated=[],
                 neighbors=[], n_neighbors=5):
        super().__init__(matrix, co_rated, neighbors, n_neighbors)
        self._init_model(similarities, SIMILARITIES_KEY,
                         self._init_similarities)
        self._init_model(neighbors, NEIGHBORS_KEY, self._init_neighborhood)

    def _init_similarity(self, user_id, another_user_id):
        number_rated_items_user = len(self.co_rated_between(user_id, user_id))
        number_rated_items_another_user = len(self.co_rated_between(
            another_user_id, another_user_id))
        number_of_co_rated_items = len(self.co_rated_between(user_id,
                                                             another_user_id))
        self.model[SIMILARITIES_KEY][(user_id, another_user_id)] = cos_sim(
            number_of_co_rated_items, number_rated_items_user,
            number_rated_items_another_user)

    def _update_similarities(self, user_id):
        members = list(range(0, len(self.matrix)))
        members.remove(user_id)
        for another_user_id in members:
            self._init_similarity(user_id, another_user_id)

    def similarity_between(self, user, another_user):
        return self.model[SIMILARITIES_KEY][(user, another_user)]

    def new_rating(self, rating):
        user_id, item_id = rating[0], rating[1]
        self.matrix[user_id][item_id] = 1
        self._update_co_rated(user_id, item_id)
        self._update_similarities(user_id)
        self._init_neighborhood()

    def recommend(self, user_id, n_products):
        item_ids = [i for i in range(0, len(self.matrix[user_id]))
                    if self.matrix[user_id][i] is None]
        return sorted(item_ids,
                      key=lambda item_id:
                      self._activation_weight(user_id, item_id))[-n_products:]

    def _activation_weight(self, user_id, item_id):
        nbs = self.neighborhood_of(user_id)
        len_nbs = len(nbs)
        return sum([self.similarity_between(user_id, another_user_id)
                    for another_user_id in nbs
                    if self.matrix[another_user_id][item_id]
                    is not None]) / len_nbs
