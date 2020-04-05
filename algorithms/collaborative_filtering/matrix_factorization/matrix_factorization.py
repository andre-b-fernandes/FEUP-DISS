from algorithms.collaborative_filtering import CollaborativeFiltering
from data_structures import DynamicArray

U_DECOMPOSED_KEY = "U"
V_DECOMPOSED_KEY = "V"
P_KEY = "P"


class MatrixFactorization(CollaborativeFiltering):
    def __init__(self, matrix, u, v, p, lf):
        super().__init__(matrix)
        self.latent_factors = lf
        self._init_u_v(u, v)
        self._init_model(p, P_KEY, self._init_p)

    def _initial_training(self):
        for user_id, ratings in enumerate(self.matrix):
            for item_id, rating in enumerate(ratings):
                if rating is not None:
                    self.new_rating((user_id, item_id, rating))

    def _init_u_v(self, u, v):
        self._init_model(u, U_DECOMPOSED_KEY, self._init_u)
        self._init_model(v, V_DECOMPOSED_KEY, self._init_v)

    def _init_u(self):
        raise NotImplementedError("_init_u is not implemented.")

    def _init_v(self):
        raise NotImplementedError("_init_v is not implemented.")

    def _init_p(self):
        self.model[P_KEY] = DynamicArray(
            default_value=lambda: DynamicArray(default_value=lambda: 0))

    def u(self):
        return self.model[U_DECOMPOSED_KEY]

    def v(self):
        return self.model[V_DECOMPOSED_KEY]
