from algorithms.collaborative_filtering import CollaborativeFiltering

U_DECOMPOSED_KEY = "U"
V_DECOMPOSED_KEY = "V"


class MatrixFactorization(CollaborativeFiltering):
    def __init__(self, matrix, u, v, lf):
        super().__init__(matrix)
        self.latent_factors = lf
        self._init_u_v(u, v)

    def _init_u_v(self, u, v):
        self._init_model(u, U_DECOMPOSED_KEY, self._init_u)
        self._init_model(v, V_DECOMPOSED_KEY, self._init_v)

    def _init_u(self):
        raise NotImplementedError("_init_u is not implemented.")

    def _init_v(self):
        raise NotImplementedError("_init_v is not implemented.")

    def u(self):
        return self.model[U_DECOMPOSED_KEY]

    def v(self):
        return self.model[V_DECOMPOSED_KEY]
