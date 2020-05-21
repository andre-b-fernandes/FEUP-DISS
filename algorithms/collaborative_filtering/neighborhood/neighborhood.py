from data_structures import DynamicArray
from utils import knn


class NeighborhoodCF:
    # initialize neighborhood models
    def _init_neighborhood(self, candidate_set):
        neighbors = DynamicArray(
            [self._neighborhood(
                ide) for ide in candidate_set], default_value=lambda: list())
        return neighbors

    def _neighborhood(self, ident):
        candidates = self.users.difference({ident})
        return knn(ident, candidates, self.n_neighbors,
                   self.similarity_between)

    def neighborhood_of(self, identifier):
        return self.neighbors[identifier]

    def similarity_between(self, elem, another_elem):
        return self.similarities[(elem, another_elem)]
