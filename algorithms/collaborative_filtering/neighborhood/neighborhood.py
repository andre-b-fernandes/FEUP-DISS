from data_structures import DynamicArray
from utils import knn


class NeighborhoodCF:
    """
    Description
        A class which aims at calculating neighborhood
        models using standard knn.
    """
    def __init__(self, neighbors, n_neighbors):
        """
        Description
            NeighborhoodCF's constructor.

        Arguments
            :param neighbors: The neighborhood model.
            :type neighbors: list
            :param n_neighbors: Number of neighbors to compute.
            :type n_neighbors: int
        """
        self.n_neighbors = n_neighbors
        self.neighbors = self._init_model(
            neighbors, self._init_neighborhood)

    # initialize neighborhood models
    def _init_neighborhood(self, candidate_set):
        """
        Description
            A function which computes and returns
            a neighborhood for a candidate set, which is a DynamicArray object.

        Arguments
            :param candidate_set: A set of elements to candidate neighbors for
            :type candidate_set: set
        """
        neighbors = DynamicArray(
            [self._neighborhood(
                ide) for ide in candidate_set], default_value=lambda: list())
        return neighbors

    def _neighborhood(self, ident):
        """
        Description
            A function which computes and returns the neighborhood
            of an element.

        Argument
            :param ident: The element to calculate the neighborhood for.
            :type ident: int
        """
        candidates = self.users.difference({ident})
        return knn(ident, candidates, self.n_neighbors,
                   self.similarity_between)

    def neighborhood_of(self, ident):
        """
        Description
            A function which returns the neighborhood of an
            element.

        Argument
            :param ident: Element of which we want to return the neighborbood.
            :type ident: int
        """
        return self.neighbors[ident]

    def similarity_between(self, elem, another_elem):
        """
        Description
            A function which returns the similarity between two elements.

        Arguments
            :param elem: The first element
            :type elem: int
            :param another_elem: The second element
            :type another_elem: int
        """
        return self.similarities[(elem, another_elem)]
