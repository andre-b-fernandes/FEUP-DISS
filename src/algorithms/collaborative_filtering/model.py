class CollaborativeFiltering:
    """
        The definition of a Collaborative Filtering Model.\n

        Attributes
        ----------

        matrix : array
            The ratings matrix
        model: array
            The collaborative filtering model
    """
    def __init__(self, matrix):
        """
            Constructor

            Parameters
            ----------
            matrix: array
                Ratings matrix.
        """
        self.matrix = matrix
        self.model = dict()        