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
    def __init__(self, matrix, model):
        """
            Constructor

            Parameters
            ----------
            matrix: array
                Ratings matrix.
            model: array
                the collaborative filtering model
        """
        self.matrix = matrix
        self._validate_matrix(matrix)
        self.model = model
    
    def _validate_matrix(self, mat):
        """
            Function to validate if the matrix passed is valid.

            Raises
            ------
            TypeError
                If it is not a valid array of arrays(or similar).
            ValueError
                If it has incorrect dimensions.
        """
        n_rows = len(mat)
        # Assuming user on the rows and items on the columns
        for row in mat:
            if not(type(row) is list):
                raise TypeError("The matrix has incorrect type of values")
            if len(row) != n_rows:
                raise ValueError("All the rows of the matrix must have the same number of columns!")
            
        
        