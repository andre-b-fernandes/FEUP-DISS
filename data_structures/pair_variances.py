class PairVariances:
    """
    Description
        A class which implements a useful data structure
        which stores variance pairs for the user-based neighboorhod
        explicit algorithm.
    """
    def __init__(self):
        """
        Description
            PairVariances's constructor.
        """
        self.first_variance = 0
        self.second_variance = 0

    def set_variance(self, first, second, f_var, s_var):
        """
        Description
            A function which sets the variances of the pair of elements.

        Arguments
            :param first: The first element.
            :type first: int
            :param second: The second element.
            :type second: int
            :param f_var: The first element's variance.
            :type f_var: int
            :param s_var: The second element's variance.
            :type s_var: int
        """
        if first < second:
            f_var, s_var = s_var, f_var
        self.first_variance, self.second_variance = f_var, s_var

    def variance(self, first, second):
        """
        Description
            A function which returns the variance given a specific
            pair.

        Arguments
         :param first: The first element.
         :type first: int
         :param second: The second element.
         :type second: int
        """
        if first > second:
            return self.first_variance
        else:
            return self.second_variance
