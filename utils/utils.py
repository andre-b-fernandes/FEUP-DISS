from math import sqrt


def cosine_similarity(n_co_elements, n_first_element, n_second_element):
    """
    Description
        A function which returns the cosine similarity between two elements.

    Arguments
        :param n_co_elements: Number of co-elements.
        :type n_co_elements: int
        :param n_first_element: Size of the first element.
        :type n_first_element: int
        :param n_second_element: Size of the second element
        :type n_second_element: int
    """
    try:
        return n_co_elements / (sqrt(n_first_element) * sqrt(n_second_element))
    except ZeroDivisionError:
        return 0


def covariance(co_elements, first_set, second_set,
               first_set_avg, second_set_avg):
    """
    Description
        A function which returns the covariance between two elements.

    Arguments
        :param co_elements: Number of co-elements.
        :type co_elements: int
        :param first_set: The first element.
        :type first_set: list
        :param second_set: The second element.
        :type second_set: list
        :param first_set_avg: Average of first set.
        :type first_set_avg: float
        :param second_set_avg: Average of second set.
        :type second_set_avg: float.
    """
    return sum([(first_set[identifier] - first_set_avg) * (
        second_set[identifier] - second_set_avg)
        for identifier in co_elements])


def variance(co_elements, elements, avg_elements):
    """
    Description
        A function which returns the variance of an element.

    Arguments
        :param co_elements: Number of co-elements.
        :type co_elements: int
        :param elements: The element.
        :type elements: list
        :param avg_elements: Element's average.
        :type avg_elements: list
    """
    return (sum([(elements[identifier] - avg_elements)**2
                 for identifier in co_elements]))


def pearson_correlation_terms(co_elements, first_set, second_set,
                              first_set_avg, second_set_avg):
    """
    Description
        A function which returns the pearson correlation terms between
        two elements.

    Arguments
        :param co_elements: Number of co-elements.
        :type co_elements: int
        :param first_set: The first element.
        :type first_set: list
        :param second_set: The second element.
        :type second_set: list
        :param first_set_avg: Average of first element.
        :type first_set_avg: float
        :param second_set_avg: Average of second element.
        :type second_set_avg: float.
    """
    cov = covariance(co_elements, first_set, second_set,
                     first_set_avg, second_set_avg)
    variance_first = variance(co_elements, first_set, first_set_avg)
    variance_second = variance(co_elements, second_set, second_set_avg)
    pearson_corr = pearson_correlation(cov, variance_first, variance_second)
    return (cov, variance_first, variance_second, pearson_corr)


def pearson_correlation(covariance, variance_first, variance_second):
    """
    Description
        A function which returns the pearson correlation between
        two elements.

    Arguments
        :param covariance: The covariance between the two elements.
        :type covariance: float
        :param variance_first: The first element's variance.
        :type variance_first: float
        :param variance_second: The second element's variance.
        :type variance_second: float
    """
    try:
        return covariance / (sqrt(variance_first) * sqrt(variance_second))
    except ZeroDivisionError:
        return 0


# returns the ids
def knn(element_id, candidates, n, heuristic):
    """
    Description
        A function which returns the k(n)-nearest-neighbors of the
        element_id according to an heuristic.

    Arguments
        :param element_id: The element for which to calculate neighbors for.
        :type element_id: int
        :param candidates: The candidates to neighbors.
        :type candidates: set
        :param n: Number of neighbors.
        :type n: int
        :param heuristic: Heuristic to sort the candidates by.
        :type heuristic: function
    """
    return sorted(candidates,
                  key=lambda another_element_id: heuristic(
                      element_id, another_element_id))[-n:]


def avg(elements):
    """
    Description
        A function which computes and returns the average of a set.

    Arguments
        :param elements: The element to calculate the average for.
        :type elements: list
    """
    not_none = list(filter(None, elements))
    return 0 if len(not_none) == 0 else sum(not_none) / len(not_none)


def increment_avg(old_avg, new_value, elements):
    """
    Description
        A function which increments and returns the average of a set.

    Arguments
        :param old_avg: The currrent average.
        :type old_avg: float
        :param new_value: The new value which changes the average.
        :type new_value: float
        :param elements: The element for which we want to increment the \
            average.
        :type elements: list
    """
    not_none = list(filter(None, elements))
    return old_avg + (new_value - old_avg) / len(not_none)
