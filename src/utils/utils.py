from math import sqrt


def cosine_similarity(n_co_elements, n_first_element, n_second_element):
    try:
        return n_co_elements / (sqrt(n_first_element) * sqrt(n_second_element))
    except ZeroDivisionError:
        return 0


def covariance(co_elements, first_set, second_set,
               first_set_avg, second_set_avg):
    return sum([(first_set[identifier] - first_set_avg) * (
        second_set[identifier] - second_set_avg)
        for identifier in co_elements])


def variance(co_elements, elements, avg_elements):
    return (sum([(elements[identifier] - avg_elements)**2
                 for identifier in co_elements]))


def pearson_correlation_terms(co_elements, first_set, second_set,
                              first_set_avg, second_set_avg):
    cov = covariance(co_elements, first_set, second_set,
                     first_set_avg, second_set_avg)
    variance_first = variance(co_elements, first_set, first_set_avg)
    variance_second = variance(co_elements, second_set, second_set_avg)
    pearson_corr = pearson_correlation(cov, variance_first, variance_second)
    return (cov, variance_first, variance_second, pearson_corr)


def pearson_correlation(covariance, variance_first, variance_second):
    try:
        return covariance / (sqrt(variance_first) * sqrt(variance_second))
    except ZeroDivisionError:
        return 0


# returns the ids
def knn(element_id, candidates, n, heuristic):
    return sorted(candidates,
                  key=lambda another_element_id: heuristic(
                      element_id, another_element_id))[-n:]


def avg(elements):
    not_none = list(filter(None, elements))
    return 0 if len(not_none) == 0 else sum(not_none) / len(not_none)
