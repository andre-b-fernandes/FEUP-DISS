from math import sqrt

def cosine_similarity(first_element, second_element):
    pass

def covariance(co_elements, first_set, second_set, first_set_avg, second_set_avg):
    return sum([ (first_set[identifier] - first_set_avg)*(second_set[identifier] - second_set_avg) for identifier in co_elements])

def standard_deviation(co_elements, elements, avg_elements):
    return sqrt(sum([ (elements[identifier] - avg_elements)**2 for identifier in co_elements]))

def pearson_correlation(co_elements, first_set, second_set , first_set_avg, second_set_avg):
    cov = covariance(co_elements, first_set, second_set, first_set_avg, second_set_avg)
    std_dev_fist = standard_deviation(co_elements, first_set, first_set_avg)
    std_dev_second = standard_deviation(co_elements, second_set, second_set_avg)
    pearson_correlation = 0 if cov == 0 else cov/ (std_dev_fist * std_dev_second)

    return (cov, std_dev_fist, std_dev_second, pearson_correlation)