import numpy as np
from math import sqrt
from ...model import CollaborativeFiltering
from .....utils.utils import covariance, variance, pearson_correlation_terms, pearson_correlation
from .....data_structures.symmetric_matrix import SymmetricMatrix

AVG_RATINGS_KEY = "avg_ratings"
CO_RATED_KEY = "co_rated"
SIMILARITIES_KEY = "similarities"

COVARIANCE_KEY = "B"
VARIANCE_FIRST_KEY = "C"
VARIANCE_SECOND_KEY = "D"
SIM_VALUE_KEY = "similarity_value"

#TODO USE THE SIMETRIC MATRIX AND CHANGE THE TESTS
class UserBasedCollaborativeFiltering(CollaborativeFiltering):
    """
        The definition of the user based collaborative filtering algorithm.
        It extends the collaborative filtering base class.
        This class is to be used for explicit feedback.

        Attributes
        ----------

        matrix : array
            The ratings matrix.
        model: array.
            The user-user similarities.

        
    """
    def __init__(self, matrix = [], similarities = [], avg_ratings = dict(), co_rated = []):
        """
            Constructor

            Parameters
            ----------
            matrix : array
                The ratings matrix.
            similarities: array.
                A user-user matrix containing similarities.
            avg_ratings:
                A dictionary indexed by user_id with average ratings
        """
        super().__init__(matrix)
        self._init_model(avg_ratings, AVG_RATINGS_KEY, self._init_avg_ratings)
        self._init_model(co_rated, CO_RATED_KEY, self._init_co_rated)
        self._init_model(similarities, SIMILARITIES_KEY, self._init_similarities)        
    
    def _init_model(self, model, model_name, callback):
        if len(model) == 0:
            callback()
        else:
            self.model[model_name] = model
    
    #initializing the co rated items with the item id's
    def _init_co_rated(self):
        self.model[CO_RATED_KEY] = SymmetricMatrix(len(self.matrix), [])
        for index , user in enumerate(self.matrix):
            for another_index in range(0, index + 1):
                another_user = self.matrix[another_index]
                self.model[CO_RATED_KEY][(index,another_index)] = [ user_tuple[0] for user_tuple, another_user_tuple in zip(enumerate(user), enumerate(another_user)) if (user_tuple[1] is not None and another_user_tuple[1] is not None) ]            
            
                
    
    def _update_co_rated(self, user_id, item_id):
        for another_user_id in range(0,len(self.matrix)):
            if self.matrix[another_user_id][item_id] is not None:
                self.model[CO_RATED_KEY][(user_id,another_user_id)].append(item_id)
                
    #initializing the average ratings
    def _init_avg_ratings(self):
        self.model[AVG_RATINGS_KEY] = dict()
        for index, user in enumerate(self.matrix):
            self.model[AVG_RATINGS_KEY][index] = 0 if len(list(filter(None, user))) == 0 else sum(filter(None,user))/len(list(filter(None,user)))
    
    # Assuming user in rows and items in columns
    def _init_similarities(self):
        self.model[SIMILARITIES_KEY] = SymmetricMatrix(len(self.matrix), dict())
        for index , user in enumerate(self.matrix):
            for another_index in range(0, index + 1):
                another_user = self.matrix[another_index]
                sim = pearson_correlation_terms(self.model[CO_RATED_KEY][(index,another_index)], user, another_user, self.model[AVG_RATINGS_KEY][index], self.model[AVG_RATINGS_KEY][another_index])                
                self._update_similarities(index, another_index, sim[0], sim[1], sim[2], (round(sim[3], 5)))
    
    #(user_id, item_id, new_rating, new_avg_rating)
    def _unpack_values(self, rating, another_user_id):
        user_id = rating[0]
        current_item_id = rating[1]
        new_rating = rating[2]
        new_avg_rating = rating[3]
        another_user_avg_rating = self.model[AVG_RATINGS_KEY][another_user_id]
        another_user_ratings = self.matrix[another_user_id]
        co_rated = self.model[CO_RATED_KEY][(user_id,another_user_id)]
        old_avg_rating = self.model[AVG_RATINGS_KEY][user_id]
        difference_avg = new_avg_rating - old_avg_rating
        co_rated_length = len(co_rated)
        another_user_rating = another_user_ratings[current_item_id]

        return user_id, current_item_id, new_rating, new_avg_rating, another_user_avg_rating, another_user_ratings, co_rated, old_avg_rating, difference_avg, co_rated_length, another_user_rating

    def _update_similarities(self, user_id, another_user_id, cov, variance_first, variance_second, result):
        self.model[SIMILARITIES_KEY][(user_id,another_user_id)] = dict()
        self.model[SIMILARITIES_KEY][(user_id,another_user_id)][COVARIANCE_KEY] = cov
        self.model[SIMILARITIES_KEY][(user_id,another_user_id)][VARIANCE_FIRST_KEY] = variance_first
        self.model[SIMILARITIES_KEY][(user_id,another_user_id)][VARIANCE_SECOND_KEY] = variance_second
        self.model[SIMILARITIES_KEY][(user_id,another_user_id)][SIM_VALUE_KEY] = result
    
    def _update_similarities_with_terms(self, user_id, another_user_id, e, f, g):
        cov = self.model[SIMILARITIES_KEY][(user_id,another_user_id)][COVARIANCE_KEY] + e
        variance_first = self.model[SIMILARITIES_KEY][(user_id,another_user_id)][VARIANCE_FIRST_KEY] + f
        variance_second = self.model[SIMILARITIES_KEY][(user_id,another_user_id)][VARIANCE_SECOND_KEY] + g
        corr = pearson_correlation(cov, variance_first, variance_second)
        self._update_similarities(user_id, another_user_id, cov, variance_first, variance_second, (round(corr,5)))

    # arguments as ((user_id, item_id, new_rating, new_avg_rating ), another_user_id)
    def _new_rating_terms(self, rating, another_user_id):
        user_id, _current_item_id, new_rating, new_avg_rating, another_user_avg_rating, another_user_ratings, co_rated, old_avg_rating, difference_avg, co_rated_length, another_user_rating = self._unpack_values(rating, another_user_id)
        #had rated
        e,f,g = 0,0,0
        if another_user_rating is not None:
            e = ( new_rating - new_avg_rating) * (another_user_rating - another_user_avg_rating) - sum([ difference_avg * (another_user_ratings[item_id] - another_user_avg_rating) for item_id in co_rated])
            f = ( new_rating - new_avg_rating)**2 + co_rated_length*(difference_avg**2) - 2*sum([ difference_avg*( self.matrix[user_id][item_id] - old_avg_rating) for item_id in co_rated])
            g = ( another_user_rating - another_user_avg_rating )**2
        #had not rated
        else:
            e = - sum([ (difference_avg) * ( another_user_ratings[item_id] - another_user_avg_rating ) for item_id in co_rated])
            f = co_rated_length * (difference_avg)**2 - 2*sum([difference_avg * (self.matrix[user_id][item_id] - old_avg_rating) for item_id in co_rated])

        return e,f,g
    
    def _new_rating_avg(self, user_id, rating):
        old_avg_rating = self.model[AVG_RATINGS_KEY][user_id]
        q = len(self.model[CO_RATED_KEY][(user_id,user_id)])
        new_avg_rating = (rating/(q + 1))+(old_avg_rating * q / (q + 1))
        return new_avg_rating

    #new rating incoming as (user_id, item_id, rating) 
    def _new_rating(self, user_id, item_id, rating):
        new_avg_rating = self._new_rating_avg(user_id, rating)

        members = list(range(0,len(self.matrix)))
        members.remove(user_id)
        for another_user_id in members:
            e,f,g = self._new_rating_terms((user_id, item_id, rating, new_avg_rating), another_user_id)
            self._update_similarities_with_terms(user_id, another_user_id, e, g, f)
        
        self.model[AVG_RATINGS_KEY][user_id] = new_avg_rating
    
    def _update_avg_rating(self, user_id, item_id, rating):
        old_user_rating = self.matrix[user_id][item_id]
        diff_ratings = rating - old_user_rating
        old_avg_rating = self.model[AVG_RATINGS_KEY][user_id]
        q = len(self.model[CO_RATED_KEY][user_id][user_id])
        new_avg_rating = (diff_ratings/q) + old_avg_rating
        return new_avg_rating

    #arguments as ((user_id, item_id, new_rating, new_avg_rating ), another_user_id)
    def _update_rating_terms(self, rating, another_user_id):
        user_id, current_item_id, new_rating, new_avg_rating, another_user_avg_rating, another_user_ratings, co_rated, old_avg_rating, difference_avg, co_rated_length, another_user_rating = self._unpack_values(rating, another_user_id)
        old_user_rating = self.matrix[user_id][current_item_id]
        diff_ratings = new_rating - old_user_rating        
        #had rated
        e,f,g = 0,0,0
        if another_user_rating is not None:
            e = diff_ratings * (another_user_rating - another_user_avg_rating) - sum([ difference_avg * ( another_user_ratings[item_id] - another_user_avg_rating) for item_id in co_rated])
            f = diff_ratings**2 + 2*diff_ratings*(new_rating - new_avg_rating) + co_rated_length*difference_avg**2 - 2*sum([ difference_avg * ( self.matrix[user_id][item_id] - old_avg_rating) for item_id in co_rated])
        #hadn't rated
        else:
            e = -sum([ difference_avg * ( another_user_ratings[item_id] - another_user_avg_rating) for item_id in co_rated])
            f = co_rated_length*difference_avg**2 - 2*sum([ difference_avg * ( self.matrix[user_id][item_id] - old_avg_rating ) for item_id in co_rated ])
        
        return e,f,g

    #update rating incoming as (user_id, item_id, rating)   
    def _update_rating(self, user_id, item_id, rating):
        new_avg_rating = self._update_avg_rating(user_id, item_id, rating)
        
        members = list(range(0,len(self.matrix)))
        members.remove(user_id)
        for another_user_id in members:
            e,f,g = self._update_rating_terms((user_id, item_id, rating, new_avg_rating), another_user_id)
            self._update_similarities_with_terms(user_id, another_user_id, e, g, f)
        
        self.model[AVG_RATINGS_KEY][user_id] = new_avg_rating

    #new stream incoming as (user_id, item_id, rating)
    def new_stream(self, user_id, item_id, rating):
        #rating update
        if self.matrix[user_id][item_id] is not None:
            self._update_rating(user_id, item_id, rating)
            self.matrix[user_id][item_id] = rating
            self._update_co_rated(user_id, item_id)            
        #new rating
        else:
            self._new_rating(user_id, item_id, rating)
            self.matrix[user_id][item_id] = rating
            self._update_co_rated(user_id, item_id)
            
    
    def similarities(self):
        return self.model[SIMILARITIES_KEY]

    def similarity_between(self, user, another_user):
        return self.model[SIMILARITIES_KEY][(user,another_user)][SIM_VALUE_KEY]
    
    def covariance_between(self, user, another_user):
        return self.model[SIMILARITIES_KEY][(user,another_user)][COVARIANCE_KEY]
    
    def variance(self, user, another_user):
        if user > another_user:
            return self.model[SIMILARITIES_KEY][(user,another_user)][VARIANCE_SECOND_KEY]
        else:
            return self.model[SIMILARITIES_KEY][(user,another_user)][VARIANCE_FIRST_KEY]
        
    def similarity_terms_between(self, user, another_user):
        return self.model[SIMILARITIES_KEY][(user,another_user)]
    
    def avg_rating(self, user_id):
        return self.model[AVG_RATINGS_KEY][user_id]
    
    def avg_ratings(self):
        return self.model[AVG_RATINGS_KEY]

    def co_rated(self):
        return self.model[CO_RATED_KEY]
    
    def co_rated_between(self, user_id, another_user_id):
        return self.model[CO_RATED_KEY][(user_id,another_user_id)]