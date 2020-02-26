import numpy as np
from math import sqrt
from ...model import CollaborativeFiltering

AVG_RATINGS_KEY = "avg_ratings"
CO_RATED_KEY = "co_rated"
SIMILARITIES_KEY = "similarities"

COVARIANCE_KEY = "A"
STD_DEV_FIRST_KEY = "B"
STD_DEV_SEC_KEY = "C"
SIM_VALUE_KEY = "similarity_value"


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
    def __init__(self, matrix, similarities = [], avg_ratings = dict(), co_rated = []):
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
        super(matrix)
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
        self.model[CO_RATED_KEY] = np.zeros((len(self.matrix), len(self.matrix)))
        for index , user in enumerate(self.matrix):
            for another_index, another_user in enumerate(self.matrix):
                self.model[CO_RATED_KEY][index][another_index] = [ user_tuple[0] for user_tuple, another_user_tuple in zip(enumerate(user), enumerate(another_user)) if (user_tuple[1] is not None and another_user_tuple[1] is not None) ]
    
    def _update_co_rated(self, rating):
        for another_index, co_rated_list in enumerate(self.model[CO_RATED_KEY][rating[0]]):
                if self.matrix[another_index][rating[1]] is not None:
                    self.model[CO_RATED_KEY][rating[0]][another_index] = np.append(co_rated_list, rating[1])
                
    #initializing the average ratings
    def _init_avg_ratings(self):
        self.model[AVG_RATINGS_KEY] = dict()
        for index, user in enumerate(self.matrix):
            self.model[AVG_RATINGS_KEY][index] = sum(user)/len(user)
    
    # Assuming user in rows and items in columns
    def _init_similarities(self):
        self.model[SIMILARITIES_KEY] = np.zeros((len(self.matrix), len(self.matrix)))
        for index , user in enumerate(self.matrix):
            for another_index, another_user in enumerate(self.matrix):
                sim = self._calculate_pearson_similarity((index, user), (another_index, another_user))
                self._update_similarities(index, another_index, sim[0], sim[1], sim[2])

    def _update_similarities(self, user_id, another_user_id, cov, std_dev1, std_dev2):
        self.model[SIMILARITIES_KEY][user_id][another_user_id] = dict()
        self.model[SIMILARITIES_KEY][user_id][another_user_id][COVARIANCE_KEY] = cov
        self.model[SIMILARITIES_KEY][user_id][another_user_id][STD_DEV_FIRST_KEY] = std_dev1
        self.model[SIMILARITIES_KEY][user_id][another_user_id][STD_DEV_SEC_KEY] = std_dev2
        self.model[SIMILARITIES_KEY][user_id][another_user_id][SIM_VALUE_KEY] = cov/(std_dev1 * std_dev2)
                
    #Uses the Pearson Similarity Coefficient to calculate the similarities between both users.
    def _calculate_pearson_similarity(self, user, another_user):
        #id's of co_rated items
        co_ratings = self.model[CO_RATED_KEY][user[0]][another_user[0]]
        user_avg = self.model[AVG_RATINGS_KEY][user[0]]
        another_user_avg = self.model[AVG_RATINGS_KEY][another_user[0]]

        covariance = sum([ (user[1][item_id] - user_avg)*(another_user[1][item_id] - another_user_avg) for item_id in co_ratings])
        std_dev_user = sqrt(sum([ (user[1][item_id] - user_avg)**2 for item_id in co_ratings]))
        std_dev_another_user = sqrt(sum([ (another_user[1][item_id] - another_user_avg)**2 for item_id in co_ratings]))
        
        return (covariance, std_dev_user, std_dev_another_user, (covariance/(std_dev_user * std_dev_another_user)))
    
    #new rating incoming as (user_id, item_id, rating)    
    def _new_rating(self, rating):
        old_avg_rating = self.model[AVG_RATINGS_KEY][rating[0]]
        new_size = len(self.model[CO_RATED_KEY][rating[0]][rating[0]])
        new_avg_rating = (rating[2]/new_size)+(old_avg_rating * (new_size - 1) / new_size)
        self.model[AVG_RATINGS_KEY][rating[0]] =  new_avg_rating
        difference_avg = new_avg_rating - old_avg_rating
        e,f,g = 0,0,0
        for another_index, another_user in enumerate(self.matrix):
            #had rated
            if another_user[rating[1]] is not None:
                e = ( rating[2] - new_avg_rating) * (another_user[rating[1]] - self.model[AVG_RATINGS_KEY][another_index]) - sum([ difference_avg * (another_user[rating[1]] - self.model[AVG_RATINGS_KEY][another_index]) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index] ])
                f = (rating[2] - new_avg_rating)**2 + len(self.model[CO_RATED_KEY][rating[0]][another_index])*difference_avg**2 - 2*sum([ difference_avg*( self.matrix[rating[0]][item_id] - old_avg_rating) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index] ])
                g = ( another_user[rating[1]] - self.model[AVG_RATINGS_KEY][another_index])**2
            #had not rated
            else:
                e = sum([ (difference_avg) * ( another_user[item_id] - self.model[AVG_RATINGS_KEY][another_index] ) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index] ])
                g = 0
                f = len(self.model[CO_RATED_KEY][rating[0]][another_index]) * (difference_avg)**2 - 2*sum([difference_avg * (self.matrix[rating[0]][item_id] - new_avg_rating) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index]] )

            self._update_similarities(rating[0], another_index, self.model[SIMILARITIES_KEY][rating[0]][another_index][COVARIANCE_KEY] + e, self.model[SIMILARITIES_KEY][rating[0]][another_index][STD_DEV_FIRST_KEY] + f, self.model[SIMILARITIES_KEY][rating[0]][another_index][STD_DEV_SEC_KEY] + g)

    #update rating incoming as (user_id, item_id, rating)   
    def _update_rating(self, rating):
        diff_ratings = rating[2] - self.matrix[rating[0]][rating[1]]
        old_avg_rating = self.model[AVG_RATINGS_KEY][rating[0]]
        new_size = len(self.model[CO_RATED_KEY][rating[0]][rating[0]])
        new_avg_rating = (diff_ratings/new_size - 1) + old_avg_rating
        self.model[AVG_RATINGS_KEY][rating[0]] = new_avg_rating
        difference_avg = new_avg_rating - old_avg_rating
        e,f,g = 0,0,0
        for another_index, another_user in enumerate(self.matrix):
            #had rated
            if another_user[rating[1]] is not None:
                e = diff_ratings * (another_user[rating[1]] - self.model[AVG_RATINGS_KEY][another_index]) - sum([ difference_avg * ( another_user[item_id] - self.model[AVG_RATINGS_KEY][another_index]) for item_id in self.model[CO_RATED_KEY]][rating[0]][another_index])
                f = diff_ratings**2 + 2*diff_ratings*(rating[2] - new_avg_rating) + len(self.model[CO_RATED_KEY][rating[0]][another_index])*difference_avg**2 - 2*sum([ difference_avg * ( self.matrix[rating[0]][item_id] - old_avg_rating) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index]])
                g = 0
            #hadn't rated
            else:
                e = -sum([ difference_avg * ( another_user[item_id] - self.model[AVG_RATINGS_KEY][another_index]) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index]])
                f = len(self.model[CO_RATED_KEY][rating[0]][another_index])*difference_avg**2 - 2*sum([ difference_avg * ( self.matrix[rating[0]][item_id] - old_avg_rating ) for item_id in self.model[CO_RATED_KEY][rating[0]][another_index] ])
                g = 0

            self._update_similarities(rating[0], another_index, self.model[SIMILARITIES_KEY][rating[0]][another_index][COVARIANCE_KEY] + e, self.model[SIMILARITIES_KEY][rating[0]][another_index][STD_DEV_FIRST_KEY] + f, self.model[SIMILARITIES_KEY][rating[0]][another_index][STD_DEV_SEC_KEY] + g)
        
    #new stream incoming as (user_id, item_id, rating)
    def new_stream(self, rating):
        #rating update
        if self.matrix[rating[0]][rating[1]] != None:
            self._new_rating(rating)
        #new rating
        else:
            self._update_co_rated(rating)
        self.matrix[rating[0]][rating[1]] = rating[2]
    

        
    
    

        
    
    
    
    
        

