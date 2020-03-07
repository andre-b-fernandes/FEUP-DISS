class PairVariances:    
    def set_variance(self, first, second, first_variance, second_variance):
        if first < second:
            first_variance, second_variance = second_variance, first_variance
        self.first_variance, self.second_variance = first_variance, second_variance
    
    def variance(self, first, second):
        if first > second:
            return self.first_variance
        else:
            return self.second_variance
    