class PairVariances:

    def set_variance(self, first, second, f_var, s_var):
        if first < second:
            f_var, s_var = s_var, f_var
        self.first_variance, self.second_variance = f_var, s_var

    def variance(self, first, second):
        if first > second:
            return self.first_variance
        else:
            return self.second_variance
