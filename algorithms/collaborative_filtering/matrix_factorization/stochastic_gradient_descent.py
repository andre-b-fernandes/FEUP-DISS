from numpy import array
from data_structures import DynamicArray
from random import uniform


class SGD:
    def __init__(self, lr, reg):
        self.learning_rate = lr
        self.reg_factor = reg

    def _update_factors(self, user_id, item_id, error):
        u_factors = array(self.u[user_id])
        v_factors = array(self.v.col(item_id))
        updated_u = u_factors + self.learning_rate * (
            error * v_factors - self.reg_factor * u_factors)
        updated_v = v_factors + self.learning_rate * (
            error * u_factors - self.reg_factor * v_factors)
        self.u[user_id] = DynamicArray(
            list(updated_u), default_value=uniform(0, 1))
        self.v.set_col(item_id, updated_v)
