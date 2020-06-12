from abc import ABC


class PrequentialEvaluator(ABC):
    """
    Description
        A generic class which encapsulates common logic
        of the explicit and implicit version.
    """
    def __init__(self, model, window, n_rec):
        """
        Description
            PrequentialEvaluator's constructor.

        Arguments
            :param model: A recommendation algorithm object.
            :type model: CollaborativeFiltering
            :param window: The size of the evaluation window.
            :type window: int
            :param n_rec: Number of items to be recommended.
            :type n_rec: int
        """
        self.model = model
        self.window = window
        self.window_counter = 0
        self.window_data = []
        self.window_avg_error = 1
        self.n_rec = n_rec

    def _calculate_window_error(self):
        """
        Description
            A function which updates the average error of the
            current window.
        """
        self.window_avg_error = (sum(
            self.window_data) / self.window_counter)

    def _increment_counter(self):
        """
        Description
            A function which increments the current window counter.
        """
        self.window_counter += 1

    def _check_counter(self):
        """
        Description
            A function which checks the current window counter
            reseting it and calculating the window error if reached
            its limit.
        """
        if self.window is None:
            self._calculate_window_error()
        elif self.window_counter >= self.window:
            self._calculate_window_error()
            self.window_data = []
            self.window_counter = 0
