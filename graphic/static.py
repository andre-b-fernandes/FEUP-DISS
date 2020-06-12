import matplotlib.pyplot as plt
from progress.bar import Bar


class EvaluationStatic:
    """
    Description
        A class which displays a graphic which portrays the evolution
        of processing and recommendation time as well as accuracy
        of an algorithm when a dataset is processed incrementally.
    """
    def __init__(self, stream, evaluator):
        """
        Description
            EvaluationStatic's constructor.

        Arguments
            :param stream: A stream of ratings.
            :type stream: list
            :param evaluator: A evaluator object.
            :type evaluator: PrequentialEvaluator
        """
        self.stream = stream
        self.evaluator = evaluator
        self.x = range(len(stream))
        self.err_rate = []
        self.elap_nr = []
        self.elap_rec = []

    def evaluate(self):
        """
        Description
            A function which evaluates the data stream.
        """
        bar = Bar('Evaluating', max=len(self.x))
        for element in self.stream:
            err, elap_rec, elap_nr = self.evaluator.new_rating(element)
            self.err_rate.append(err)
            self.elap_rec.append(elap_rec)
            self.elap_nr.append(elap_nr)
            bar.next()
        bar.finish()

    def plot(self):
        """
        Description
            A function which plots the 3 subplots.
        """
        fig, axs = plt.subplots(3)
        fig.suptitle('Metrics')
        axs[0].plot(self.x, self.err_rate, "r", label="Average error.")
        axs[0].legend()
        axs[1].plot(self.x, self.elap_nr, "g", label="Rating process time.")
        axs[1].legend()
        axs[2].plot(self.x, self.elap_rec, "b", label="Recommendation time.")
        axs[2].legend()

    def export(self, show=False):
        """
        Description
            A function which exports the plots to an image format. Displays
            it if show=True.

        Arguments
            :param show: Does it show the image or not.
            :type show: boolean
        """
        if show:
            plt.show()
        plt.savefig("output.png")

    def process(self, show=False):
        """
        Description
            A function which processes a data stream and creates a graph.

        Arguments
            :param show: Does it show the image or not.
            :type show: boolean
        """
        self.evaluate()
        self.plot()
        self.export(show)
