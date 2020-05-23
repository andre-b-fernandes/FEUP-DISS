import matplotlib.pyplot as plt
from progress.bar import Bar


class EvaluationStatic:
    def __init__(self, stream, evaluator):
        self.stream = stream
        self.evaluator = evaluator
        self.x = range(len(stream))
        self.err_rate = []
        self.elap_nr = []
        self.elap_rec = []

    def evaluate(self):
        bar = Bar('Evaluating', max=len(self.x))
        for element in self.stream:
            err, elap_rec, elap_nr = self.evaluator.new_rating(element)
            self.err_rate.append(err)
            self.elap_rec.append(elap_rec)
            self.elap_nr.append(elap_nr)
            bar.next()
        bar.finish()

    def plot(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Metrics')
        axs[0].plot(self.x, self.err_rate, "r", label="Error")
        axs[0].legend()
        axs[1].plot(self.x, self.elap_nr, "g", label="Elapsed NR time")
        axs[1].legend()
        axs[2].plot(self.x, self.elap_rec, "b", label="Elapsed Rec time")
        axs[2].legend()

    def show(self):
        plt.show()

    def process(self):
        self.evaluate()
        self.plot()
        self.show()
