import sys
import getopt
from src.evaluators.prequential.implicit_feedback.prequential_evaluator import PrequentialEvaluator
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.lsh_neighborhood import LSHBased
from src.streams.file_loader import file_generator_evaluator

path = getopt.getopt(sys.argv[1:], "")[1][1]

model = file_generator_evaluator(path, PrequentialEvaluator,
                                 LSHBased, sep="\t")
