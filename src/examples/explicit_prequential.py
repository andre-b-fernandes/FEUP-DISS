import sys
import getopt
from src.evaluators.prequential.explicit_feedback.prequential_evaluator import PrequentialEvaluatorExplicit
from src.algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedExplicitCF
from src.streams.file_loader import file_generator_evaluator

path = getopt.getopt(sys.argv[1:], "")[1][1]

model = file_generator_evaluator(path, PrequentialEvaluatorExplicit,
                                 UserBasedExplicitCF, sep="\t")