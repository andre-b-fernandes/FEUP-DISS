import sys
import getopt
from algorithms.collaborative_filtering.neighborhood.\
    explicit_feedback import UserBasedExplicitCF
from evaluators.prequential.\
    explicit_feedback import PrequentialEvaluatorExplicit
from graphic import EvaluationAnimation
from stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
fs.process_stream_eval_anim(
    PrequentialEvaluatorExplicit,
    UserBasedExplicitCF,
    EvaluationAnimation
)
