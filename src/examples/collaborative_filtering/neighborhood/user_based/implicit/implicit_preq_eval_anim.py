import sys
import getopt
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback import UserBasedImplicitCF
from src.evaluators.prequential.implicit_feedback import PrequentialEvaluatorImplicit
from src.graphic import EvaluationAnimation
from src.stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
fs.process_stream_eval_anim(PrequentialEvaluatorImplicit, UserBasedImplicitCF, EvaluationAnimation)
