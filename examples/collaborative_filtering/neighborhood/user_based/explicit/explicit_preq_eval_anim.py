import sys
import getopt
import UserBasedExplicitCF
from evaluators import PrequentialEvaluatorExplicit
from graphic import EvaluationAnimation
from stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
fs.process_stream_eval_anim(
    PrequentialEvaluatorExplicit,
    UserBasedExplicitCF,
    EvaluationAnimation
)
