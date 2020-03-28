import sys
import getopt
from src import UserBasedExplicitCF
from src.evaluators import PrequentialEvaluatorExplicit
from src.graphic import EvaluationAnimation
from src.stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
fs.process_stream_eval_anim(
    PrequentialEvaluatorExplicit,
    UserBasedExplicitCF,
    EvaluationAnimation
)
