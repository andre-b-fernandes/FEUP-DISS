import sys
import getopt
from algorithms.collaborative_filtering.matrix_factorization\
    .explicit_feedback import MatrixFactorizationExplicit
from evaluators.prequential.explicit_feedback import (
    PrequentialEvaluatorExplicit)
from stream.file_stream.explicit import FileStreamExplicit
from graphic import EvaluationAnimation

path = getopt.getopt(sys.argv[1:], "")[1][0]
fs = FileStreamExplicit(path, sep="\t")
cf = MatrixFactorizationExplicit(lf=10)
ev = PrequentialEvaluatorExplicit(cf)
fs.process_stream_eval_anim(ev, EvaluationAnimation)
