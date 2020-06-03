import sys
import getopt
from algorithms.collaborative_filtering.matrix_factorization\
    .explicit_feedback import MFExplicitPrepSGD
from evaluators.prequential.explicit_feedback import (
    PrequentialEvaluatorExplicit)
from stream.file_stream.explicit import FileStreamExplicit
from graphic import EvaluationStatic

path = getopt.getopt(sys.argv[1:], "")[1][0]
fs = FileStreamExplicit(path, sep="\t")
cf = MFExplicitPrepSGD(lf=20)
ev = PrequentialEvaluatorExplicit(cf)
stat = EvaluationStatic(fs.stream, ev)
stat.process()
