import sys
import getopt
from algorithms.collaborative_filtering.matrix_factorization\
    .explicit_feedback import MatrixFactorizationExplicit
from evaluators.prequential.explicit_feedback import (
    PrequentialEvaluatorExplicit)
from stream.file_stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][0]
fs = FileStream(path, sep="\t")
cf = MatrixFactorizationExplicit(lf=10)
ev = PrequentialEvaluatorExplicit(cf)
fs.process_stream(ev)
