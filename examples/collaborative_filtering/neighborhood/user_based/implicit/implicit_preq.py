import sys
import getopt
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import ItemLSH
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import UserBasedImplicitCF
from evaluators.prequential.\
    implicit_feedback import PrequentialEvaluatorImplicit
from stream.file_stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]

fs = FileStream(path, sep="\t")
cf = ItemLSH()
ev = PrequentialEvaluatorImplicit(cf, window=10)
fs.process_stream(ev)
