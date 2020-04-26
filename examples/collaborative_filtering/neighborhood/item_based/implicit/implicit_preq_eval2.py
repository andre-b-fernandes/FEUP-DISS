import sys
import getopt
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import ItemBasedImplicitCF
# from evaluators.prequential.\
#      implicit_feedback import PrequentialEvaluatorImplicit
from stream.file_stream.implicit import FileStreamImplicit

path = getopt.getopt(sys.argv[1:], "")[1][0]
fs = FileStreamImplicit(path, sep="\t")
cf = ItemBasedImplicitCF()
cf.parallel_process_stream(fs.stream, 4)
# ev = PrequentialEvaluatorImplicit(cf)
# fs.process_stream(ev)
