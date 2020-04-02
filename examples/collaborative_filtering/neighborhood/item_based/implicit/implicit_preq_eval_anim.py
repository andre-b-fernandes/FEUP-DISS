import sys
import getopt
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import ItemLSH
from stream.file_stream import FileStream
from evaluators.prequential.\
    implicit_feedback import PrequentialEvaluatorImplicit
from graphic.animation import EvaluationAnimation

path = getopt.getopt(sys.argv[1:], "")[1][0]
fs = FileStream(path, sep="\t")
fs.process_stream_eval_anim(
    PrequentialEvaluatorImplicit, ItemLSH, EvaluationAnimation)
