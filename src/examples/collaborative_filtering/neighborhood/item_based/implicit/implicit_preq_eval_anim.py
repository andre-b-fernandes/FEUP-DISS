import sys
import getopt
from src.algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import ItemLSH
from src.stream.file_stream import FileStream
from src.evaluators.prequential.\
    implicit_feedback import PrequentialEvaluatorImplicit
from src.graphic.animation import EvaluationAnimation

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
fs.process_stream_eval_anim(
    PrequentialEvaluatorImplicit, ItemLSH, EvaluationAnimation)
