import sys
import getopt
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import ItemLSH
from stream.file_stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
cf = ItemLSH()
fs.process_stream(cf)
