import sys
import getopt
from algorithms.collaborative_filtering.neighborhood.\
    implicit_feedback import UserBasedImplicitCF
from stream.file_stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]

fs = FileStream(path, sep="\t")
cf = UserBasedImplicitCF()
fs.process_stream(cf)
