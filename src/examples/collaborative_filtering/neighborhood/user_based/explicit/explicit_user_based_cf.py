import sys
import getopt
from src.algorithms.collaborative_filtering.neighborhood.\
    explicit_feedback import UserBasedExplicitCF
from src.stream import FileStream

path = getopt.getopt(sys.argv[1:], "")[1][1]
fs = FileStream(path, sep="\t")
fs.process_stream(UserBasedExplicitCF())
