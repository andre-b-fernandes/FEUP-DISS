
import sys
import getopt
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.lsh_neighborhood import LSHBased
from src.streams.file_loader import file_generator

path = getopt.getopt(sys.argv[1:], "")[1][1]

model = file_generator(path, LSHBased, sep="\t")
