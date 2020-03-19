import sys
import getopt
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.user_based_cf import UserBasedImplicitCF
from src.streams.file_loader import file_generator

path = getopt.getopt(sys.argv[1:], "")[1][1]

model = file_generator(path, UserBasedImplicitCF, sep="\t")
