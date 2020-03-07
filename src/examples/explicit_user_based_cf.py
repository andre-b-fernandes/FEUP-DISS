import sys, getopt
from ..algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedCollaborativeFiltering
from ..streams.file_loader import file_generator

path = getopt.getopt(sys.argv[1:],"")[1][1]

model = file_generator(path, UserBasedCollaborativeFiltering, sep="\t")
