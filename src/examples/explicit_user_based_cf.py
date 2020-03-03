from ..algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedCollaborativeFiltering
from ..streams.file_loader import file_generator

model = file_generator("../datasets/ml-100k/u.data", UserBasedCollaborativeFiltering, sep="\t")