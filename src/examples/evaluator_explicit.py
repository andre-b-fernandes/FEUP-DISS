from random import choice, randint
from src.graphic.animation import EvaluationAnimation
from src.evaluators.prequential.explicit_feedback.prequential_evaluator import PrequentialEvaluatorExplicit
from src.algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import UserBasedExplicitCF

n_items = 300
n_users = 300
n_ratings = 2000

matrix = [[None for _ in range(n_items)] for _ in range(n_users)]
streams = [(choice(range(n_users)), choice(range(n_items)), randint(0, 10)) for _ in range(n_ratings)]
lsh = UserBasedExplicitCF(matrix)
evaluator = PrequentialEvaluatorExplicit(lsh, window=10)
anim = EvaluationAnimation(streams, evaluator)
anim.show()
