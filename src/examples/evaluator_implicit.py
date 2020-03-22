from random import choice
from src.graphic.animation import EvaluationAnimation
from src.evaluators.prequential.implicit_feedback.prequential_evaluator import PrequentialEvaluator
from src.algorithms.collaborative_filtering.neighborhood.implicit_feedback.lsh_neighborhood import LSHBased

n_items = 300
n_users = 300
n_ratings = 2000

matrix = [[None for _ in range(n_items)] for _ in range(n_users)]
streams = [(choice(range(n_users)), choice(range(n_items))) for _ in range(n_ratings)]
lsh = LSHBased(matrix)
evaluator = PrequentialEvaluator(lsh, window=10)
anim = EvaluationAnimation(streams, evaluator)
anim.show()
