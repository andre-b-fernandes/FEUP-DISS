import unittest
from src.algorithms.collaborative_filtering.neighborhood.explicit_feedback.user_based_cf import CollaborativeFiltering

class UserBasedCollaborativeFilteringTest(unittest.TestCase):
    def verify_model_initialization(self):
        matrix = [
                [ 3, None , None , None , None, None, None , None , None , None ],
                [ 5, None , None , None , None, None, None , None , None , None ],
                [ 8, None , None , None , None, None, None , None , None , None ],
                [ 5, None , None , None , None, None, None , None , None , None ],
                [ 1, None , None , None , None, None, None , None , None , None ],
                [ 9, None , None , None , None, None, None , None , None , None ],
                [ 10, None , None , None , None, None, None , None , None , None ],
                [ 7, None , None , None , None, None, None , None , None , None ],
                [ 6, None , None , None , None, None, None , None , None , None ]
            ]
        pass


if __name__ == '__main__':
    unittest.main()