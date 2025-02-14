import numpy as np


def closest_to_A_perpendicular_to_B(A, B):
    """Returns closest vector to A which is perpendicular to B"""
    return A - B * np.dot(B, A)
