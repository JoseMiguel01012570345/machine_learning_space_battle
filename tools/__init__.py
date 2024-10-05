import numpy as np

def matrix_or_matrix(a,b):
    """
    returns the matrix a or b, this is, a binary matrix where each component (i,j) is 1 if a[i,j] or
    b[i,j] are equal to 1, else 0
    """
    return np.sign(a + b)