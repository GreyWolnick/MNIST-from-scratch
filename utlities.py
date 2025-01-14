import math
import random

def transpose(A: list[list[float]]) -> list[list[float]]:
    """ Compute the transpose of matrix A """
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matrix_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """ Add each element of matrix A by the corresponding element of matrix B """
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_subtract(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """ Subtract each element of matrix A by the corresponding element of matrix B """
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """ Compute the dot product of matrix A and matrix B """
    d_1, d_shared, d_2 = len(A), len(A[0]), len(B[0])
    C = [[0] * d_2 for _ in range(d_1)]

    for i in range(d_1):
        for j in range(d_2):
            for k in range(d_shared):
                C[i][j] += A[i][k] * B[k][j]

    return C

def matrix_divide(A: list[list[float]], B: list[list[float]], stability_term=0) -> list[list[float]]:
    """ Element-wise divides matrix A by matrix B """
    return [[A[i][j] / (B[i][j] + stability_term) for j in range(len(A[0]))] for i in range(len(A))]

def scalar_matrix_multiply(s, A: list[list[float]]) -> list[list[float]]:
    """ Multiplies each element of a matrix by a scalar. """
    return [[s * A_ij for A_ij in row] for row in A]

def scalar_divides_matrix(s, A: list[list[float]]) -> list[list[float]]:
    """ Divides each element of a matrix by a scalar. """
    return [[A_ij / s for A_ij in row] for row in A]

def scalar_matrix_exp(exp: float, A: list[list[float]]) -> list[list[float]]:
    """ Raises each element of a matrix to the power of a scalar exponent. """
    return [[A_ij ** exp for A_ij in row] for row in A]

def create_zeros(shape: tuple) -> list:
    """ Given a vector/matrix shape, return a zero vector/matrix of the same shape """
    if len(shape) == 1:
        return [0] * shape[0]
    elif len(shape) == 2:
        return [[0]* shape[1] for _ in range(shape[0])]

def init_weight_matrix(in_features: int, out_features: int) -> list[list[float]]:
    """ Initialize a weight matrix using Xavier initialization """
    k = math.sqrt(1/in_features)
    return [[random.uniform(-k,k) for _ in range(out_features)] for _ in range(in_features)]

def init_bias_vector(in_features: int, out_features: int) -> list[float]:
    """ Initialize a bias vector using Xavier initialization """
    k = math.sqrt(1 / in_features)
    return [random.uniform(-k,k) for _ in range(out_features)]
