import math
import numpy as np


def array_copy(src, src_pos, dest, dest_pos, length):
    for i in range(length):
        dest[i + dest_pos] = src[i + src_pos]


def multiply_on_vector(matrix, vector):
    rows = len(matrix)
    columns = len(matrix[0])
    ans = np.zeros(rows)
    for i in range(0, rows, 1):
        for j in range(0, columns, 1):
            ans[i] += matrix[i][j] * vector[j]
    return ans


def multiply_vect_on_vect_trans(vec1, vec2):
    ans = np.zeros((len(vec1), len(vec2)))
    for i in range (0, len(vec1)):
        for j in range(0, len(vec2)):
            ans[i][j] = vec1[i] * vec2[j]
    return ans


def plus(a, b):
    ans = np.zeros((len(a), len(a[0])))
    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            ans[i][j] = a[i][j] + b[i][j]
    return ans


def calc_aminus_b(a, b):
    ans = np.zeros(len(a))
    for i in range(0, len(a)):
        ans[i] = a[i] - b[i]
    return ans


def divide(vector, value):
    ans = np.zeros(len(vector))
    for i in range(0, len(vector)):
        ans[i] = vector[i] / value
    return ans


def vector_length_square(vector):
    ans = 0
    for i in range(0, len(vector)):
        ans += vector[i] * vector[i]
    return ans


def vector_length(vector):
    return math.sqrt(vector_length_square(vector))


def get_max_from_matrix_diag(square_matrix):
    max_value = square_matrix[0][0]
    for i in range(1, len(square_matrix)):
        max_value = max(max_value, square_matrix[i][i])
    return max_value


def plus_diag(matrix, diag):
    ans = np.copy(matrix)
    for i in range(0, len(diag)):
        ans[i][i] += diag[i]
    return ans


def create_ident_diag(size, value):
    ans = np.zeros(size)
    for i in range(0, size):
        ans[i] = value
    return ans


def get_diag_from_matrix(matrix):
    ans = np.zeros(len(matrix))
    for i in range(0, len(matrix)):
        ans[i] = matrix[i][i]
    return ans


def multiply_vect_on_value(vector, value):
    ans = np.zeros(len(vector))
    for i in range(0, len(vector)):
        ans[i] = vector[i] * value
    return ans


def calc_a_plus_b(a, b):
    ans = np.zeros(len(a))
    for i in range(0, len(a)):
        ans[i] = a[i] + b[i]
    return ans


def multiply_vect_trans_on_vect(vec1, vec2):
    ans = 0
    for i in range(0, len(vec1)):
        ans += vec1[i] * vec2[i]
    return ans
