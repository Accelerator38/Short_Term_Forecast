import math
import numpy as np
from numba import njit


@njit
def calc_inverse(matrix):
    transpose_a = transpose(matrix)
    diag = decompose_and_get_diag(transpose_a)
    return solve_inverse(transpose_a, diag, create_ident_matrix(len(transpose_a)))


@njit
def transpose(a):
    transpose_m = np.zeros((len(a), len(a[0])))
    for row in range(0, len(a), 1):
        row_i = a[row]
        for column in range(0, len(a[0]), 1):
            transpose_m[column][row] = row_i[column]
    return transpose_m


@njit
def decompose_and_get_diag(transpose_matrix):
    min_size = min(len(transpose_matrix), len(transpose_matrix[0]))
    diag = np.zeros(min_size)
    for minor in range(0, min_size, 1):
        perform_reflection(minor, transpose_matrix, diag)
    return diag


@njit
def perform_reflection(minor, matrix, diag):
    qrt_minor = matrix[minor]
    x_norm_sqr = 0.0
    for row in range(minor, len(qrt_minor), 1):
        c = qrt_minor[row]
        x_norm_sqr += c * c
    a = -math.sqrt(x_norm_sqr) if qrt_minor[minor] > 0.0 else math.sqrt(x_norm_sqr)
    diag[minor] = a
    if a != 0.0:
        qrt_minor[minor] -= a
        for col in range(minor + 1, len(matrix), 1):
            qrt_col = matrix[col]
            alpha = 0.0
            for row in range(minor, len(qrt_col), 1):
                alpha -= qrt_col[row] * qrt_minor[row]
            alpha /= a * qrt_minor[minor]
            for row in range(minor, len(qrt_col), 1):
                qrt_col[row] -= alpha * qrt_minor[row]


@njit
def create_ident_matrix(dimension):
    matrix = np.zeros((dimension, dimension))
    for i in range(0, dimension, 1):
        matrix[i][i] = 1.0
    return matrix


@njit
def solve_inverse(transpose_a, diag, identity):
    n = len(transpose_a)
    m = len(transpose_a[0])
    if len(identity) != m or not is_non_singular(diag):
        return np.zeros((len(transpose_a), len(transpose_a[0])))
    else:
        columns = len(identity[0])
        c_blocks = int((columns + 52 - 1) / 52)
        x_blocks = create_blocks_layout(n, columns)
        y = np.zeros((len(identity), 52))
        alpha = [0.0] * 52
        for k_block in range(0, c_blocks, 1):
            k_start = k_block * 52
            k_end = min(k_start + 52, columns)
            k_width = k_end - k_start
            copy_sub_matrix(0, m - 1, k_start, k_end - 1, y, identity)
            for j in range(0, min(m, n), 1):
                qrt_minor = transpose_a[j]
                factor = 1.0 / (diag[j] * qrt_minor[j])
                fill_matrix(alpha, 0, k_width, 0.0)
                for row in range(j, m, 1):
                    d = qrt_minor[row]
                    y_row = y[row]
                    for k in range(0, k_width, 1):
                        alpha[k] += d * y_row[k]
                for row in range(0, k_width, 1):
                    alpha[row] *= factor
                for row in range(j, m, 1):
                    d = qrt_minor[row]
                    y_row = y[row]
                    for k in range(0, k_width, 1):
                        y_row[k] += alpha[k] * d
            for j in range(len(diag) - 1, -1, -1):
                j_block = int(j / 52)
                j_start = j_block * 52
                factor = 1.0 / diag[j]
                y_j = y[j]
                x_block = x_blocks[j_block * c_blocks + k_block]
                index = (j - j_start) * k_width
                for k in range(0, k_width, 1):
                    y_j[k] *= factor
                    x_block[index] = y_j[k]
                    index += 1
                qrt_j = transpose_a[j]
                for i in range(0, j, 1):
                    r_ij = qrt_j[i]
                    y_i = y[i]
                    for k in range(0, k_width, 1):
                        y_i[k] -= y_j[k] * r_ij
    return get_data(n, columns, x_blocks)


@njit
def is_non_singular(diag):
    for i in range(0, len(diag), 1):
        if abs(diag[i]) <= 0.0:
            return False
    return True


@njit
def create_blocks_layout(rows, columns):
    block_rows = int((rows + 52 - 1) / 52)
    block_columns = int((columns + 52 - 1) / 52)
    blocks = np.zeros((block_rows * block_columns, min(52, rows) * min(52, columns)))
    return blocks


@njit
def copy_sub_matrix(start_row, end_row, start_column, end_column, destination, data):
    rows_count = end_row + 1 - start_row
    columns_count = end_column + 1 - start_column
    if len(destination) >= rows_count and len(destination[0]) >= columns_count:
        for i in range(start_row, end_row + 1, 1):
            row_i = data[i]
            for j in range(start_column, end_column + 1, 1):
                destination[i - start_row][j - start_column] = row_i[j]


@njit
def fill_matrix(a, from_index, to_index, val):
    for i in range(from_index, to_index, 1):
        a[i] = val


@njit
def get_data(rows, columns, blocks):
    data = np.zeros((rows, columns))
    block_rows = int((rows + 52 - 1) / 52)
    block_columns = int((columns + 52 - 1) / 52)
    last_columns = int(columns - (block_columns - 1) * 52)
    for i_block in range(0, block_rows, 1):
        p_start = int(i_block * 52)
        p_end = min(p_start + 52, rows)
        regular_pos = 0
        last_pos = 0
        for p in range(p_start, p_end, 1):
            data_p = data[p]
            block_index = int(i_block * block_columns)
            data_pos = 0
            for j_block in range(0, block_columns - 1, 1):
                array_copy(blocks[block_index], regular_pos, data_p, data_pos, 52)
                block_index += 1
                data_pos += 52
            array_copy(blocks[block_index], last_pos, data_p, data_pos, last_columns)
            regular_pos += 52
            last_pos += last_columns
    return data


@njit
def array_copy(src, src_pos, dest, dest_pos, length):
    for i in range(length):
        dest[i + dest_pos] = src[i + src_pos]


@njit
def multiply_vector(matrix, blocks, v):
    rows = len(matrix)
    columns = len(matrix[0])
    block_size = 52
    out = np.zeros(rows)
    block_rows = int((len(matrix) + block_size - 1) / block_size)
    block_columns = int((columns + block_size - 1) / block_size)
    for i_block in range(0, block_rows, 1):
        p_start = i_block * block_size
        p_end = min(p_start + block_size, rows)
        for j_block in range(0, block_columns, 1):
            block = blocks[i_block * block_columns + j_block]
            q_start = int(j_block * block_size)
            q_end = min(q_start + block_size, columns)
            k = 0
            for p in range(p_start, p_end, 1):
                sum_p = 0
                q = q_start
                while q < q_end - 3:
                    sum_p += block[k] * v[q] + block[k + 1] * v[q + 1] + block[k + 2] * v[q + 2] + block[k + 3] * v[
                        q + 3]
                    k += 4
                    q += 4
                while q < q_end:
                    sum_p += block[k] * v[q]
                    k += 1
                    q += 1
                out[p] += sum_p
    return out


@njit
def multiply_on_vector(matrix, vector):
    rows = len(matrix)
    columns = len(matrix[0])
    ans = np.zeros(rows)
    for i in range(0, rows, 1):
        for j in range(0, columns, 1):
            ans[i] += matrix[i][j] * vector[j]
    return ans


@njit
def multiply_vect_on_vect_trans(vec1, vec2):
    ans = np.zeros((len(vec1), len(vec2)))
    for i in range (0, len(vec1)):
        for j in range(0, len(vec2)):
            ans[i][j] = vec1[i] * vec2[j]
    return ans


@njit
def plus(a, b):
    ans = np.zeros((len(a), len(a[0])))
    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            ans[i][j] = a[i][j] + b[i][j]
    return ans


@njit
def calc_aminus_b(a, b):
    ans = np.zeros(len(a))
    for i in range(0, len(a)):
        ans[i] = a[i] - b[i]
    return ans


@njit
def divide(vector, value):
    ans = np.zeros(len(vector))
    for i in range(0, len(vector)):
        ans[i] = vector[i] / value
    return ans


@njit
def vector_length_square(vector):
    ans = 0
    for i in range(0, len(vector)):
        ans += vector[i] * vector[i]
    return ans


@njit
def vector_length(vector):
    return math.sqrt(vector_length_square(vector))


@njit
def array_from_another(arr, start, end):
    ans = np.zeros(end - start)
    index = 0
    for i in range(start, end):
        ans[index] = arr[i]
        index += 1
    return ans


@njit
def get_max_from_matrix_diag(square_matrix):
    max_value = square_matrix[0][0]
    for i in range(1, len(square_matrix)):
        max_value = max(max_value, square_matrix[i][i])
    return max_value


@njit
def plus_diag(matrix, diag):
    ans = np.copy(matrix)
    for i in range(0, len(diag)):
        ans[i][i] += diag[i]
    return ans


@njit
def create_ident_diag(size, value):
    ans = np.zeros(size)
    for i in range(0, size):
        ans[i] = value
    return ans


@njit
def get_diag_from_matrix(matrix):
    ans = np.zeros(len(matrix))
    for i in range(0, len(matrix)):
        ans[i] = matrix[i][i]
    return ans


@njit
def multiply_vect_on_value(vector, value):
    ans = np.zeros(len(vector))
    for i in range(0, len(vector)):
        ans[i] = vector[i] * value
    return ans


@njit
def calc_a_plus_b(a, b):
    ans = np.zeros(len(a))
    for i in range(0, len(a)):
        ans[i] = a[i] + b[i]
    return ans


@njit
def multiply_vect_trans_on_vect(vec1, vec2):
    ans = 0
    for i in range(0, len(vec1)):
        ans += vec1[i] * vec2[i]
    return ans


@njit
def multiply_vect_on_value(vector, value):
    ans = np.zeros(len(vector))
    for i in range(0, len(vector)):
        ans[i] = vector[i] * value
    return ans
