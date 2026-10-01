import numpy as np
from numba import njit

import models_transflow_siam.transflow.constants as const
import models_transflow_siam.transflow.matrix as matrix
import math


@njit
def solve_lm_lu(fun, jac, first, max_iter, max_error, delta_error, answers, args=(), constants=(), display=False):
    iter = 0
    lamda = 0.1
    weights_amount = len(first)
    ans_amount = len(answers)
    weights = np.zeros(len(first))
    for i in range(0, weights_amount):
        weights[i] = first[i]
    diagonal = np.zeros(len(first))
    is_solve = True
    while is_solve:
        e = calc_errors(fun, weights, args, constants, answers)
        error = calc_error_square(e)
        old_error = error
        jacobian = jac(weights, args, constants)
        grad = calc_grad(weights_amount, e, jacobian)
        hessian = calc_hessian(weights_amount, ans_amount, jacobian)
        for i in range(0, weights_amount):
            diagonal[i] = hessian[i][i]
        err_lamda = update_lu(fun, args, constants, answers, weights, grad, hessian, diagonal, error, lamda)
        error = err_lamda[0]
        lamda = err_lamda[1]
        iter += 1
        if display:
            print("iteration: ", iter, " error: ", error)
        is_solve = (iter < max_iter) and (error > max_error) and (old_error - error > delta_error)
    return weights


@njit
def solve_lm_qr(fun, jac, first, max_iter, max_error, delta_error, answers, args=(), constants=(), display=False):
    iter = 0
    lamda = 0.1
    weights_amount = len(first)
    ans_amount = len(answers)
    weights = np.zeros(len(first))
    for i in range(0, weights_amount):
        weights[i] = first[i]
    diagonal = np.zeros(len(first))
    is_solve = True
    while is_solve:
        e = calc_errors(fun, weights, args, constants, answers)
        error = calc_error_square(e)
        old_error = error
        jacobian = jac(weights, args, constants)
        grad = calc_grad(weights_amount, e, jacobian)
        hessian = calc_hessian(weights_amount, ans_amount, jacobian)
        for i in range(0, weights_amount):
            diagonal[i] = hessian[i][i]
        err_lamda = update_qr(fun, args, constants, answers, weights, grad, hessian, diagonal, error, lamda)
        error = err_lamda[0]
        lamda = err_lamda[1]
        iter += 1
        if display:
            print("iteration: ", iter, " error: ", error)
        is_solve = (iter < max_iter) and (error > max_error) and (old_error - error > delta_error)
    return weights


@njit
def solve_lm_lu_with_bounds(fun, jac, get_validated_bounds, first, bound_strategy, bounds, max_iter, max_error, delta_error,
                            broyden_period, hessian_update_strategy, answers, args=(), constants=(), display=False):
    iter = 0
    lamda = 0.1
    fun_calls = 0
    weights_amount = len(first)
    ans_amount = len(answers)
    current_bounds = bounds
    weights_int = a_int_all_with_strategy(current_bounds, first, bound_strategy)
    diagonal = np.zeros(len(first))
    fun_calls = fun_calls + 1
    e = calc_errors(fun, first, args, constants, answers)
    fun_calls = fun_calls + weights_amount * 2
    jacobian = jac(weights_int, args, constants, current_bounds)
    error, old_error, weights_int, weights_old, e, e_old, lamda, current_bounds, fun_calls_cur = \
        execute_iteration_lm_lu(fun, args, constants, answers, get_validated_bounds, current_bounds, bound_strategy, lamda,
                                weights_int, e, jacobian, diagonal, weights_amount, ans_amount, display, iter, hessian_update_strategy)
    iter += 1
    fun_calls = fun_calls + fun_calls_cur
    is_solve = (iter < max_iter) and (error > max_error) and (old_error - error > delta_error)
    if not is_solve:
        return iter, fun_calls, a_ext_all_with_strategy(current_bounds, weights_int, bound_strategy)

    while is_solve:
        if iter % broyden_period == 0:
            jacobian = get_estimated_jacobian(jacobian, e_old, e, weights_old, weights_int)
        else:
            fun_calls = fun_calls + weights_amount * 2
            jacobian = jac(weights_int, args, constants, current_bounds)
        error, old_error, weights_int, weights_old, e, e_old, lamda, current_bounds, fun_calls_cur = \
            execute_iteration_lm_lu(fun, args, constants, answers, get_validated_bounds, current_bounds, bound_strategy, lamda,
                                    weights_int, e, jacobian, diagonal, weights_amount, ans_amount, display, iter, hessian_update_strategy)
        iter += 1
        fun_calls = fun_calls + fun_calls_cur
        is_solve = (iter < max_iter) and (error > max_error) and (old_error - error > delta_error)
    return iter, fun_calls, a_ext_all_with_strategy(current_bounds, weights_int, bound_strategy)


@njit
def solve_lm_damping_control(fun, jac, get_validated_bounds, first, bound_strategy, bounds, max_iter, delta_error,
                             e1, e2, e3, e4, tau, broyden_period, hessian_update_strategy, answers, args=(), constants=(), display=False):
    weights_amount = len(first)
    ans_amount = len(answers)
    k = 0
    v = 2
    fun_calls = 0
    current_bounds = bounds
    p_int = a_int_all_with_strategy(bounds, first, bound_strategy)
    fun_calls = fun_calls + 1
    values = calc_errors(fun, first, args, constants, answers)
    error = matrix.vector_length(values)
    fun_calls = fun_calls + weights_amount * 2
    jacobian = jac(p_int, args, constants, bounds)
    grad = calc_grad(weights_amount, values, jacobian)
    hessian = calc_hessian(weights_amount, ans_amount, jacobian)
    stop = error < e1
    lamda = tau * matrix.get_max_from_matrix_diag(hessian)
    if display:
        print("iteration: ", k, " error: ", error, " weights: ", first)
    while not stop and k < max_iter:
        k = k + 1
        rho = -1
        inner_while = True
        while inner_while:
            if hessian_update_strategy == const.HESSIAN_UPDATE_STRATEGY_MARQUARDT:
                new_diag = matrix.get_diag_from_matrix(hessian)
                for i in range(0, len(new_diag)):
                    if new_diag[i] == 0.0:
                        new_diag[i] = 1.0
                hessian_with_lamda = matrix.plus_diag(hessian, matrix.multiply_vect_on_value(new_diag, lamda))
            else:
                hessian_with_lamda = matrix.plus_diag(hessian, matrix.create_ident_diag(len(hessian), lamda))
            piv = [0] * len(hessian_with_lamda)
            lu = calc_LU_decomposition(hessian_with_lamda, piv)
            deltas = solve(lu, grad, piv)
            if matrix.vector_length(deltas) <= e2 * (matrix.vector_length(p_int) + e2):
                stop = True
            else:
                p_int_new = matrix.calc_a_plus_b(p_int, deltas)
                current_bounds = get_validated_bounds(a_ext_all_with_strategy(current_bounds, p_int_new, bound_strategy), args, constants)
                fun_calls = fun_calls + 1
                values_new = calc_errors(fun, a_ext_all_with_strategy(current_bounds, p_int_new, bound_strategy), args, constants, answers)
                error_new = matrix.vector_length(values_new)
                er_old_sqrt = error * error
                er_new_sqrt = error_new * error_new
                rho_bot = (matrix.multiply_vect_trans_on_vect(deltas, matrix.calc_a_plus_b(matrix.multiply_vect_on_value(deltas, lamda), grad)))
                rho = (er_old_sqrt - er_new_sqrt) / rho_bot
                if rho > 0:
                    stop = (error - error_new) < e4 * error
                    p_int_old = p_int
                    p_int = p_int_new
                    error_old = error
                    error = error_new
                    values_old = values
                    values = values_new
                    if k % broyden_period == 0:
                        jacobian = get_estimated_jacobian(jacobian, values_old, values, p_int_old, p_int)
                    else:
                        fun_calls = fun_calls + weights_amount * 2
                        jacobian = jac(p_int, args, constants, current_bounds)
                    grad = calc_grad(weights_amount, values, jacobian)
                    hessian = calc_hessian(weights_amount, ans_amount, jacobian)
                    stop = stop or matrix.vector_length(grad) < e1 or abs(error_old - error) <delta_error
                    lamda = lamda * max(1.0 / 3.0, 1.0 - math.pow(2 * rho - 1, 3.0))
                    v = 2
                else:
                    lamda = lamda * v
                    v = 2 * v
            inner_while = not (rho > 0 or stop)
        stop = error < e3 or stop
        if display:
            print("iteration: ", k, " error: ", error, " weights: ", a_ext_all_with_strategy(bounds, p_int, bound_strategy))
    return k, fun_calls, a_ext_all_with_strategy(current_bounds, p_int, bound_strategy)


@njit
def calc_errors(fun, weights, args, constants, answers):
    errors = np.zeros(len(answers))
    calcs = fun(weights, args, constants)
    for i in range(0, len(answers)):
        errors[i] = calcs[i] - answers[i]
    return errors


@njit
def calc_error_square(errors):
    ans = 0.0
    for i in range(0, len(errors)):
        ans += pow(errors[i], 2.0)
    return ans / 2


@njit
def calc_grad(weights_amount, errors, jacobian):
    grad = np.zeros(weights_amount)
    for i in range(0, len(errors)):
        er = - errors[i]
        derivative = jacobian[i]
        for j in range(0, weights_amount):
            grad[j] += er * derivative[j]
    return grad


@njit
def calc_hessian(weights_amount, ans_amount, jacobian):
    hessian = np.zeros((weights_amount, weights_amount))
    for i in range(0, ans_amount):
        derivative = jacobian[i]
        for j in range(0, weights_amount):
            for k in range(0, weights_amount):
                hessian[j][k] += derivative[j] * derivative[k]
    return hessian


@njit
def update_lu(fun, args, constants, answers, weights, grad, hessian, diagonal, error, lamda):
    done = False
    starting_error = error
    current_error = starting_error
    weights_start = clone_weights(weights)

    while not done:
        apply_lambda(hessian, diagonal, lamda, len(weights))
        piv = [0] * len(hessian)
        lu = calc_LU_decomposition(hessian, piv)
        singular = check_is_nonsingular(lu)
        if singular:
            deltas = solve(lu, grad, piv)
            update_weights(weights, weights_start, deltas)
            current_error = calc_error_square(calc_errors(fun, weights, args, constants, answers))
        if not singular or current_error >= starting_error:
            lamda *= const.SCALE_LAMBDA
            if lamda > const.LAMBDA_MAX:
                lamda = const.LAMBDA_MAX
                done = True
        else:
            lamda /= const.SCALE_LAMBDA
            done = True
    return [current_error, lamda]


@njit
def apply_lambda(hessian, diagonal, lamda, weights_amount):
    for i in range(0, weights_amount):
        hessian[i][i] = diagonal[i] + lamda


@njit
def calc_LU_decomposition(hessian, piv):
    LU = np.zeros((len(hessian), len(hessian[0])))
    for i in range(0, len(hessian)):
        matrix.array_copy(hessian[i], 0, LU[i], 0, len(hessian[0]))
    rows = len(LU)
    cols = len(LU[0])
    for i in range(0, rows):
        piv[i] = i
    LUcolj = np.zeros(rows)
    for j in range(0, cols):
        for i in range(0, rows):
            LUcolj[i] = LU[i][j]
        for i in range(0, rows):
            LUrowi = LU[i]
            kmax = min(i, j)
            s = 0.0
            for k in range(0, kmax):
                s += LUrowi[k] * LUcolj[k]
            LUcolj[i] -= s
            LUrowi[j] = LUcolj[i]
        p = j
        for i in range(j + 1, rows):
            if abs(LUcolj[i]) > abs(LUcolj[p]):
                p = i
        if p != j:
            for k in range(0, cols):
                t = LU[p][k]
                LU[p][k] = LU[j][k]
                LU[j][k] = t
            k = piv[p]
            piv[p] = piv[j]
            piv[j] = k

        if j < rows and LU[j][j] != 0.0:
            for i in range(j + 1, rows):
                LU[i][j] /= LU[j][j]
    return LU


@njit
def clone_weights(weights):
    clone = np.zeros(len(weights))
    for i in range(0, len(weights)):
        clone[i] = weights[i]
    return clone


@njit
def check_is_nonsingular(LU):
    for i in range(0, len(LU[0])):
        if LU[i][i] == 0:
            return False
    return True


@njit
def solve(LU, value, piv):
    count = len(value)
    b = np.zeros(count)
    for i in range(0, len(b)):
        b[i] = value[piv[i]]
    rows = len(LU[0])
    columns = len(LU[0])
    lu = LU
    X = np.zeros(count)
    for i in range(0, rows):
        X[i] = b[i]
        for j in range(0, i):
            X[i] -= lu[i][j] * X[j]
    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, i, -1):
            X[i] -= lu[i][j] * X[j]
        X[i] /= lu[i][i]
    return X


@njit
def update_qr(fun, args, constants, answers, weights, grad, hessian, diagonal, error, lamda):
    done = False
    starting_error = error
    current_error = starting_error
    weights_start = clone_weights(weights)

    while not done:
        apply_lambda(hessian, diagonal, lamda, len(weights))
        deltas = matrix.multiply_on_vector(matrix.calc_inverse(hessian), grad)
        update_weights(weights, weights_start, deltas)
        current_error = calc_error_square(calc_errors(fun, weights, args, constants, answers))

        if starting_error > current_error:
            lamda /= const.SCALE_LAMBDA
            done = True
        else:
            lamda *= const.SCALE_LAMBDA
            if lamda > const.LAMBDA_MAX:
                lamda = const.LAMBDA_MAX
                done = True
    return [current_error, lamda]


@njit
def update_weights(weights, weights_start, deltas):
    for i in range(0, len(weights)):
        weights[i] = weights_start[i] + deltas[i]


@njit
def a_int_all_with_strategy(bounds, a_exts, bound_strategy):
    if bound_strategy == const.BOUND_STRATEGY_HELMUS:
        return a_int_all_helmus(bounds, a_exts)
    else:
        return a_exts


@njit
def a_ext_all_with_strategy(bounds, a_ints, bound_strategy):
    if bound_strategy == const.BOUND_STRATEGY_HELMUS:
        return a_ext_all_helmus(bounds, a_ints)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT:
        return a_ext_all_left_to_right(bounds, a_ints)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_MIRROR:
        return a_ext_all_mirror(bounds, a_ints)
    else:
        return a_ints


@njit
def a_int_all_helmus(bounds, a_exts):
    a_int_ans = np.zeros(len(a_exts))
    for i in range(0, len(a_exts)):
        a_int_ans[i] = a_int_helmus(bounds[i], a_exts[i])
    return a_int_ans


@njit
def a_ext_all_helmus(bounds, a_ints):
    a_ext_ans = np.zeros(len(a_ints))
    for i in range(0, len(a_ints)):
        a_ext_ans[i] = a_ext_helmus(bounds[i], a_ints[i])
    return a_ext_ans


@njit
def a_ext_all_left_to_right(bounds, x_arr):
    ans = np.zeros(len(x_arr))
    for i in range(0, len(x_arr)):
        ans[i] = a_ext_cycled_left_to_right(bounds[i], x_arr[i])
    return ans


@njit
def a_ext_all_mirror(bounds, x_arr):
    ans = np.zeros(len(x_arr))
    for i in range(0, len(x_arr)):
        ans[i] = a_ext_cycled_mirror(bounds[i], x_arr[i])
    return ans


@njit
def a_int_helmus(bound, a_ext_v):
    return math.asin(2.0 * (a_ext_v - bound[0]) / (bound[1] - bound[0]) - 1.0)


@njit
def a_ext_helmus(bound, a_int_v):
    return bound[0] + (math.sin(a_int_v) + 1.0) * (bound[1] - bound[0]) / 2.0


@njit
def a_ext_with_strategy(bound, a_int_v, bound_strategy):
    if bound_strategy == const.BOUND_STRATEGY_HELMUS:
        return a_ext_helmus(bound, a_int_v)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT:
        return a_ext_cycled_left_to_right(bound, a_int_v)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_MIRROR:
        return a_ext_cycled_mirror(bound, a_int_v)
    else:
        return a_int_v


@njit
def a_ext_cycled_left_to_right(bound, x):
    lower = bound[0]
    upper = bound[1]
    step = upper - lower
    if x > upper:
        delta = x - upper
        return lower + delta % step
    elif x < lower:
        delta = lower - x
        return upper - delta % step
    else:
        return x


@njit
def a_ext_cycled_mirror(bound, x):
    lower = bound[0]
    upper = bound[1]
    step = upper - lower
    if x > upper:
        delta = x - upper
        count = int(delta / step)
        if count % 2 == 0:
            return upper - (delta - count * step)
        else:
            return lower + (delta - count * step)
    elif x < lower:
        delta = lower - x
        count = int(delta / step)
        if count % 2 == 0:
            return lower + (delta - count * step)
        else:
            return upper - (delta - count * step)
    else:
        return x


@njit
def execute_iteration_lm_lu(fun, args, constants, answers, get_validated_bounds, bounds, bound_strategy, lamda, weights_int,
                            e, jacobian, diagonal, weights_amount, ans_amount, display, iter, hessian_update_strategy):
    error = calc_error_square(e)
    old_error = error
    grad = calc_grad(weights_amount, e, jacobian)
    hessian = calc_hessian(weights_amount, ans_amount, jacobian)
    for i in range(0, weights_amount):
        diagonal[i] = hessian[i][i]
    weights_int_old = np.copy(weights_int)
    e_old = np.copy(e)
    error, lamda, e, current_bounds, fun_calls = update_lu_with_bounds(fun, args, constants, answers, get_validated_bounds, bounds,
                                                                       bound_strategy, weights_int, grad, hessian, diagonal, error, e, lamda, hessian_update_strategy)
    if display:
        print("iteration: ", iter, " error: ", error, " weights: ", a_ext_all_with_strategy(bounds, weights_int, bound_strategy))
    return error, old_error, weights_int, weights_int_old, e, e_old, lamda, current_bounds, fun_calls


@njit
def update_lu_with_bounds(fun, args, constants, answers, get_validated_bounds, bounds, bound_strategy, weights_int, grad, hessian, diagonal, error, e, lamda, hessian_update_strategy):
    done = False
    starting_error = error
    current_error = starting_error
    current_e = e
    weights_int_start = clone_weights(weights_int)
    current_bounds = bounds
    fun_calls = 0

    while not done:
        apply_lambda_with_strategy(hessian, diagonal, lamda, len(weights_int), hessian_update_strategy)
        piv = [0] * len(hessian)
        lu = calc_LU_decomposition(hessian, piv)
        singular = check_is_nonsingular(lu)
        if singular:
            deltas = solve(lu, grad, piv)
            update_weights(weights_int, weights_int_start, deltas)
            current_bounds = get_validated_bounds(a_ext_all_with_strategy(current_bounds, weights_int, bound_strategy), args, constants)
            fun_calls = fun_calls + 1
            current_e = calc_errors(fun, a_ext_all_with_strategy(bounds, weights_int, bound_strategy), args, constants, answers)
            current_error = calc_error_square(current_e)
        if not singular or current_error >= starting_error:
            lamda *= const.SCALE_LAMBDA
            if lamda > const.LAMBDA_MAX:
                lamda = const.LAMBDA_MAX
                done = True
        else:
            lamda /= const.SCALE_LAMBDA
            done = True
    return current_error, lamda, current_e, current_bounds, fun_calls


@njit
def apply_lambda_with_strategy(hessian, diagonal, lamda, weights_amount, hessian_update_strategy):
    if hessian_update_strategy == const.HESSIAN_UPDATE_STRATEGY_MARQUARDT:
        for i in range(0, weights_amount):
            plus_value = lamda if diagonal[i] == 0 else lamda * diagonal[i]
            hessian[i][i] = diagonal[i] + plus_value
    else:
        for i in range(0, weights_amount):
            hessian[i][i] = diagonal[i] + lamda


@njit
def get_estimated_jacobian(jacobianOld, valuesOld, valuesNew, weightsOld, weightsNew):
    values_delta = matrix.calc_aminus_b(valuesNew, valuesOld)
    weights_delta = matrix.calc_aminus_b(weightsNew, weightsOld)
    jac_on_delta = matrix.multiply_on_vector(jacobianOld, weights_delta)
    top = matrix.calc_aminus_b(values_delta, jac_on_delta)
    bot = matrix.vector_length_square(weights_delta)
    jac_delta = matrix.multiply_vect_on_vect_trans(matrix.divide(top, bot), weights_delta)
    return matrix.plus(jacobianOld, jac_delta)
