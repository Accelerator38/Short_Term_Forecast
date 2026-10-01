import math

import numpy as np
import models_transflow_siam.transflow.no_jit.matrices as matrix
import models_transflow_siam.transflow.constants as const

# solve_lm_lu_with_bounds возвращает результат поиска коэффициентов для решения задачи поиска минимума функции. Используется
# алгоритм Левеньерга-Маргуардта с модификациями: ограничение поиска в задданом диапазоне, стратегия обновления
# матрицы Гессе, аппроксимация матрицы Якоби.
#
# входные параметры:
# def fun - функция расчета отклонений fun(double[] x, *args), принимает double[] x - массив значений искомых парамтеров,
#       args - кортеж дополнительных параметров, необходимых для расчета. Возвращает массив отклонений double[] в формате
#       (посчитанное значение - фактическое);
# def jac - функция расчета якобиана jac(def fun, double[] x_arr, double[][] current_bounds, int bound_strategy, int answers_amount, *args)
#       принимает def fun - функцию, описанную выше, double[] x_arr - массив текущих искомых параметров, double[][] current_bounds - обновленный массив
#       границ формата [[min, max], ...](подается во внутренней логике оптимизатора нужен для того, чтобы какой-либо параметр
#       не выходил за определенную границу, которая может меняться), int bound_strategy - параметр, отвечающий за способ ограничения поиска),
#       int answers_amount - число элементов в выходной функции fun (подается во внутренней логике оптимизатора),  *args - кортеж
#       дополнительных параметров, необходимых для расчета. jac Возвращает матрицу Якоби в виде двумерного массива double[][]
#       (Если вместо jac подать None, то якобиан будет расчитываться по умолчанию численным способом);
# def get_validated_bounds - функция по обновлению границ поиска в зависимости от текущих параметров.
#       get_validated_bounds(double[] x_arr, bounds, *args) принимает double[] x_arr - текущие значения искомых параметров,
#       bounds - массив изначально заданных границ в формате [[min, max], ...], *args - кортеж дополнительных параметров,
#       необходимых для расчета. Возвращает double[][] обновленный двумерный массив границ в формате [[min, max], ...]
# double[] x0 - массив начального приближения
# int bound_strategy - параметр, отвечащий за стратегию ограничения поиска в задданых границах. Может быть равен:
#       BOUND_STRATEGY_HELMUS = 0
#       BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT = 1
#       BOUND_STRATEGY_CYCLE_MIRROR = 2
# double[][] bounds - двумерный массив границ поиска в формате [[min, max], ...]
# int max_iter - критейри остановки. Максимальное значение итераций марквардта при решении задачи поиска минимума
# double max_error - критерий остановки. Максимальное квадратичное отклониние
# double delta_error - критерий остановки.  максимальное квадратичное отклониние от предыдущей итерации марквардта
# int broyden_period - параметр, отвечающий за периоодичность замены численного расчета матрицы Якоби аппроксимацией
#       Бройдена. Каждая broyden_period итерация марквардта использует аппроксимацию матрицы Якоби
# int hessian_update_strategy - параметр, отвечающий за способ обновелния матрицы Гессе. Может быть равен:
#                               HESSIAN_UPDATE_STRATEGY_MARQUARDT = 0
#                               HESSIAN_UPDATE_STRATEGY_DEFAULT = 1
# boolean display - параметр отвечающий за вывод информации расчета по итерациям
# *args - кортеж дополнительных параметров, необходимых для расчета fun.
#
# Возвращаемый параметр:
# tuple(int iter, int fun_calls, double[] x), где iter - количество итераций марквардта, fun_calls - количество вызовов функции
#       x - массив найденных параметров
def solve_lm_lu_with_bounds(fun, x0, bounds, get_validated_bounds, hyper_params, jac=None, display=None, *args):
    bound_strategy = hyper_params['bound_strategy']
    max_iter = hyper_params['max_iter']
    max_error = hyper_params['max_error']
    delta_error = hyper_params['delta_error']
    broyden_period = hyper_params['broyden_period']
    hessian_update_strategy = hyper_params['hessian_update_strategy']
    x0 = np.ndarray.copy(x0)
    if jac is None:
        jac = calc_numerical_jacobian
    iter = 0
    lamda = 0.1
    fun_calls = 0
    weights_amount = len(x0)
    current_bounds = bounds
    x_int = a_int_all_with_strategy(current_bounds, x0, bound_strategy)
    diagonal = np.zeros(len(x0))
    fun_calls = fun_calls + 1
    e = fun(x0, *args)
    ans_amount = len(e)
    fun_calls = fun_calls + weights_amount * 2
    jacobian = jac(fun, x_int, current_bounds, bound_strategy, ans_amount, *args)
    error, old_error, x_int, weights_old, e, e_old, lamda, current_bounds, fun_calls_cur = \
        execute_iteration(fun, get_validated_bounds, current_bounds, bounds, bound_strategy, lamda,
                          x_int, e, jacobian, diagonal, weights_amount, ans_amount, display, iter, hessian_update_strategy, *args)
    iter += 1
    fun_calls = fun_calls + fun_calls_cur
    is_solve = (iter < max_iter) and (error > max_error) and (old_error - error > delta_error)
    if not is_solve:
        return iter, fun_calls, a_ext_all_with_strategy(current_bounds, x_int, bound_strategy)

    while is_solve:
        if iter % broyden_period == 0:
            weights_delta = matrix.calc_aminus_b(x_int, weights_old)
            broyden_bot = matrix.vector_length_square(weights_delta)
            if broyden_bot != 0:
                jacobian = get_estimated_jacobian(jacobian, e_old, e, weights_delta, broyden_bot)
            else:
                fun_calls = fun_calls + weights_amount * 2
                jacobian = jac(fun, x_int, current_bounds, bound_strategy, ans_amount, *args)
        else:
            fun_calls = fun_calls + weights_amount * 2
            jacobian = jac(fun, x_int, current_bounds, bound_strategy, ans_amount, *args)
        error, old_error, x_int, weights_old, e, e_old, lamda, current_bounds, fun_calls_cur = \
            execute_iteration(fun, get_validated_bounds, current_bounds, bounds, bound_strategy, lamda,
                              x_int, e, jacobian, diagonal, weights_amount, ans_amount, display, iter, hessian_update_strategy, *args)
        iter += 1
        fun_calls = fun_calls + fun_calls_cur
        is_solve = (iter < max_iter) and (error > max_error) and (old_error - error > delta_error)
    return iter, fun_calls, a_ext_all_with_strategy(current_bounds, x_int, bound_strategy)


# solve_lm_lu_with_bounds возвращает результат поиска коэффициентов для решения задачи поиска минимума функции. Используется
# алгоритм Левеньерга-Маргуардта с модификациями: ограничение поиска в задданом диапазоне, стратегия обновления
# матрицы Гессе, аппроксимация матрицы Якоби.
#
# входные параметры:
# def fun - функция расчета отклонений fun(double[] x, *args), принимает double[] x - массив значений искомых парамтеров,
#       args - кортеж дополнительных параметров, необходимых для расчета. Возвращает массив отклонений double[] в формате
#       (посчитанное значение - фактическое);
# def jac - функция расчета якобиана jac(def fun, double[] x_arr, double[][] current_bounds, int bound_strategy, int answers_amount, *args)
#       принимает def fun - функцию, описанную выше, double[] x_arr - массив текущих искомых параметров, double[][] current_bounds - обновленный массив
#       границ формата [[min, max], ...](подается во внутренней логике оптимизатора нужен для того, чтобы какой-либо параметр
#       не выходил за определенную границу, которая может меняться), int bound_strategy - параметр, отвечающий за способ ограничения поиска),
#       int answers_amount - число элементов в выходной функции fun (подается во внутренней логике оптимизатора),  *args - кортеж
#       дополнительных параметров, необходимых для расчета. jac Возвращает матрицу Якоби в виде двумерного массива double[][]
#       (Если вместо jac подать None, то якобиан будет расчитываться по умолчанию численным способом);
# def get_validated_bounds - функция по обновлению границ поиска в зависимости от текущих параметров.
#       get_validated_bounds(double[] x_arr, bounds, *args) принимает double[] x_arr - текущие значения искомых параметров,
#       bounds - массив изначально заданных границ в формате [[min, max], ...], *args - кортеж дополнительных параметров,
#       необходимых для расчета. Возвращает double[][] обновленный двумерный массив границ в формате [[min, max], ...]
# double[] x0 - массив начального приближения
# int bound_strategy - параметр, отвечащий за стратегию ограничения поиска в задданых границах. Может быть равен:
#       BOUND_STRATEGY_HELMUS = 0
#       BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT = 1
#       BOUND_STRATEGY_CYCLE_MIRROR = 2
# double[][] bounds - двумерный массив границ поиска в формате [[min, max], ...]
# int max_iter - критейри остановки. Максимальное значение итераций марквардта при решении задачи поиска минимума
# double delta_error - критерий остановки.  максимальное квадратичное отклониние от предыдущей итерации марквардта
# double tau - отвечат за начальное задание лямбды в алгоритме Марквардта
# double e1 - критерий остановки. Максимальная величина градиента
# double e2 - критерий остановки. Максимальное относительное изменение длины вектора искомых параметров
# double e3 - критерий остановки. Максимальное квадратичное отклониние
# double e4 - критерий остановки. Максимальное относительное уменьшение величины ошибки
# int broyden_period - параметр, отвечающий за периоодичность замены численного расчета матрицы Якоби аппроксимацией
#       Бройдена. Каждая broyden_period итерация марквардта использует аппроксимацию матрицы Якоби
# int hessian_update_strategy - параметр, отвечающий за способ обновелния матрицы Гессе. Может быть равен:
#                               HESSIAN_UPDATE_STRATEGY_MARQUARDT = 0
#                               HESSIAN_UPDATE_STRATEGY_DEFAULT = 1
# boolean display - параметр отвечающий за вывод информации расчета по итерациям
# *args - кортеж дополнительных параметров, необходимых для расчета fun.
#
# Возвращаемый параметр:
# tuple(int iter, int fun_calls, double[] x), где iter - количество итераций марквардта, fun_calls - количество вызовов функции
#       x - массив найденных параметров
def solve_lm_damping_control(fun, jac, get_validated_bounds, x0, bound_strategy, bounds, max_iter, delta_error,
                             e1, e2, e3, e4, tau, broyden_period, hessian_update_strategy, display, *args):
    if jac is None:
        jac = calc_numerical_jacobian
    weights_amount = len(x0)
    k = 0
    v = 2
    fun_calls = 0
    current_bounds = bounds
    p_int = a_int_all_with_strategy(bounds, x0, bound_strategy)
    fun_calls = fun_calls + 1
    values = fun(x0, *args)
    ans_amount = len(values)
    error = matrix.vector_length(values)
    fun_calls = fun_calls + weights_amount * 2
    jacobian = jac(fun, p_int, bounds, bound_strategy, ans_amount, *args)
    grad = calc_grad(weights_amount, values, jacobian)
    hessian = calc_hessian(weights_amount, ans_amount, jacobian)
    stop = error < e1
    lamda = tau * matrix.get_max_from_matrix_diag(hessian)
    if display:
        print("iteration: ", k, " error: ", error, " weights: ", x0)
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
                current_bounds = get_validated_bounds(a_ext_all_with_strategy(current_bounds, p_int_new, bound_strategy), bounds, *args)
                fun_calls = fun_calls + 1
                values_new = fun(a_ext_all_with_strategy(current_bounds, p_int_new, bound_strategy), *args)
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
                        weights_delta = matrix.calc_aminus_b(p_int, p_int_old)
                        broyden_bot = matrix.vector_length_square(weights_delta)
                        if broyden_bot != 0:
                            jacobian = get_estimated_jacobian(jacobian, values_old, values, weights_delta, broyden_bot)
                        else:
                            fun_calls = fun_calls + weights_amount * 2
                            jacobian = jac(fun, p_int, current_bounds, bound_strategy, ans_amount, *args)
                    else:
                        fun_calls = fun_calls + weights_amount * 2
                        jacobian = jac(fun, p_int, current_bounds, bound_strategy, ans_amount, *args)
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


def execute_iteration(fun, get_validated_bounds, current_bounds, init_bounds, bound_strategy, lamda, x_int,
                      e, jacobian, diagonal, weights_amount, ans_amount, display, iter, hessian_update_strategy, *args):
    error = calc_error_square(e)
    old_error = error
    grad = calc_grad(weights_amount, e, jacobian)
    hessian = calc_hessian(weights_amount, ans_amount, jacobian)
    for i in range(0, weights_amount):
        diagonal[i] = hessian[i][i]
    weights_int_old = np.copy(x_int)
    e_old = np.copy(e)
    error, lamda, e, current_bounds_new, fun_calls = update_lu(fun, get_validated_bounds, current_bounds, init_bounds, bound_strategy,
                                                               x_int, grad, hessian, diagonal, error, e, lamda,
                                                               hessian_update_strategy, *args)
    if display:
        print("iteration: ", iter, " error: ", error, " weights: ", a_ext_all_with_strategy(current_bounds, x_int, bound_strategy))
    return error, old_error, x_int, weights_int_old, e, e_old, lamda, current_bounds_new, fun_calls


def a_int_all_with_strategy(bounds, a_exts, bound_strategy):
    if bound_strategy == const.BOUND_STRATEGY_HELMUS:
        return a_int_all_helmus(bounds, a_exts)
    else:
        return a_exts


def a_ext_all_with_strategy(bounds, a_ints, bound_strategy):
    if bound_strategy == const.BOUND_STRATEGY_HELMUS:
        return a_ext_all_helmus(bounds, a_ints)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT:
        return a_ext_all_left_to_right(bounds, a_ints)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_MIRROR:
        return a_ext_all_mirror(bounds, a_ints)
    else:
        return a_ints


def a_int_all_helmus(bounds, a_exts):
    a_int_ans = np.zeros(len(a_exts))
    for i in range(0, len(a_exts)):
        a_int_ans[i] = a_int_helmus(bounds[i], a_exts[i])
    return a_int_ans


def a_ext_all_helmus(bounds, a_ints):
    a_ext_ans = np.zeros(len(a_ints))
    for i in range(0, len(a_ints)):
        a_ext_ans[i] = a_ext_helmus(bounds[i], a_ints[i])
    return a_ext_ans


def a_ext_all_left_to_right(bounds, x_arr):
    ans = np.zeros(len(x_arr))
    for i in range(0, len(x_arr)):
        ans[i] = a_ext_cycled_left_to_right(bounds[i], x_arr[i])
    return ans


def a_ext_all_mirror(bounds, x_arr):
    ans = np.zeros(len(x_arr))
    for i in range(0, len(x_arr)):
        ans[i] = a_ext_cycled_mirror(bounds[i], x_arr[i])
    return ans


def a_int_helmus(bound, a_ext_v):
    return math.asin(2.0 * (a_ext_v - bound[0]) / (bound[1] - bound[0]) - 1.0)


def a_ext_helmus(bound, a_int_v):
    return bound[0] + (math.sin(a_int_v) + 1.0) * (bound[1] - bound[0]) / 2.0


def a_ext_with_strategy(bound, a_int_v, bound_strategy):
    if bound_strategy == const.BOUND_STRATEGY_HELMUS:
        return a_ext_helmus(bound, a_int_v)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT:
        return a_ext_cycled_left_to_right(bound, a_int_v)
    elif bound_strategy == const.BOUND_STRATEGY_CYCLE_MIRROR:
        return a_ext_cycled_mirror(bound, a_int_v)
    else:
        return a_int_v


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


def a_ext_cycled_mirror(bound, x):
    if not check_value_in_range(x, bound):
        step = bound[1] - bound[0]
        pi_half = math.pi / 2.0
        sin_x = math.sin(x * math.pi / step - pi_half - bound[0] * math.pi / step)
        return (math.asin(sin_x) + pi_half) / math.pi * step + bound[0]
    else:
        return x


def check_value_in_range(x, bound):
    return bound[0] <= x <= bound[1]


def get_estimated_jacobian(jacobianOld, valuesOld, valuesNew, weights_delta, bot):
    values_delta = matrix.calc_aminus_b(valuesNew, valuesOld)
    jac_on_delta = matrix.multiply_on_vector(jacobianOld, weights_delta)
    top = matrix.calc_aminus_b(values_delta, jac_on_delta)
    jac_delta = matrix.multiply_vect_on_vect_trans(matrix.divide(top, bot), weights_delta)
    return matrix.plus(jacobianOld, jac_delta)


def calc_numerical_jacobian(fun, x_arr, current_bounds, bound_strategy, answers_amount, *args):
    x_ext = a_ext_all_with_strategy(current_bounds, x_arr, bound_strategy)
    jacobian = np.zeros((answers_amount, len(x_arr)))
    for i in range(0, len(x_arr)):
        min_max_cur = current_bounds[i]
        true_x_int = x_arr[i]
        true_x_ext = x_ext[i]
        d_x = abs(0.01 * true_x_int)
        if d_x == 0:
            d_x = 0.1
        d_x_lower = d_x
        d_x_upper = d_x
        if x_ext[i] == min_max_cur[0] or x_ext[i] == min_max_cur[1]:
            d_x_upper = d_x_lower * 1.2
        x_ext[i] = a_ext_with_strategy(min_max_cur, true_x_int + d_x_upper, bound_strategy)
        upper2 = fun(x_ext, *args)
        x_ext[i] = a_ext_with_strategy(min_max_cur, true_x_int - d_x_lower, bound_strategy)
        lower2 = fun(x_ext, *args)
        for j in range(0, len(upper2)):
            jacobian[j][i] = (upper2[j] - lower2[j]) / (d_x_upper + d_x_lower)
        x_ext[i] = true_x_ext
    return jacobian


def calc_errors(fun, weights, args, constants, answers):
    errors = np.zeros(len(answers))
    calcs = fun(weights, args, constants)
    for i in range(0, len(answers)):
        errors[i] = calcs[i] - answers[i]
    return errors


def calc_error_square(errors):
    ans = 0.0
    for i in range(0, len(errors)):
        ans += pow(errors[i], 2.0)
    return ans / 2


def calc_grad(weights_amount, errors, jacobian):
    grad = np.zeros(weights_amount)
    for i in range(0, len(errors)):
        er = - errors[i]
        derivative = jacobian[i]
        for j in range(0, weights_amount):
            grad[j] += er * derivative[j]
    return grad


def calc_hessian(weights_amount, ans_amount, jacobian):
    hessian = np.zeros((weights_amount, weights_amount))
    for i in range(0, ans_amount):
        derivative = jacobian[i]
        for j in range(0, weights_amount):
            for k in range(0, weights_amount):
                hessian[j][k] += derivative[j] * derivative[k]
    return hessian


def update_lu(fun, get_validated_bounds, current_bounds, init_bounds, bound_strategy, weights_int, grad, hessian,
              diagonal, error, e, lamda, hessian_update_strategy, *args):
    done = False
    starting_error = error
    current_error = starting_error
    current_e = e
    weights_int_start = clone_weights(weights_int)
    current_bounds_new = current_bounds
    fun_calls = 0

    while not done:
        apply_lambda(hessian, diagonal, lamda, len(weights_int), hessian_update_strategy)
        piv = [0] * len(hessian)
        lu = calc_LU_decomposition(hessian, piv)
        singular = check_is_nonsingular(lu)
        if singular:
            deltas = solve(lu, grad, piv)
            update_weights(weights_int, weights_int_start, deltas)
            current_bounds_new = get_validated_bounds(a_ext_all_with_strategy(current_bounds_new, weights_int, bound_strategy), init_bounds, *args)
            fun_calls = fun_calls + 1
            current_e = fun(a_ext_all_with_strategy(current_bounds_new, weights_int, bound_strategy), *args)
            current_error = calc_error_square(current_e)
        if not singular or current_error >= starting_error:
            lamda *= const.SCALE_LAMBDA
            if lamda > const.LAMBDA_MAX:
                lamda = const.LAMBDA_MAX
                done = True
        else:
            lamda /= const.SCALE_LAMBDA
            done = True
    return current_error, lamda, current_e, current_bounds_new, fun_calls


def apply_lambda(hessian, diagonal, lamda, weights_amount, hessian_update_strategy):
    if hessian_update_strategy == const.HESSIAN_UPDATE_STRATEGY_MARQUARDT:
        for i in range(0, weights_amount):
            plus_value = lamda if diagonal[i] == 0 else lamda * diagonal[i]
            hessian[i][i] = diagonal[i] + plus_value
    else:
        for i in range(0, weights_amount):
            hessian[i][i] = diagonal[i] + lamda


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


def clone_weights(weights):
    clone = np.zeros(len(weights))
    for i in range(0, len(weights)):
        clone[i] = weights[i]
    return clone


def check_is_nonsingular(LU):
    for i in range(0, len(LU[0])):
        if LU[i][i] == 0:
            return False
    return True


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


def update_weights(weights, weights_start, deltas):
    for i in range(0, len(weights)):
        weights[i] = weights_start[i] + deltas[i]
