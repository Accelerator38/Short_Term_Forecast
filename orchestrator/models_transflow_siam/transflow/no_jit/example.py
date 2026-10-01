import numpy as np
from models_transflow_siam.transflow.no_jit.optimize_lestsqr import solve_lm_lu_with_bounds, solve_lm_damping_control, \
    a_ext_with_strategy, a_ext_all_with_strategy
from models_transflow_siam.transflow import constants as const

# постановка задачи: требуется описать 3 точки xy[[1, -1.5], [2, 0], [3, 4.5]] функцией вида y(x) = a * x^2 - b * x
# с наименьшей погрешностью при помощи изменения коэффициентов a и b
def fun_custom(x, a, b):
    return a * x * x - b * x


def fun(coefs, *args):
    # будем подавать точки в args [[x, y], ...],
    # coefs - искомые коэффициенты a и b исходного уравнения y(x) = a * x^2 - b * x
    xy = args[0]
    a = coefs[0]
    b = coefs[1]
    # составляется система уравнений из n элементов, где n - количество точек и в каждом уравнении все переводится
    # в одну сторону
    eq_system_error = np.zeros(len(xy))
    for i in range(0, len(xy)):
        eq_system_error[i] = fun_custom(xy[i][0], a, b) - xy[i][1]
    return eq_system_error


def get_validated_bounds(coefs, init_bounds, *args):
    # здесь описывается логика по обновлению границ в зависимости от текущих значений параметров.
    # Например следует сделать b > a, и при этом не терять начальные заданные границы.
    # Сделаем копию начальных границ и обновим границы в тех местах, где необхоимо
    new_bounds = np.ndarray.copy(init_bounds)
    a = coefs[0]
    b = coefs[1]
    min_max_a = new_bounds[0]
    min_max_b = new_bounds[1]
    min_max_a[1] = min(min_max_a[1], b)
    min_max_b[0] = max(min_max_b[0], a)
    return new_bounds


def jac(fun, coefs, current_bounds, bound_strategy, ans_amount, *args):
    # численная реализация расчета матрицы якоби.
    # К сожалению из-за внедрения границ поиска в алгоритм марквардта, описанного в работе Investigation of
    # Electrochemical Processes in Solid Oxide Fuel Cells by Modified Levenberg–Marquardt Algorithm: A New Automatic
    # Update Limit Strategy требуется соблюсти требования.
    # Для расчета частных производных по методу двусторонней разницы изменять coefs - внутренние неограниченные
    # параметры и оборачивать их функцией a_ext_all_with_strategy для перевода в ограниченные и произвести расчет функции
    # уже по полученным ограниченным значениям.
    #
    # В целом можно воспользоваться реализацией ниже, но необходимо обрабатывать такие параметры, которые привносят
    # изменение функции только при изменении параметра не менее чем на 1 (такие параметры как n_frac и тип модели границы).
    # Сделать это можно при том условии если известны какие параметры находятся в coefs. Ниже показано как и в каких
    # местах следует добавить обработку таковых параметров, и для понимания данные параметры будут обозначены как 'param_integer'
    x_ext = a_ext_all_with_strategy(current_bounds, coefs, bound_strategy)
    jacobian = np.zeros((ans_amount, len(coefs)))
    for i in range(0, len(coefs)):
        min_max_cur = current_bounds[i]
        true_x_int = coefs[i]
        true_x_ext = x_ext[i]
        # if coefs[i] is param_integer:
        #      d_x = 1.0
        # else:
        #      d_x = abs(0.01 * true_x_int)
        # закомментированный код вместо нижней строки
        d_x = abs(0.01 * true_x_int)
        if d_x == 0:
            d_x = 0.1
        d_x_lower = d_x
        d_x_upper = d_x
        if x_ext[i] == min_max_cur[0] or x_ext[i] == min_max_cur[1]:
            d_x_upper = d_x_lower * 1.2
            # закомментированный код вместо верхней строки
            # if coefs[i] is param_integer:
            #      d_x_upper = d_x_lower + 1.0
            # else:
            #      d_x_upper = d_x_lower * 1.2
        x_ext[i] = a_ext_with_strategy(min_max_cur, true_x_int + d_x_upper, bound_strategy)
        upper2 = fun(x_ext, *args)
        x_ext[i] = a_ext_with_strategy(min_max_cur, true_x_int - d_x_lower, bound_strategy)
        lower2 = fun(x_ext, *args)
        for j in range(0, len(upper2)):
            jacobian[j][i] = (upper2[j] - lower2[j]) / (d_x_upper + d_x_lower)
        x_ext[i] = true_x_ext
    return jacobian


# второй вариант якобиана
def jac2(fun, coefs, current_bounds, bound_strategy, ans_amount, *args):
    x_ext = a_ext_all_with_strategy(current_bounds, coefs, bound_strategy)
    jacobian = np.zeros((ans_amount, len(coefs)))
    for i in range(0, len(coefs)):
        min_max_cur = current_bounds[i]
        true_x_ext = x_ext[i]
        # if coefs[i] is param_integer:
        #      d_x = 1.0
        # else:
        #      d_x = abs(0.01 * true_x_ext)
        # закомментированный код вместо нижней строки
        d_x = abs(0.01 * true_x_ext)
        if d_x == 0:
            d_x = 0.1
        d_x_lower = d_x
        d_x_upper = d_x
        if x_ext[i] == min_max_cur[0] or x_ext[i] == min_max_cur[1]:
            d_x_upper = d_x_lower * 1.2
            # закомментированный код вместо верхней строки
            # if coefs[i] is param_integer:
            #      d_x_upper = d_x_lower + 1.0
            # else:
            #      d_x_upper = d_x_lower * 1.2
        x_ext[i] = a_ext_with_strategy(min_max_cur, true_x_ext + d_x_upper, bound_strategy)
        upper2 = fun(x_ext, *args)
        x_ext[i] = a_ext_with_strategy(min_max_cur, true_x_ext - d_x_lower, bound_strategy)
        lower2 = fun(x_ext, *args)
        for j in range(0, len(upper2)):
            jacobian[j][i] = (upper2[j] - lower2[j]) / (d_x_upper + d_x_lower)
        x_ext[i] = true_x_ext
    return jacobian


xy_fact = np.array([[1.0, -1.5], [2.0, 0.0], [3.0, 4.5]])
# начальное приближение коэффициентов a и b
coefs0 = np.array([0.5, 1.2])
bounds = np.array([[-5.0, 5.0], [-3.0, 6.0]])
# параметры оптимизатора lm_default
bound_strategy = const.BOUND_STRATEGY_CYCLE_MIRROR
max_iter = 25
max_error = 1e-6
delta_error = 1e-6
broyden_period = 5
hessian_update_strategy = const.HESSIAN_UPDATE_STRATEGY_MARQUARDT
# использование default levenberg-marguardt implementation
#   без использования функции расчета якобиана
ans_default_no_jac = solve_lm_lu_with_bounds(fun, None, get_validated_bounds, coefs0, bound_strategy, bounds, max_iter,
                                             max_error, delta_error, broyden_period, hessian_update_strategy, False, xy_fact)
print('lm-default-no_jac', ans_default_no_jac)
#   с использованием функции расчета якобиана
ans_default_jac = solve_lm_lu_with_bounds(fun, jac, get_validated_bounds, coefs0, bound_strategy, bounds, max_iter,
                                          max_error, delta_error, broyden_period, hessian_update_strategy, False, xy_fact)
print('lm-default-jac', ans_default_jac)

# использование levenberg-marguardt with damping control implementation
# параметры оптимизатора lm_damping
e1 = 1e-6
e2 = 1e-6
e3 = max_error
e4 = 1e-6
tau = 9.63e-6
#   без использования функции расчета якобиана
ans_damp_no_jac = solve_lm_damping_control(fun, None, get_validated_bounds, coefs0, bound_strategy, bounds, max_iter,
                                           delta_error, e1, e2, e3, e4, tau, broyden_period, hessian_update_strategy, False, xy_fact)
print('lm-damping-no_jac', ans_damp_no_jac)
#   с использованием функции расчета якобиана
ans_damp_jac = solve_lm_damping_control(fun, jac, get_validated_bounds, coefs0, bound_strategy, bounds, max_iter,
                                           delta_error, e1, e2, e3, e4, tau, broyden_period, hessian_update_strategy, False, xy_fact)
print('lm-damping-jac', ans_damp_jac)
