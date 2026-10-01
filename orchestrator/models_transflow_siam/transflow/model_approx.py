import math
import numpy as np
from numba import njit

from models_transflow_siam.transflow.optimize_lestsq import solve_lm_lu as solve_lm


@njit
def create_approximation(model):
    window = 6
    step = 4
    amount_complete = int((len(model) + step - window) / step)
    bound = amount_complete * step + window - step
    size = amount_complete + 1 if len(model) > bound else amount_complete
    ans = np.zeros((size, 6))
    for i in range(0, amount_complete):
        start = i * step
        current_times = [0.0] * window
        current_pressures = [0.0] * window
        for j in range(0, window):
            current_times[j] = model[start + j][0]
            current_pressures[j] = model[start + j][1]
        ans[i] = find_part(current_times, current_pressures)
    if len(model) > bound:
        current_times = [0.0] * window
        current_pressures = [0.0] * window
        start = len(model) - window
        for j in range(0, window):
            current_times[j] = model[start + j][0]
            current_pressures[j] = model[start + j][1]
        ans[size - 1] = find_part(current_times, current_pressures)
    return ans


@njit
def find_part(t, p):
    def fun(coefs, args, constants):
        ans = [0.0] * len(args)
        for i in range(0, len(args)):
            log_time = math.log(args[i])
            ans[i] = calc_polynomial(coefs, log_time, 4)
        return ans
    def jac(coefs, args, constants):
        ans = np.zeros((len(args), len(coefs)))
        for i in range(0, len(coefs)):
            orig = coefs[i]
            delta = orig * 0.1
            if delta == 0:
                delta = 0.1
            coefs[i] = orig + delta
            plus_values = [0.0] * len(args)
            for j in range(0, len(args)):
                log_time = math.log(args[j])
                plus_values[j] = calc_polynomial(coefs, log_time, 4)
            coefs[i] = orig - delta
            minus_values = [0.0] * len(args)
            for j in range(0, len(args)):
                log_time = math.log(args[j])
                minus_values[j] = calc_polynomial(coefs, log_time, 4)
            coefs[i] = orig
            for j in range(0, len(plus_values)):
                ans[j][i] = (plus_values[j] - minus_values[j]) / (2.0 * delta)
        return ans
    first = [1.0, 1.0, 1.0, 1.0, 1.0]
    sol = solve_lm(fun, jac, first, 50, 1E-15, 0.0, p, args=(t))
    calc = np.zeros(len(sol) + 1)
    calc[0] = t[max(int(round(len(t) / 2)) - 1, 0)]
    for i in range(0, len(sol)):
        calc[i + 1] = sol[i]
    return calc


@njit
def calc_polynomial(coefs, x, pow):
    value = 0
    for j in range(0, pow):
        value += coefs[j] * math.pow(x, (pow - j))
    value += coefs[pow]
    return value


@njit
def calc_approximation(approx, t):
    pair = find_closest(approx, t)
    log_time = math.log(t)
    return calc_polynomial(np.delete(pair, 0), log_time, 4)


@njit
def find_closest(array, value):
    if value < array[0][0]:
        return array[0]
    if value > array[len(array) - 1][0]:
        return array[len(array) - 1]
    lo = 0
    hi = len(array) - 1
    while lo <= hi:
        mid = int((hi + lo) / 2)
        if value < array[mid][0]:
            hi = mid - 1
        elif value > array[mid][0]:
            lo = mid + 1
        else:
            return array[mid]
    return array[lo] if (array[lo][0] - value) < (value - array[hi][0]) else array[hi]
