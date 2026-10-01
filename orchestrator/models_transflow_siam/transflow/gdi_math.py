import math
import numpy as np
from numba import njit

import models_transflow_siam.transflow.constants as const


@njit
def bessel_i0(x):
    ax = abs(x)
    if ax < 3.75:
        y = x / 3.75
        y = y * y
        ans = 1.0 + y * (3.5156229 + y * (
                3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
    else:
        y = 3.75 / ax
        ans = (math.exp(ax) / math.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (
                -0.157565e-2 + y * (0.916281e-2 + y * (
                -0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))))

    return ans


@njit
def bessel_i1(x):
    ax = abs(x)
    if ax < 3.75:
        y = x / 3.75
        y = y * y
        ans = ax * (0.5 + y * (0.87890594 + y * (
                0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))))
    else:
        y = 3.75 / ax
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
        ans *= (math.exp(ax) / math.sqrt(ax))
    if x < 0.0:
        ans = -ans
    return ans


@njit
def bessel_k0(x):
    if x <= 2.0:
        y = x * x / 4.0
        ans = (-math.log(x / 2.0) * bessel_i0(x)) + (-0.57721566 + y * (0.42278420 + y * (
                0.23069756 + y * (0.3488590e-1 + y * (0.262698e-2 + y * (0.10750e-3 + y * 0.74e-5))))))
    else:
        y = 2.0 / x
        ans = (math.exp(-x) / math.sqrt(x)) * (1.25331414 + y * (-0.7832358e-1 + y * (
                0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y * (-0.251540e-2 + y * 0.53208e-3))))))
    return ans


@njit
def bessel_k1(x):
    if x <= 2.0:
        y = x * x / 4.0
        ans = (math.log(x / 2.0) * bessel_i1(x)) + (1.0 / x) * (1.0 + y * (0.15443144 + y * (
                -0.67278579 + y * (-0.18156897 + y * (-0.1919402e-1 + y * (-0.110404e-2 + y * (-0.4686e-4)))))))
    else:
        y = 2.0 / x
        ans = (math.exp(-x) / math.sqrt(x)) * (1.25331414 + y * (0.23498619 + y * (
                -0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2 + y * (0.325614e-2 + y * (-0.68245e-3)))))))
    return ans


@njit
def bessel_ki1(x):
    a = [1.25331414, 0.11190289, 0.02576646, 0.00933994, 0.00417454, 0.00163271, 0.00033934]
    ans = 0
    if 7 <= x < 40:
        for i in range(0, 7):
            ans = ans + pow(-1, i) * a[i] * pow(x / 7, -i) / math.sqrt(x) / math.exp(x)
        return ans
    elif x < 7:
        n_double = const.INTEGRAL_QUALITY
        dz = (7 - x) / n_double
        for i in range(0, const.INTEGRAL_QUALITY):
            z = dz * i + dz / 2 + x
            ans = ans + bessel_k0(z) * dz
        return ans + 4.001E-4
    return ans


@njit
def stephest_left(n):
    sum_p = 0
    N = const.STEFEST_QUALITY
    half_n = N / 2
    start_p = int((n + 1) / 2)
    end_p = min(n, N / 2) + 1
    for p in range(int(start_p), int(end_p), 1):
        sum_p = sum_p + (math.pow(-1, n + half_n) * math.pow(p, half_n) * factorial(2 * p)) / (
                factorial(half_n - p) * factorial(p) * factorial(p - 1) * factorial(
            n - p) * factorial(2 * p - n))
    return sum_p


@njit
def min(a, b):
    if a <= b:
        return a
    else:
        return b


@njit
def factorial(n):
    factorial = 1
    for i in range(1, int(n + 1), 1):
        factorial = factorial * i
    return factorial


@njit
def auto_time(lower_bound, upper_bound, quality, step=None):
    if step is None:
        return auto_time_log(lower_bound, upper_bound, quality)
    else:
        return auto_time_log_lin(lower_bound, upper_bound, quality, step)


@njit
def auto_time_log(lower_bound, upper_bound, quality):
    if lower_bound >= upper_bound:
        lower_bound = math.pow(10, math.log10(upper_bound) - 1)
    buffer = []
    upper_bound = math.pow(10, math.log10(upper_bound) + 1 / quality)
    key = 0
    is_fill_log = True
    while is_fill_log:
        buffer.append(math.pow(10, math.log10(upper_bound) - key / quality))
        key += 1
        is_fill_log = math.pow(10, math.log10(upper_bound) - key / quality) > lower_bound
    a_time = np.zeros(len(buffer) + 2)
    a_time[0] = math.pow(10, math.log10(lower_bound) - 1 / quality)
    a_time[1] = lower_bound
    for i in range(0, len(buffer)):
        a_time[i + 2] = buffer[len(buffer) - 1 - i]
    return a_time


@njit
def auto_time_log_lin(lower_bound, upper_bound, log_quality, step):
    if lower_bound >= upper_bound:
        lower_bound = math.pow(10, math.log10(upper_bound) - 1)
    buffer = []
    upper_bound = math.pow(10, math.log10(upper_bound) + 1 / log_quality)
    key = 0
    lin_value = math.pow(10, math.log10(upper_bound) - (key + 1) / log_quality) + step
    is_fill_log = True
    while is_fill_log:
        log_value = math.pow(10, math.log10(upper_bound) - key / log_quality)
        log_plus = math.pow(10, math.log10(upper_bound) - (key + 1) / log_quality)
        if log_value - log_plus > step:
            while lin_value > log_plus:
                buffer.append(lin_value)
                lin_value -= step
        if log_value - log_plus < step:
            buffer.append(log_value)
        key += 1
        is_fill_log = log_value > lower_bound
    buffer.pop()
    a_time = np.zeros(len(buffer) + 2)
    a_time[0] = math.pow(10, math.log10(lower_bound) - 1 / log_quality)
    a_time[1] = lower_bound
    for i in range(0, len(buffer)):
        a_time[i + 2] = buffer[len(buffer) - 1 - i]
    return a_time
